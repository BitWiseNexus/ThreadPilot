"""Reply-classifier tests.

Every case here is a REAL example from the dataset that the v1 classifier got
wrong, taken from the disagreement list produced by
scripts/validate_reply_proxy.py. They exist because this module decides what is
admitted to the retrieval index: a silent regression here would not crash
anything, it would quietly change what "grounded in brand precedent" means and
shift every downstream quality number.
"""

from __future__ import annotations

import pytest

from threadpilot.data.clean import (
    ReplyKind, classify_reply, has_mojibake, looks_truncated, multipart_info,
    normalize, strip_signature,
)


# --------------------------------------------------------------------------
# Cases the v1 classifier got wrong. Each is a real reply from twcs.csv.
# --------------------------------------------------------------------------
USABLE_CASES = [
    # v1 bug 1: blunt length gate dumped short-but-useful replies into bare_ack.
    ("Has the order been marked as dispatched yet?", "43-char diagnostic question"),
    ("You can check your subscriptions here https://t.co/abc", "37-char link answer"),
    ("Sorry! This link will be what you're looking for: https://t.co/abc",
     "apology + specific resource"),
    # v1 bug 2: all channel switches discarded, including specific ones.
    ("We're concerned to hear this. Can you please DM us your booking reference, "
     "email address and phone number?", "handoff naming exactly what is needed"),
    ("I would recommend running a virus scan if you've clicked anything suspicious.",
     "concrete recommendation"),
    ("The baggage policy changed some time ago. We'll remind our staff at "
     "Amsterdam airport to follow the right guidelines.", "explains + commits"),
]

UNUSABLE_CASES = [
    ("We're following you, so feel free to DM us anytime. We're here 24/7. Many thanks.",
     "bare channel switch - names nothing"),
    ("We agree this isn't what you should expect when travelling with us, "
     "Christian. Please be assured your feedback will be passed to",
     "source-truncated mid-clause"),
    ("Oh no! So sorry to hear that!", "pure pleasantry"),
    ("The orders are shipped and mapped for the delivery based on the estimates "
     "provided during checkout. (2/2)", "part 2 of a split reply"),
    ("Thanks for reaching out!", "pure pleasantry"),
]


@pytest.mark.parametrize("text,why", USABLE_CASES)
def test_usable_replies_are_admitted(text: str, why: str) -> None:
    f = classify_reply(text)
    assert f.usable, f"should be usable ({why}) but got {f.kind.value}"


@pytest.mark.parametrize("text,why", UNUSABLE_CASES)
def test_unusable_replies_are_excluded(text: str, why: str) -> None:
    f = classify_reply(text)
    assert not f.usable, f"should NOT be usable ({why}) but got {f.kind.value}"


def test_truncation_detection_sees_terminal_punctuation():
    """Regression test for a silent bug that would have wrecked every metric.

    _body() strips trailing punctuation for emptiness checks. An earlier version
    passed that stripped string to looks_truncated(), which then could never
    observe a terminal '?' or '.' -- so EVERY reply past the length threshold was
    classified as truncated. Nothing crashed; the numbers were just wrong.
    """
    long_complete = (
        "We're concerned to hear this, Charlotte, and we'll look into it for you "
        "right away. Can you please DM us your booking reference?"
    )
    assert len(long_complete) > 105
    assert classify_reply(long_complete).kind is not ReplyKind.TRUNCATED

    long_cut = (
        "We agree this isn't what you should expect when travelling with us, "
        "Christian. Please be assured your feedback will be passed to"
    )
    assert classify_reply(long_cut).kind is ReplyKind.TRUNCATED

    # Short replies without punctuation are not truncation - just terse.
    assert not looks_truncated("Sure thing")


def test_handoff_implies_escalation_but_bare_ack_does_not():
    """A message the brand historically took private is one a human handled, so
    it is an escalation signal (Phase 4) rather than something to discard."""
    handoff = classify_reply(
        "Can you DM us your order number and we'll take a look?")
    assert handoff.kind.implies_escalation
    assert handoff.usable  # usable AND an escalation signal

    bare = classify_reply("Please DM us.")
    assert bare.kind.implies_escalation
    assert not bare.usable

    assert not classify_reply("Oh no, so sorry!").kind.implies_escalation


def test_multipart_ignores_implausible_markers():
    """'24/7' and '1/2 price' must not be read as part markers."""
    assert multipart_info("(1/2) first half") == (1, 2)
    assert multipart_info("part 2/3 here") == (2, 3)
    assert multipart_info("We're here 24/7") == (None, None)
    assert multipart_info("no marker at all") == (None, None)


def test_signature_stripping():
    """Agent sign-offs are identity, not content; leaving them in makes two
    unrelated replies by the same agent look similar to the embedder."""
    assert strip_signature("Thanks for letting us know! ^AS") == \
        "Thanks for letting us know!"
    assert strip_signature("We can help with that *KittyG") == "We can help with that"
    # Must not eat real trailing content.
    assert strip_signature("Try restarting your device") == "Try restarting your device"


def test_normalize_masks_urls_and_handles():
    out = normalize("@115712 see https://t.co/abc for details ^TK")
    assert "<user>" in out and "<url>" in out
    assert "115712" not in out and "t.co" not in out


def test_mojibake_flagged_not_repaired():
    """U+FFFD is pre-existing damage in the Kaggle source (~0.004% of texts).
    Flagged rather than guessed at -- inventing the original character would be
    fabrication."""
    assert has_mojibake("click �Message� at the top")
    assert not has_mojibake("click Message at the top")


def test_every_kind_has_a_defined_usability():
    """Guards against a new ReplyKind being added without deciding whether it is
    admitted to the index -- the decision that silently defines grounding."""
    for kind in ReplyKind:
        assert isinstance(kind.usable_as_precedent, bool)
        assert isinstance(kind.implies_escalation, bool)


# --------------------------------------------------------------------------
# Regressions for two bugs that silently reordered the brand-selection table.
# Neither crashed anything; both just made the numbers wrong.
# --------------------------------------------------------------------------
SIGNATURE_STYLES = [
    ("Keep us posted /AY", "Keep us posted"),                  # SpotifyCares
    ("send us your phone#? -DR", "send us your phone#?"),      # comcastcares
    ("Thanks for letting us know! ^AS", "Thanks for letting us know!"),
    ("We can help with that *KittyG", "We can help with that"),
    ("Have a great day - Tom", "Have a great day"),
]


@pytest.mark.parametrize("raw,expected", SIGNATURE_STYLES)
def test_all_observed_signature_styles_are_stripped(raw: str, expected: str) -> None:
    """Five different sign-off styles appear across brands. Missing one is not
    cosmetic: an unstripped signature leaves the body ending in letters, which
    the truncation detector reads as a reply cut off mid-clause. With only three
    styles handled, SpotifyCares ("/AY") and comcastcares ("-DR") measured 53%
    and 54% truncated against AmazonHelp's 4%, moving SpotifyCares from 2nd to
    9th in the brand table.
    """
    assert strip_signature(raw) == expected


@pytest.mark.parametrize("raw", [
    "please check in/out times",
    "available and/or refundable",
    "Try restarting your device",
])
def test_signature_stripping_does_not_eat_real_content(raw: str) -> None:
    """The widened pattern must still require initials or a capitalised name, so
    lowercase tails like 'and/or' survive."""
    assert strip_signature(raw) == raw


def test_truncation_requires_a_dangling_function_word():
    """A missing full stop is not truncation. Real replies often just skip it,
    and treating those as cut off is what inflated whole brands' rates."""
    complete = ("Hey. Could you try to refresh the page? You can also give it "
                "another shot using a private browser or an incognito window. "
                "Keep us posted")
    assert classify_reply(complete).kind is not ReplyKind.TRUNCATED

    for cut in [
        "We agree this isn't what you should expect when travelling with us, "
        "Christian. Please be assured your feedback will be passed to",
        "we deliver majority of orders by 8 PM on the Estimated Delivery Date. "
        "I request you to kindly wait till the time and",
    ]:
        assert classify_reply(cut).kind is ReplyKind.TRUNCATED, cut[-30:]


def test_all_parts_of_a_split_reply_are_fragments():
    """Part 1 is the opening of an answer, not an answer. Reassembly is the
    better fix (Phase 4); excluding is the conservative interim choice."""
    assert classify_reply(
        "Kindly report this to our support team 1/3").kind is ReplyKind.FRAGMENT
    assert classify_reply(
        "and they will investigate further. (2/3)").kind is ReplyKind.FRAGMENT
