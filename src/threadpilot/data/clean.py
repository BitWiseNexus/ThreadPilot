"""Text normalisation and reply classification.

This module decides what is admitted to the retrieval index, so it determines
what "grounded in how the brand has historically resolved similar issues" can
possibly mean. It has been through two rounds of measured revision.

## v1 -> v2, and why

v1 classified replies five ways and scored **Cohen kappa 0.20** against an
independent LLM rater (scripts/validate_reply_proxy.py) - i.e. barely better
than chance. Reading the disagreements showed three distinct errors, all of them
mine rather than the rater's:

1. **A blunt length gate was the dominant bug.** Any reply whose body fell under
   50 characters was called an empty pleasantry. That threw away
   "Has the order been marked as dispatched yet?" (43 chars - a useful
   diagnostic question) and "You can check your subscriptions here <link>"
   (37 chars - a complete answer). Usefulness is not a function of length, so
   v2 detects content *signals* instead and applies no length gate.

2. **"Channel switch" was too coarse.** v1 discarded every "DM us" reply.
   But "Can you DM us your booking reference and we'll take a look" names
   exactly what the brand needs and is the brand's genuine handling pattern for
   account-specific issues. It is real precedent - and, usefully, a strong
   *escalate* signal, since a message the brand historically took private is a
   message a human should handle. v2 splits HANDOFF_WITH_ASK (usable) from
   CHANNEL_SWITCH_BARE (not). This reverses decision D8; see the decision log.

3. **Source truncation was invisible.** Some replies in the dataset are cut off
   mid-clause ("...your feedback will be passed to"). v1 counted those as
   complete self-contained answers. v2 detects them.

All heuristics here are my own, not borrowed. None of them is trusted: the
validation script re-measures agreement, and the report carries the resulting
kappa next to any number that depends on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")

# Agent sign-offs. Brands in this dataset use at least five different styles,
# and missing any of them is not cosmetic -- an unstripped signature leaves the
# body ending in letters, which the truncation detector then reads as a reply cut
# off mid-clause.
#
# That is not hypothetical: with only the caret/asterisk/dash-name styles handled,
# SpotifyCares ("/AY") and comcastcares ("-DR") measured 53% and 54% "truncated"
# against AmazonHelp's 4%, which pushed SpotifyCares from 2nd to 9th in the brand
# table. A formatting detail silently drove a project-level decision.
#
# Styles covered:  ^AS   *KittyG   ~JD   /AY   -DR   - Tom
# The trailing token must look like initials ([A-Z]{1,4}) or a capitalised name
# ([A-Z] then letters), so lowercase tails such as "and/or" or "check in/out"
# are left alone.
SIGNATURE_RE = re.compile(
    r"(?:[\^*~/]|\s[-–])\s?(?:[A-Z]{1,4}|[A-Z][A-Za-z]{1,14})\s*$")

# "(1/2)", "2/2", "1/3". A part 2+ is a sentence fragment, not a standalone reply.
MULTIPART_RE = re.compile(r"[(\[]?\s*(\d{1,2})\s*/\s*(\d{1,2})\s*[)\]]?")

CHANNEL_SWITCH_RE = re.compile(
    r"\bd\.?m\.?s?\b|direct message|\bp\.?m\.?\b|private message|"
    r"check your (?:inbox|messages)|(?:send|shoot) us a (?:message|note)|"
    r"message us|we(?:'ve| have) (?:responded|replied|sent).{0,20}(?:dm|message)|"
    r"follow (?:us|and)\b|following you|private(?:ly)? (?:secured )?link|"
    r"\bcall us\b|(?:call|contact) (?:us )?(?:at|on) ?\+?\d|1[-\s]?8\d\d|"
    r"email us|contact (?:my|our) colleagues|"
    r"(?:can'?t|cannot|unable to) (?:discuss|help).{0,25}(?:here|publicly|twitter)",
    re.IGNORECASE,
)

# A handoff that names what the brand actually needs is precedent, not a brush-off.
INFO_REQUEST_RE = re.compile(
    r"booking (?:reference|ref|number)|record locator|reservation number|"
    r"confirmation (?:number|code)|order (?:number|id|details)|"
    r"tracking number|account (?:number|details|email)|"
    r"(?:full )?name|email address|phone number|post ?code|zip ?code|"
    r"serial number|imei|last (?:four|4)|date of (?:birth|travel)|"
    r"flight number|which (?:order|item|account|device)",
    re.IGNORECASE,
)

# A URL plus a directive is a specific, actionable answer.
DIRECTIVE_RE = re.compile(
    r"\b(?:contact|reach out|get in touch|go to|head (?:to|over)|click|"
    r"check (?:out|here|your)|visit|see|register|sign up|submit|apply|"
    r"fill (?:out|in)|report (?:it|this)|use (?:this|the)|track|find|browse|"
    r"more info|details|available|looking for|here)\b",
    re.IGNORECASE,
)

# Instructional / explanatory content: the marks of an actual answer.
INSTRUCTION_RE = re.compile(
    r"\b(?:you (?:can|should|will|need|may|could)|please (?:try|check|ensure|note)|"
    r"try (?:again|to|restarting)|make sure|we (?:can|will|have|do|don't|cannot)|"
    r"it (?:should|will|can|may)|this (?:is|means|happens)|"
    r"i(?:'d| would) recommend|recommend|the (?:order|policy|item|issue|delay)|"
    r"has been|have been|is (?:not )?(?:available|eligible|possible)|"
    r"unfortunately|because|due to|once|after|within \d|\bin \d+ ?(?:hours|days|"
    r"business days|minutes))\b",
    re.IGNORECASE,
)

# Pure pleasantry with nothing else. Matched against the WHOLE body, no length gate.
BARE_ACK_RE = re.compile(
    r"^(?:oh no|so sorry|sorry (?:to hear|about|for)(?: that| this)?|"
    r"we(?:'re| are) (?:so )?sorry(?: to hear| about that)?|apologies|"
    r"that'?s (?:not|never) what we (?:like to hear|want)|we hear you|"
    r"thanks(?: so much)?(?: for (?:reaching out|letting us know|your patience|"
    r"the (?:info|update)))?|thank you|noted|we understand|"
    r"we(?:'ll| will) (?:look into (?:this|it)|check(?: this)?)|"
    r"sending (?:you )?(?:good vibes|our best)|hope (?:you|this)[\w\s]{0,25})"
    r"[\s.!,:;)–-]*$",
    re.IGNORECASE,
)

TERMINAL_PUNCT = tuple(".!?…:;)\"'")
TRUNCATION_MIN_CHARS = 105

# Truncation is "ends mid-clause", not merely "ends without a full stop". Plenty
# of real replies just skip the final period ("...or an incognito window. Keep us
# posted"), and calling those truncated was inflating the rate for whole brands.
# A dangling function word is the reliable signal: English sentences do not end
# on a preposition, article, conjunction or auxiliary.
DANGLING_WORDS = frozenset("""
a an the to of for with in on at by from as and or but if it is are was were
be been being will would can could should may might must have has had do does
did we you your our their his her its this that these those there here about
into onto over under out up off than then so because while when where which
who whom what how any some no not
""".split())


class ReplyKind(str, Enum):
    """What kind of precedent a historical brand reply provides."""

    SELF_CONTAINED = "self_contained"        # answers or explains directly
    LINK_REFERRAL = "link_referral"          # answers by naming a resource
    DIAGNOSTIC_ASK = "diagnostic_ask"        # asks a specific progressing question
    HANDOFF_WITH_ASK = "handoff_with_ask"    # to DM, but names what is needed
    CHANNEL_SWITCH_BARE = "channel_switch_bare"  # "DM us" - no answer
    BARE_ACK = "bare_ack"                    # polite, empty
    FRAGMENT = "fragment"                    # part 2+ of a split reply
    TRUNCATED = "truncated"                  # cut off mid-clause in the source

    @property
    def usable_as_precedent(self) -> bool:
        return self in _USABLE

    @property
    def implies_escalation(self) -> bool:
        """A message the brand historically took private is one a human handled.

        Used as an escalation signal in Phase 4 rather than being thrown away -
        one of the more useful things to come out of the v1 validation failure.
        """
        return self in (ReplyKind.HANDOFF_WITH_ASK, ReplyKind.CHANNEL_SWITCH_BARE)


_USABLE = frozenset({
    ReplyKind.SELF_CONTAINED, ReplyKind.LINK_REFERRAL,
    ReplyKind.DIAGNOSTIC_ASK, ReplyKind.HANDOFF_WITH_ASK,
})


@dataclass(frozen=True)
class ReplyFeatures:
    kind: ReplyKind
    has_url: bool
    has_signature: bool
    has_info_request: bool
    part: int | None
    n_parts: int | None
    n_chars: int

    @property
    def usable(self) -> bool:
        return self.kind.usable_as_precedent


def strip_signature(text: str) -> str:
    return SIGNATURE_RE.sub("", text).strip()


def multipart_info(text: str) -> tuple[int | None, int | None]:
    """(part, n_parts) for a split reply, else (None, None).

    Only trusts small plausible splits (total 2-9, part <= total) so that "24/7"
    and "1/2 price" are not misread as part markers.
    """
    for m in MULTIPART_RE.finditer(text):
        part, total = int(m.group(1)), int(m.group(2))
        if 2 <= total <= 9 and 1 <= part <= total:
            return part, total
    return None, None


def normalize(text: str, mask_urls: bool = True, mask_handles: bool = True) -> str:
    """Canonical form used for embedding and display."""
    s = str(text)
    s = strip_signature(s)
    if mask_urls:
        s = URL_RE.sub(" <url> ", s)
    if mask_handles:
        s = HANDLE_RE.sub(" <user> ", s)
    s = MULTIPART_RE.sub(" ", s)
    return WHITESPACE_RE.sub(" ", s).strip()


def _body_punct(text: str) -> str:
    """Content with signature, urls, handles and part markers removed, but with
    terminal punctuation PRESERVED.

    Kept separate from _body() because truncation detection needs to see whether
    the reply ends in punctuation. Collapsing the two caused a silent bug: with
    punctuation already stripped, looks_truncated() could never observe a
    terminal '?' or '.', so every reply over the length threshold was
    misclassified as truncated. Caught by the sanity cases in
    tests/test_clean.py, which is why those exist.
    """
    b = normalize(text, mask_urls=True, mask_handles=True)
    return b.replace("<url>", "").replace("<user>", "").strip()


def _body(text: str) -> str:
    """_body_punct() with trailing punctuation removed, for emptiness and length
    checks where '!!!' should not count as content."""
    return _body_punct(text).strip(" .,!?:;-–")


def looks_truncated(body_punct: str) -> bool:
    """Source-side truncation: a long reply that stops mid-clause.

    Requires all three of:
      * length at/above the threshold (short replies are terse, not cut off),
      * no terminal punctuation, and
      * a dangling final function word, or a trailing comma.

    The third condition is what makes this usable. Without it, any reply merely
    missing its final full stop was flagged, which produced 53%/54% "truncated"
    for two brands against 4% for another and reordered the brand table.

    Takes the punctuation-PRESERVING body; passing the punctuation-stripped body
    silently makes the second condition always true past the length threshold.
    """
    b = body_punct.rstrip()
    if len(b) < TRUNCATION_MIN_CHARS:
        return False
    if b.endswith(TERMINAL_PUNCT):
        return False
    if b.endswith(","):
        return True
    last = b.split()[-1].strip(".,!?;:\"'()").lower() if b.split() else ""
    return last in DANGLING_WORDS


def classify_reply(text: str) -> ReplyFeatures:
    """Eight-way classification of a historical brand reply.

    Check order encodes precedence deliberately:
      fragment/truncated first  - an incomplete reply is unusable whatever it says
      handoff before link       - "here's a link, but also DM us" is still a handoff
      content signals before bare_ack - a short answer is an answer
    """
    raw = str(text)
    has_url = bool(URL_RE.search(raw))
    has_sig = bool(SIGNATURE_RE.search(raw))
    part, n_parts = multipart_info(raw)
    body_punct = _body_punct(raw)
    body = _body(raw)
    has_info_req = bool(INFO_REQUEST_RE.search(raw))
    has_directive = bool(DIRECTIVE_RE.search(raw))
    has_instruction = bool(INSTRUCTION_RE.search(raw))
    is_question = "?" in body

    # Any part of a split reply, INCLUDING part 1, is incomplete on its own -
    # "report this to our support team 1/3" is the opening of an answer, not an
    # answer. Reassembling parts into whole replies is the better fix and is a
    # Phase 4 task (see docs/memory.md TODOs); until then, excluding them is the
    # conservative choice and it is reported, not hidden.
    if n_parts is not None and n_parts > 1:
        kind = ReplyKind.FRAGMENT
    elif looks_truncated(body_punct):
        kind = ReplyKind.TRUNCATED
    elif CHANNEL_SWITCH_RE.search(raw):
        # Usable only if it names what is needed or otherwise instructs.
        kind = (ReplyKind.HANDOFF_WITH_ASK
                if (has_info_req or is_question or has_instruction)
                else ReplyKind.CHANNEL_SWITCH_BARE)
    elif has_url and has_directive:
        kind = ReplyKind.LINK_REFERRAL
    elif is_question and (has_info_req or has_instruction or len(body) > 25):
        kind = ReplyKind.DIAGNOSTIC_ASK
    elif BARE_ACK_RE.match(body) or len(body) < 12:
        kind = ReplyKind.BARE_ACK
    elif has_instruction or has_directive or len(body) >= 40:
        kind = ReplyKind.SELF_CONTAINED
    else:
        kind = ReplyKind.BARE_ACK

    return ReplyFeatures(
        kind=kind, has_url=has_url, has_signature=has_sig,
        has_info_request=has_info_req, part=part, n_parts=n_parts,
        n_chars=len(body),
    )


def has_mojibake(text: str) -> bool:
    """U+FFFD appears in ~0.004% of source texts - pre-existing scrape damage in
    the Kaggle dataset, not introduced here. Flagged, never silently repaired:
    guessing the original character would be fabrication."""
    return "�" in str(text)
