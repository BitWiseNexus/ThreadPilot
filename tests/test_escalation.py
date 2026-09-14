"""Escalation rubric tests.

The rubric is the definition of the label the whole auto/escalate evaluation
rests on. Its predecessor was loose enough that two capable labellers diverged
34% vs 74% on the same messages (Cohen kappa 0.312), so its properties are
asserted rather than assumed.
"""

from __future__ import annotations

from threadpilot import taxonomy
from threadpilot.escalation import (
    BY_CODE, CORE_QUESTION, REASONS, rubric_block, reason_codes,
)


def test_codes_unique_and_ordered():
    codes = reason_codes()
    assert codes == tuple(sorted(codes)), "codes should read in severity order"
    assert len(set(codes)) == len(codes)


def test_every_reason_has_an_operational_test():
    """A reason without a test is a vibe. Each must say what would make it fire."""
    for r in REASONS:
        assert len(r.test) > 80, f"{r.code} test is too vague to apply"
        assert len(r.label) > 10


def test_e1_explicitly_narrows_the_test_that_broke_the_first_rubric():
    """E1's predecessor ('needs private data') fired on nearly every support
    message, so it did no work and each labeller fell back on disposition.

    The fix is the added condition: it only fires when no useful generic reply
    exists. This asserts the narrowing survives, because losing it would
    silently restore the ~0.31-kappa behaviour.
    """
    e1 = BY_CODE["E1"].test.lower()
    assert "order number" in e1, "E1 must name the counter-example that narrows it"
    assert "does not" in e1 or "not" in e1


def test_rubric_is_framed_around_the_reply_not_the_message():
    """The reframing is what made the label answerable at all."""
    q = CORE_QUESTION.lower()
    assert "unedited" in q          # the rubric emphasises it in caps
    assert "reply" in q


def test_rubric_block_warns_against_deciding_by_intent():
    """Guards the anti-circularity property: if the rubric told a labeller to
    decide by intent, the golden labels would restate gate G1 and measuring G1
    against them would be circular."""
    block = rubric_block().lower()
    assert "do not escalate merely because an intent sounds sensitive" in block
    for code in reason_codes():
        assert code.lower() in block


def test_rubric_names_no_specific_intent_as_always_escalating():
    """The rubric must not encode G1's intent list.

    G1 escalates by intent; the rubric judges per message. If the rubric named
    the same intents, the two would agree by construction and the Phase 6
    gates-off ablation would measure nothing.
    """
    block = rubric_block()
    for name in taxonomy.ALWAYS_ESCALATE:
        assert f"{name} is always" not in block
        assert f"always escalate {name}" not in block
