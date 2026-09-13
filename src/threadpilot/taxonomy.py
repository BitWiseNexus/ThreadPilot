"""Intent taxonomy for AmazonHelp, derived from the data (Phase 2).

## How this was produced, and why it is not a guess

Writing a plausible support taxonomy by hand would have taken ten minutes and
nobody would have noticed. The problem is that the classifier would then be
evaluated against *my guess* about AmazonHelp's traffic rather than against its
actual traffic (decision_log.md D12). So:

1. 8,000 English thread-opening customer messages were embedded with MiniLM.
2. KMeans was swept over k=4..14. **Silhouette was useless as a selector** -
   0.051 to 0.065 across the whole range, a spread of 0.014 - because short
   support text has no clean geometric cluster structure. Reported in
   `eval/results/taxonomy_k_sweep.md` as evidence that the automatic criterion
   did not discriminate, rather than as justification for a choice it did not
   make.
3. k was chosen by **interpretability** instead: at k=10 every cluster is
   nameable and distinct, while k=9 collapses "complaints about support quality"
   into a generic service bucket.
4. Clusters were named by reading exemplars, NOT term lists. This mattered:
   cluster 1's top terms were `amazon, amazonindia, amazon pay, india`, which
   reads like a payments intent, but its actual messages are generalised brand
   rage ("Such a #PoorService", "never using Amazon again"). Naming it from
   vocabulary would have produced a fictional intent.
5. Clusters 0 and 1 were merged (both generalised complaint); the rest stand.

Per-cluster evidence - sizes, distinctive terms, nearest- AND
farthest-from-centroid exemplars - is committed at
`eval/results/taxonomy_clusters.md` so the naming step is reviewable rather
than asserted.

## Deviation from the plan, stated

`docs/requirements.md` said "~6-9 intents". This lands on **10 plus `other`**.
The alternative was folding `unresolved_followup` into `service_complaint`,
which would have created a single 26% catch-all - a junk drawer that inflates
the majority-class baseline and hides a genuinely distinct triage case. More
classes costs statistical power at n=200 (roughly 18 golden examples per class,
so wide confidence intervals), and that cost is reported rather than avoided by
picking a convenient taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Disposition(str, Enum):
    """Default triage posture for an intent, before any per-message evidence."""

    ALWAYS_ESCALATE = "always_escalate"   # a wrong auto-reply causes real harm
    USUALLY_ESCALATE = "usually_escalate"  # needs account data or judgement
    AUTO_ELIGIBLE = "auto_eligible"       # a grounded reply is normally safe


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    examples: tuple[str, ...]        # verbatim from the subsample
    boundary: str                    # how to tell it from its nearest neighbour
    disposition: Disposition
    cluster: int | None              # source cluster, for auditability
    approx_share: float              # observed share of opener traffic


INTENTS: tuple[Intent, ...] = (
    Intent(
        name="delivery_delay",
        definition="An order has not arrived yet, is running late, or the "
                   "customer is asking when it will arrive. The package is "
                   "still in transit or overdue - nothing claims it was "
                   "delivered.",
        examples=(
            "my order was scheduled to arrive yesterday and didnt. I was not "
            "informed that the parcel was late",
            "i just placed an order with deliver next day by 1pm and its "
            "saying it wont arrive till monday?",
            "I ordered something via Amazon Prime midday on Friday 24th "
            "November and still have no goods.",
        ),
        boundary="vs delivery_failure: this one is LATE; delivery_failure is "
                 "marked delivered but absent, or delivered to the wrong "
                 "place. vs prime_membership: if the complaint is that a "
                 "delivery was slow, it is delivery_delay even when the word "
                 "'Prime' appears - prime_membership is about the membership "
                 "itself.",
        disposition=Disposition.AUTO_ELIGIBLE,
        cluster=3, approx_share=0.124,
    ),
    Intent(
        name="delivery_failure",
        definition="The delivery went wrong: marked delivered but not "
                   "received, left in an unsafe place, given to a neighbour, "
                   "or a complaint about the courier's behaviour.",
        examples=(
            "I've not received a parcel that's apparently been delivered?",
            "My package was delivered to my neighbor.",
            "They drove by my house three times without delivering",
        ),
        boundary="vs delivery_delay: something claims completion here - "
                 "'delivered' but absent or misplaced - whereas delivery_delay "
                 "is still pending. vs order_investigation: if the customer "
                 "supplies an order ID and asks for a lookup, prefer "
                 "order_investigation.",
        disposition=Disposition.USUALLY_ESCALATE,
        cluster=6, approx_share=0.113,
    ),
    Intent(
        name="order_investigation",
        definition="A specific order that cannot be answered without looking "
                   "it up: the customer supplies an order ID, or explicitly "
                   "asks the brand to investigate one order. Includes "
                   "unexpected cancellations and wrong items.",
        examples=(
            "please can you look into this order: 206-9303243-6219526 I paid "
            "for next day delivery and still not received!!",
            "i have just seen this order was cancelled and not by me. VERY "
            "UNHAPPY Order #701-5406600-7964269",
            "my order got delivered to someone else. Order Id: "
            "171-6576778-6197962",
        ),
        boundary="The distinguishing feature is REQUIRING PRIVATE DATA, not "
                 "the topic. A late order described generally is "
                 "delivery_delay; the same complaint with an order number and "
                 "'please look into it' is order_investigation.",
        disposition=Disposition.USUALLY_ESCALATE,
        cluster=9, approx_share=0.098,
    ),
    Intent(
        name="refund_billing",
        definition="Money: a refund not received, an unexpected or wrong "
                   "charge, cashback, price disputes, or a return that has not "
                   "produced a refund.",
        examples=(
            "Seller arranged for my return item to be picked up and now I "
            "can't get my refund.",
            "is not refunding me since a month for order no.",
            "I sent the product back & still haven't [been refunded]",
        ),
        boundary="vs order_investigation: if the ask is about MONEY owed, it "
                 "is refund_billing even when an order ID is present. vs "
                 "prime_membership: a Prime subscription charge is "
                 "refund_billing if the customer disputes the charge, "
                 "prime_membership if they ask about benefits or cancellation.",
        disposition=Disposition.ALWAYS_ESCALATE,
        cluster=2, approx_share=0.116,
    ),
    Intent(
        name="item_condition",
        definition="The item or its packaging arrived in an unacceptable "
                   "state: damaged goods, crushed boxes, or complaints about "
                   "excessive or inadequate packaging.",
        examples=(
            "disappointed to receive my parcel like this, especially with "
            "Christmas presents in which have been damaged",
            "your packaging get worse by the day: this large box for this "
            "single item is ridiculous!",
            "WTF! Why would you ship my purchase like this!!! It was a gift",
        ),
        boundary="vs delivery_failure: the package ARRIVED here; the problem "
                 "is its state, not its whereabouts.",
        disposition=Disposition.AUTO_ELIGIBLE,
        cluster=8, approx_share=0.100,
    ),
    Intent(
        name="product_digital",
        definition="Amazon's own products and digital services: the app, "
                   "Alexa, Echo, Kindle, Fire TV, Prime Video, Amazon Music - "
                   "including content availability by region.",
        examples=(
            "I would love to watch but it's 'currently unavailable to watch "
            "in' my location. Please explain",
            "can you please sort out your app, it's dreadful. After watching "
            "adverts it says 'content not available'",
            "When you are going to launch Amazon Prime Music in India ?",
        ),
        boundary="vs prime_membership: this is about the CONTENT or DEVICE "
                 "working; prime_membership is about the subscription that "
                 "grants access.",
        disposition=Disposition.AUTO_ELIGIBLE,
        cluster=5, approx_share=0.093,
    ),
    Intent(
        name="prime_membership",
        definition="The Prime subscription itself: what it includes, "
                   "membership charges, renewal, cancellation, or the benefit "
                   "not being honoured as a membership matter.",
        examples=(
            "why spend money on prime to receive something days later",
            "I don't understand why Prime isn't shipping promptly anymore.",
            "Used my Prime Membership again today for an order, not due to "
            "arrive until Tuesday, how can this be One Day delivery",
        ),
        boundary="OVERLAPS HEAVILY with delivery_delay - most observed Prime "
                 "messages are really complaints about slow delivery. Rule: "
                 "label delivery_delay unless the customer questions the "
                 "membership's value, cost, or terms. This is the single "
                 "hardest boundary in the taxonomy and is expected to be the "
                 "main source of confusion in the Phase 6 matrix.",
        disposition=Disposition.AUTO_ELIGIBLE,
        cluster=4, approx_share=0.091,
    ),
    Intent(
        name="account_security",
        definition="Account access or compromise: cannot log in, password "
                   "problems, account hacked, unauthorised access, or "
                   "fraudulent charges by a third party.",
        examples=(
            "someone has hacked my account and changed the login details as I "
            "just received an email confirming this",
            "account hacked, my email account linked with amazon changed "
            "without authorisation. Need immediate help.",
            "I am unable to log in since a month Plz help",
        ),
        boundary="vs refund_billing: an unauthorised charge by a THIRD PARTY "
                 "is account_security (the account is compromised); a wrong "
                 "charge by Amazon is refund_billing.",
        disposition=Disposition.ALWAYS_ESCALATE,
        # No clean cluster of its own: only ~2.5% of opener traffic, so KMeans
        # scattered it. Included anyway because it is the highest-stakes intent
        # in the set - a system with no label for "my account was hacked" has a
        # hole exactly where automation is most dangerous. The cost is that the
        # golden set must oversample it, which makes the golden set
        # deliberately non-representative (reported in Phase 3).
        cluster=None, approx_share=0.025,
    ),
    Intent(
        name="service_complaint",
        definition="Dissatisfaction with Amazon's support or the brand "
                   "generally, with no specific actionable request: 'your "
                   "customer service is awful', 'never using Amazon again'.",
        examples=(
            "your customer service sucks ass",
            "why is your customer service so awful",
            "Such a #PoorService given by Amazon. They never give us proper "
            "reply on call",
        ),
        boundary="vs unresolved_followup: that one is chasing a SPECIFIC "
                 "unresolved issue; this is a general grievance with nothing "
                 "concrete to action. If you can name the underlying order or "
                 "problem, it is not service_complaint.",
        # Merged from clusters 0 and 1 - both generalised complaint. Cluster 1
        # was nearly mis-named "amazon pay / India" from its term list before
        # its actual messages were read.
        disposition=Disposition.ALWAYS_ESCALATE,
        cluster=0, approx_share=0.164,
    ),
    Intent(
        name="unresolved_followup",
        definition="Chasing a problem already raised: no resolution, repeated "
                   "contact, or an explicit 'still waiting' about a known "
                   "issue.",
        examples=(
            "AGAIN not left in my safe place!! I'm getting really frustrated "
            "with this!!",
            "Still not got this ...",
            "can you give me more information than this I've been waiting in? "
            "All it's said since yesterday",
        ),
        boundary="vs service_complaint: a concrete unresolved issue exists "
                 "here. vs the delivery intents: use this when the salient "
                 "fact is that contact has ALREADY failed, not the underlying "
                 "topic.",
        disposition=Disposition.USUALLY_ESCALATE,
        cluster=7, approx_share=0.100,
    ),
    Intent(
        name="other",
        definition="Anything the taxonomy does not cover, including messages "
                   "too vague to classify. Deliberately kept as an explicit "
                   "bucket rather than forcing every message into a named "
                   "intent.",
        examples=(
            "(no fixed examples - this is the residual class)",
        ),
        boundary="Use only when no named intent applies. A message that is "
                 "merely ambiguous between two named intents is NOT other - "
                 "pick the better fit and flag the ambiguity in the golden set.",
        disposition=Disposition.USUALLY_ESCALATE,
        cluster=None, approx_share=0.0,
    ),
)

BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}
INTENT_NAMES: tuple[str, ...] = tuple(i.name for i in INTENTS)

# Intents where a wrong auto-reply causes real harm, so no amount of model
# confidence may authorise automation. Deliberately MINIMAL: over-stuffing this
# set caps achievable auto-handle coverage by construction and would let the
# system look safe by doing nothing (requirements.md section 6).
#   - account_security: fraud and compromise must reach a human quickly
#   - refund_billing:   financial commitments must not be made by a bot
#   - service_complaint: auto-replying to "your service sucks" risks inflaming
# Together ~30% of opener traffic, leaving ~70% eligible for automation.
ALWAYS_ESCALATE: tuple[str, ...] = tuple(
    i.name for i in INTENTS if i.disposition is Disposition.ALWAYS_ESCALATE
)


def describe(name: str) -> str:
    """Human-readable block for CLI output and prompt construction."""
    i = BY_NAME[name]
    return (f"{i.name} ({i.disposition.value}, ~{i.approx_share:.0%} of traffic)\n"
            f"  {i.definition}\n  Boundary: {i.boundary}")


def prompt_block() -> str:
    """The taxonomy as the classifier prompt sees it.

    Includes boundary notes, not just names: the observed failure mode in
    similar tasks is confusion between adjacent intents, and the boundaries are
    exactly the information that resolves it.
    """
    lines = []
    for i in INTENTS:
        lines.append(f"- {i.name}: {i.definition}")
        if i.name != "other":
            lines.append(f"    boundary: {i.boundary}")
    return "\n".join(lines)
