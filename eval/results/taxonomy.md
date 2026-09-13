# Intent taxonomy - AmazonHelp

*Generated from `src/threadpilot/taxonomy.py`. Do not edit by hand.*

10 named intents plus an explicit `other` bucket, derived from 8,000 English thread-opening customer messages by embedding (MiniLM) and KMeans, then named by reading exemplars.

## How k was chosen

Silhouette score was **uninformative**: 0.051-0.065 across k=4..14, a spread of 0.014. Short support text has no clean geometric cluster structure, so the automatic criterion simply rewarded the smallest k. It is reported in `taxonomy_k_sweep.md` as evidence that it did not discriminate, not as justification for a choice it did not make. k=10 was chosen because every cluster at that k is nameable and distinct, while k=9 collapses complaints about support quality into a generic service bucket.

## Why exemplars, not term lists

Cluster 1's top terms were `amazon, amazonindia, amazon pay, india`, which reads like a payments intent. Its actual messages are generalised brand rage ("Such a #PoorService", "never using Amazon again"). Naming from vocabulary would have produced a fictional intent, so every cluster was named by reading its nearest- and farthest-from-centroid messages (`taxonomy_clusters.md`).

## Escalation policy (gate G1)

`ALWAYS_ESCALATE = ['refund_billing', 'account_security', 'service_complaint']` - about **30%** of traffic, leaving ~70% eligible for automation.

Kept deliberately minimal. Over-stuffing this set would cap auto-handle coverage by construction and let the system look safe by doing nothing - the trivial always-escalate baseline already scores perfectly on false-auto-handle at zero coverage.

## Intents

| intent | share | disposition | cluster |
|---|---|---|---|
| `delivery_delay` | 12.4% | auto-eligible | 3 |
| `delivery_failure` | 11.3% | usually escalate | 6 |
| `order_investigation` | 9.8% | usually escalate | 9 |
| `refund_billing` | 11.6% | **ALWAYS ESCALATE** | 2 |
| `item_condition` | 10.0% | auto-eligible | 8 |
| `product_digital` | 9.3% | auto-eligible | 5 |
| `prime_membership` | 9.1% | auto-eligible | 4 |
| `account_security` | 2.5% | **ALWAYS ESCALATE** | - |
| `service_complaint` | 16.4% | **ALWAYS ESCALATE** | 0 |
| `unresolved_followup` | 10.0% | usually escalate | 7 |
| `other` | 0.0% | usually escalate | - |

*Shares sum slightly above 100%: `account_security` has no cluster of its own (only ~2.5% of traffic, so KMeans scattered it), and its share overlaps the clusters it was drawn from. It is included anyway because it is the highest-stakes intent in the set - a triage system with no label for "my account was hacked" has a hole exactly where automation is most dangerous.*

---

### `delivery_delay`

auto-eligible &middot; ~12.4% of traffic &middot; cluster 3

An order has not arrived yet, is running late, or the customer is asking when it will arrive. The package is still in transit or overdue - nothing claims it was delivered.

**Boundary.** vs delivery_failure: this one is LATE; delivery_failure is marked delivered but absent, or delivered to the wrong place. vs prime_membership: if the complaint is that a delivery was slow, it is delivery_delay even when the word 'Prime' appears - prime_membership is about the membership itself.

**Real examples from the data:**

- *"my order was scheduled to arrive yesterday and didnt. I was not informed that the parcel was late"*
- *"i just placed an order with deliver next day by 1pm and its saying it wont arrive till monday?"*
- *"I ordered something via Amazon Prime midday on Friday 24th November and still have no goods."*

### `delivery_failure`

usually escalate &middot; ~11.3% of traffic &middot; cluster 6

The delivery went wrong: marked delivered but not received, left in an unsafe place, given to a neighbour, or a complaint about the courier's behaviour.

**Boundary.** vs delivery_delay: something claims completion here - 'delivered' but absent or misplaced - whereas delivery_delay is still pending. vs order_investigation: if the customer supplies an order ID and asks for a lookup, prefer order_investigation.

**Real examples from the data:**

- *"I've not received a parcel that's apparently been delivered?"*
- *"My package was delivered to my neighbor."*
- *"They drove by my house three times without delivering"*

### `order_investigation`

usually escalate &middot; ~9.8% of traffic &middot; cluster 9

A specific order that cannot be answered without looking it up: the customer supplies an order ID, or explicitly asks the brand to investigate one order. Includes unexpected cancellations and wrong items.

**Boundary.** The distinguishing feature is REQUIRING PRIVATE DATA, not the topic. A late order described generally is delivery_delay; the same complaint with an order number and 'please look into it' is order_investigation.

**Real examples from the data:**

- *"please can you look into this order: 206-9303243-6219526 I paid for next day delivery and still not received!!"*
- *"i have just seen this order was cancelled and not by me. VERY UNHAPPY Order #701-5406600-7964269"*
- *"my order got delivered to someone else. Order Id: 171-6576778-6197962"*

### `refund_billing`

**ALWAYS ESCALATE** &middot; ~11.6% of traffic &middot; cluster 2

Money: a refund not received, an unexpected or wrong charge, cashback, price disputes, or a return that has not produced a refund.

**Boundary.** vs order_investigation: if the ask is about MONEY owed, it is refund_billing even when an order ID is present. vs prime_membership: a Prime subscription charge is refund_billing if the customer disputes the charge, prime_membership if they ask about benefits or cancellation.

**Real examples from the data:**

- *"Seller arranged for my return item to be picked up and now I can't get my refund."*
- *"is not refunding me since a month for order no."*
- *"I sent the product back & still haven't [been refunded]"*

### `item_condition`

auto-eligible &middot; ~10.0% of traffic &middot; cluster 8

The item or its packaging arrived in an unacceptable state: damaged goods, crushed boxes, or complaints about excessive or inadequate packaging.

**Boundary.** vs delivery_failure: the package ARRIVED here; the problem is its state, not its whereabouts.

**Real examples from the data:**

- *"disappointed to receive my parcel like this, especially with Christmas presents in which have been damaged"*
- *"your packaging get worse by the day: this large box for this single item is ridiculous!"*
- *"WTF! Why would you ship my purchase like this!!! It was a gift"*

### `product_digital`

auto-eligible &middot; ~9.3% of traffic &middot; cluster 5

Amazon's own products and digital services: the app, Alexa, Echo, Kindle, Fire TV, Prime Video, Amazon Music - including content availability by region.

**Boundary.** vs prime_membership: this is about the CONTENT or DEVICE working; prime_membership is about the subscription that grants access.

**Real examples from the data:**

- *"I would love to watch but it's 'currently unavailable to watch in' my location. Please explain"*
- *"can you please sort out your app, it's dreadful. After watching adverts it says 'content not available'"*
- *"When you are going to launch Amazon Prime Music in India ?"*

### `prime_membership`

auto-eligible &middot; ~9.1% of traffic &middot; cluster 4

The Prime subscription itself: what it includes, membership charges, renewal, cancellation, or the benefit not being honoured as a membership matter.

**Boundary.** OVERLAPS HEAVILY with delivery_delay - most observed Prime messages are really complaints about slow delivery. Rule: label delivery_delay unless the customer questions the membership's value, cost, or terms. This is the single hardest boundary in the taxonomy and is expected to be the main source of confusion in the Phase 6 matrix.

**Real examples from the data:**

- *"why spend money on prime to receive something days later"*
- *"I don't understand why Prime isn't shipping promptly anymore."*
- *"Used my Prime Membership again today for an order, not due to arrive until Tuesday, how can this be One Day delivery"*

### `account_security`

**ALWAYS ESCALATE** &middot; ~2.5% of traffic

Account access or compromise: cannot log in, password problems, account hacked, unauthorised access, or fraudulent charges by a third party.

**Boundary.** vs refund_billing: an unauthorised charge by a THIRD PARTY is account_security (the account is compromised); a wrong charge by Amazon is refund_billing.

**Real examples from the data:**

- *"someone has hacked my account and changed the login details as I just received an email confirming this"*
- *"account hacked, my email account linked with amazon changed without authorisation. Need immediate help."*
- *"I am unable to log in since a month Plz help"*

### `service_complaint`

**ALWAYS ESCALATE** &middot; ~16.4% of traffic &middot; cluster 0

Dissatisfaction with Amazon's support or the brand generally, with no specific actionable request: 'your customer service is awful', 'never using Amazon again'.

**Boundary.** vs unresolved_followup: that one is chasing a SPECIFIC unresolved issue; this is a general grievance with nothing concrete to action. If you can name the underlying order or problem, it is not service_complaint.

**Real examples from the data:**

- *"your customer service sucks ass"*
- *"why is your customer service so awful"*
- *"Such a #PoorService given by Amazon. They never give us proper reply on call"*

### `unresolved_followup`

usually escalate &middot; ~10.0% of traffic &middot; cluster 7

Chasing a problem already raised: no resolution, repeated contact, or an explicit 'still waiting' about a known issue.

**Boundary.** vs service_complaint: a concrete unresolved issue exists here. vs the delivery intents: use this when the salient fact is that contact has ALREADY failed, not the underlying topic.

**Real examples from the data:**

- *"AGAIN not left in my safe place!! I'm getting really frustrated with this!!"*
- *"Still not got this ..."*
- *"can you give me more information than this I've been waiting in? All it's said since yesterday"*

### `other`

usually escalate &middot; ~0.0% of traffic

Anything the taxonomy does not cover, including messages too vague to classify. Deliberately kept as an explicit bucket rather than forcing every message into a named intent.

**Boundary.** Use only when no named intent applies. A message that is merely ambiguous between two named intents is NOT other - pick the better fit and flag the ambiguity in the golden set.

---

## Known weaknesses

- **`prime_membership` vs `delivery_delay` is the hardest boundary.** Most observed Prime messages are really complaints about slow delivery. This is expected to be the main source of confusion in the Phase 6 matrix and is called out in advance rather than explained away afterwards.
- **`service_complaint` is the largest class (16%)** and is heterogeneous by construction, merged from two clusters of generalised grievance.
- **`account_security` is rare (~2.5%)**, so the golden set must oversample it to measure it at all. That makes the golden set deliberately NOT distribution-matched, which biases any headline accuracy figure and is reported in Phase 3.
- **English only.** Non-English traffic (Japanese, Spanish, French, German, Portuguese, Italian) was excluded before clustering because language, not intent, was otherwise the dominant signal. That is roughly a quarter of real traffic this taxonomy says nothing about.
