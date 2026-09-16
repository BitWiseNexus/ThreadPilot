# Failure analysis

*200 golden rows. Modes ordered by HARM, not frequency: a wrong auto-handle publishes a wrong answer under the brand's name, a wrong escalate costs an agent thirty seconds.*

## 1. False auto-handle (the dangerous failure)

**51 / 124** — HIGH - publishes an unreviewed reply under the brand's name

**Concentrated in:** `delivery_failure` (11), `unresolved_followup` (9), `delivery_delay` (8), `order_investigation` (7), `prime_membership` (6)

**Hypothesis.** The model proposed auto_handle on all 200 rows, so every escalate comes from a deterministic gate. The gates fire on intent (G1), missing precedent (G2), low confidence (G3) and explicit identifiers (G4) - none of which detects the two rubric tests that dominate the gold labels: E5 (anger an automated reply would worsen) and E2 (a reply that would have to commit a remedy). Those are properties of TONE and OBLIGATION, and no current gate reads either.

**Where the fix goes.** New gate. A sentiment/anger signal and a commitment-language detector would target the actual gap. Prompt changes will not help while the model's own proposal is constant.

**Real examples:**

- *"<user> <user> <url> I have ordered nd requested the call centre to deliver the product on 23rd still it got returned."*
  gold `delivery_failure`/`escalate` → predicted `delivery_failure`/`auto_handle` (gates none, sim 0.70)
  reply: "Sorry this happened. Could you please check your delivery details in the link and let us know if anything needs updating?"
- *"Hey <user> this probably wasn't a great way to ship a battery. Also the driver threw it on my porch. 😠 #amazon <user> #Lithium <url>"*
  gold `item_condition`/`escalate` → predicted `delivery_failure`/`auto_handle` (gates none, sim 0.53)
  reply: "Sorry about that. Please contact us here by phone or chat so an agent can look into it: <url>"
- *"<user> amazon has adopted a cheap way of marketing.amazon has cancelled my order without any reason"*
  gold `order_investigation`/`escalate` → predicted `order_investigation`/`auto_handle` (gates none, sim 0.84)
  reply: "Every cancelled order is followed up with an email explaining the reason; please check your email for details."

## 2. Over-escalation (costly, not dangerous)

**17 / 76** — LOW - wastes an agent's time, publishes nothing wrong

**Gates responsible:** `G1` (9), `G2` (7), `G4` (1)

**Hypothesis.** G1 escalates by INTENT, unconditionally. But the rubric judges per message, and a generically-answerable message inside an always-escalate intent ('how long do refunds normally take?') is auto-handleable. This is the predicted cost of a blunt intent-level gate, now measured.

**Where the fix goes.** Gate design: make G1 conditional on the message rather than the intent alone. That is the single highest-value change available.

**Real examples:**

- *"<user> <user> need support on order no 171-7352190-1637951. Size is very small need replacement or return"*
  gold `order_investigation`/`auto_handle` → predicted `order_investigation`/`escalate` (gates ['G4'], sim 0.63)
  reply: "Sorry the size isn’t right. Please fill out this form so we can process a replacement or return: <url>."
- *"<user> Bought Nikon B700 &amp; it's wifi and bluetooth not working. Contacted tele service centre, they were not able to answer."*
  gold `other`/`auto_handle` → predicted `other`/`escalate` (gates ['G2'], sim 0.43)
  reply: "My apologies for the trouble, kindly fill in your details here: <url> and we’ll get back to you soon."
- *"Amazon Flex needs to screen the drivers better for logistics delivery. Some of these fuckers are mad dumb."*
  gold `service_complaint`/`auto_handle` → predicted `service_complaint`/`escalate` (gates ['G1'], sim 0.66)
  reply: "<user> We’re sorry about your experience with Flex drivers. Please contact us via this secure link: <url> so we can look into it."

## 3. Adjacent-intent confusion

**39 / 200** — MEDIUM - wrong intent can mis-route and mis-gate

**Top pairs:** `delivery_failure -> delivery_delay` (4), `prime_membership -> delivery_delay` (4), `service_complaint -> other` (3), `order_investigation -> delivery_delay` (3), `unresolved_followup -> delivery_failure` (3), `refund_billing -> other` (2)

**Hypothesis.** These are the SAME pairs the two independent labellers disagreed on in Phase 3, recorded before the classifier existed. The classifier fails where the taxonomy boundary is genuinely contested, not arbitrarily - so most of this is irreducible against these labels rather than a model deficiency.

**Where the fix goes.** Taxonomy, not prompt. Merging delivery_delay/delivery_failure or sharpening the prime_membership boundary would remove more error than any prompt edit.

**Real examples:**

- *"<user> why can't I win any quiz contest even after being eligible to win it ? I wonder how you guys select winners?"*
  gold `service_complaint`/`auto_handle` → predicted `other`/`auto_handle` (gates none, sim 0.53)
  reply: "<user> We've escalated this to our team internally for an update. Kindly stay tuned."
- *"<user> <user> <user> #405-3161422-8924357 I was informed about the Guaranteed delivery yesterday and it never got delivered"*
  gold `order_investigation`/`escalate` → predicted `delivery_delay`/`escalate` (gates ['G4'], sim 0.79)
  reply: "I’m sorry for the delay. Have you received an updated delivery date? If so, please let us know."
- *"<user> why do your delivery drivers insist on trying to deliver next day items outside of commercial times? Pointless having next day."*
  gold `delivery_failure`/`auto_handle` → predicted `delivery_delay`/`auto_handle` (gates none, sim 0.79)
  reply: "Orders placed with next‑day delivery arrive one business day after dispatch. More info: <url>"

## 4. Drafting with weak precedent

**10 / 20** — MEDIUM - a confident reply with nothing behind it

**Hypothesis.** Gate G2 escalates below TAU_SIM=0.45, but similarity is a blunt proxy for 'is there relevant precedent'. A message can score 0.5 against a superficially similar but substantively different precedent and still be auto-handled.

**Where the fix goes.** Retrieval. Re-ranking, or requiring agreement among the top-k rather than the single max, would tighten this.

**Real examples:**

- *"F you <user> you lost my corgi onesie and now Halloween is ruined"*
  gold `delivery_failure`/`escalate` → predicted `delivery_failure`/`auto_handle` (gates none, sim 0.45)
  reply: "<user> Oh no, we’re sorry your corgi onesie didn’t arrive. Please reach us here: <url> so we can help."
- *"He's part cat and part boy. Notice him playing in a small cardboard box. Thanks <user> for the hours of entertainment. <url>"*
  gold `other`/`auto_handle` → predicted `other`/`auto_handle` (gates none, sim 0.46)
  reply: "Sounds like a fun time! Thanks for sharing."
- *"<user> please include legendary shows like x-files etc into your library with Musical Live concerts.. to make it more appealing."*
  gold `product_digital`/`auto_handle` → predicted `product_digital`/`auto_handle` (gates none, sim 0.47)
  reply: "Thanks for the suggestion! We'll share your feedback with the team."

## 5. EVALUATION failure: a 17% grab-bag inflates the headline

**34 / 34** — N/A - a defect in the measurement, not in the system

**Hypothesis.** `other` is the largest gold class (34/200, 17%) and the classifier scores 100% on it - yet it is not one thing. It holds praise ('Absolutely fantastic customer service'), pre-purchase questions ('When will amazon restock this item?'), checkout problems and third-party product faults. Scoring perfectly on a bag of unlike things is not a capability, but it lifts overall accuracy from 76.5% to 80.5% - 4.0% points of the headline number. Part of that is the same methodological coupling that makes 80.5% exceed the 75.0% inter-labeller ceiling: the classifier reads the taxonomy definitions I also used when adjudicating those rows into `other`.

**Where the fix goes.** Taxonomy: add `positive_feedback` and `pre_purchase`, which would break the grab-bag into measurable classes. Frozen before labelling on purpose (D12/D14), so reported rather than patched - and it is the highest-value single change in 'what I would do next'.

**Real examples:**

- *"<user> hi I've got tickets for j cole tomorrow o2 however I'm not sure if they're actual Seated tickets in the arena? It's AL101"*
  gold `other`/`auto_handle` → predicted `other`/`auto_handle` (gates none, sim 0.50)
  reply: "<user> Hi! To confirm seat details, please check the confirmation email or contact the venue."
- *"Absolutely fantastic customer service at <user> problem understood, sorted and refunded in 10 minutes. Bravo."*
  gold `other`/`auto_handle` → predicted `other`/`auto_handle` (gates none, sim 0.73)
  reply: "<user> I'm glad we could help! We love our customers!🙂"
- *"<user> When will amazon restock this item? <url>"*
  gold `other`/`auto_handle` → predicted `other`/`auto_handle` (gates none, sim 0.71)
  reply: "<user> The item is currently unavailable and we can't speculate on restock timing."
