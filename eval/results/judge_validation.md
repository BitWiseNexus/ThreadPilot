# Judge validation (pipeline)

*40 items, blind re-scoring. Judge: `qwen/qwen3.8-27b`.*

> **Who the rater is.** AI assistant that built this repo, scoring blind against the written rubric. NOT an independent human annotator - weaker evidence than the phrase 'human validation' usually implies.

## send_unedited (the headline binary)

| | value |
|---|---|
| raw agreement | 70.0% |
| Cohen kappa | 0.3333 |
| rater says send-unedited | 60.0% |
| judge says send-unedited | 75.0% |

**WEAK - judge-derived quality numbers should not carry the report**

## Ordinal criteria (quadratic-weighted kappa)

| criterion | exact | within 1 | qw-kappa | rater mean | judge mean | judge bias |
|---|---|---|---|---|---|---|
| grounded | 15% | 75% | 0.3043 | 3.6 | 4.5 | +0.90 |
| on_intent | 28% | 68% | 0.3968 | 3.225 | 4.3 | +1.07 |
| no_overcommit | 57% | 98% | 0.4565 | 4.4 | 4.85 | +0.45 |
| voice | 22% | 78% | 0.1127 | 3.625 | 4.6 | +0.97 |
| actionable | 22% | 72% | 0.3861 | 3.3 | 4.275 | +0.97 |

## Where they disagreed on send_unedited (12)

- `1051383-1051381` rater=False judge=True — judge: "Directly mirrors precedent response for similar inquiry about quiz outcomes." / rater: "Asks how winners are selected; reply says escalated internally and answers nothing."
- `1245906-1245905` rater=False judge=True — judge: "Directly addresses contact verification with standard support link." / rater: "Apologises for a hassle none was described; never answers whether this channel is right."
- `128794-128792` rater=True judge=False — judge: "Invents specific status 'not announced' not in precedent." / rater: "Matches the brand standard not-yet-announced pattern."
- `1616332-1616330` rater=False judge=True — judge: "Correctly redirects to private channel for account-specific issues." / rater: "The actual question - must I pay return postage for your error - is never answered."
- `184872-184869` rater=False judge=True — judge: "Appropriate clarification request matching brand voice for app issues." / rater: "Customer already described the fault precisely; asking for more detail reads as not having read it."
- `2119275-2119273` rater=True judge=False — judge: "Lacks link; assumes availability without verification." / rater: "Answers directly; the Belgium-specific confirmation slightly outruns the precedent but is plausible."
- `2152253-2152252` rater=False judge=True — judge: "Standard escalation path for missing items matches precedent." / rater: "Three weeks unresolved and an order number already given; asking for your details ignores both. Weak precedent at 0.43."
- `2523373-2523371` rater=False judge=True — judge: "Matches precedent for security/delivery issues; appropriate next step." / rater: "A break-in with police involved, answered with sorry about this. Severely under-weights the situation."
- `2976549-2976548` rater=False judge=True — judge: "Matches precedent for feedback links; addresses shipping complaint intent." / rater: "Asks WHO they can talk to and says DESPERATE; reply routes them to a feedback form."
- `606154-606153` rater=False judge=True — judge: "Appropriately handles positive feedback with standard link." / rater: "This is PRAISE for the packaging; reply routes it to a feedback form as though it were a complaint."
- `727109-727105` rater=False judge=True — judge: "Mirrors precedent exactly, including specific time constraint." / rater: "States a specific 20:00 cutoff that appears nowhere in the precedent - an invented specific."
- `931379-931377` rater=True judge=False — judge: "Generic link may not address specific 'sign-in button' UI issue." / rater: "Generic help link, reasonable; precedent was unrelated."
