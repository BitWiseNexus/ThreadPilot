# C3 proxy validation

The `c3_usable_rate` that decides brand selection comes from a regex classifier I wrote. This checks it against an independent rater (`qwen/qwen3.8-27b`) which was given the downstream definition of usable precedent and **no hint of the heuristic**.

- n rated: **257** (stratified across all 5 reply classes)
- raw agreement: **69.6%**
- Cohen kappa: **0.375** -> WEAK - C3 is unreliable; brand table must be re-weighted

## Agreement by reply class

| kind | n | agreement | regex_usable | llm_usable |
|---|---|---|---|---|
| bare_ack | 36 | 0.917 | 0.0 | 0.083 |
| channel_switch_bare | 36 | 0.639 | 0.0 | 0.361 |
| diagnostic_ask | 36 | 0.833 | 1.0 | 0.833 |
| fragment | 15 | 0.467 | 0.0 | 0.533 |
| handoff_with_ask | 36 | 0.833 | 1.0 | 0.833 |
| link_referral | 36 | 0.889 | 1.0 | 0.889 |
| self_contained | 36 | 0.583 | 1.0 | 0.583 |
| truncated | 26 | 0.115 | 0.0 | 0.885 |

## Agreement by brand

| brand | n | agreement | regex_usable | llm_usable |
|---|---|---|---|---|
| AmazonHelp | 92 | 0.652 | 0.522 | 0.63 |
| AmericanAir | 80 | 0.725 | 0.6 | 0.625 |
| SpotifyCares | 85 | 0.718 | 0.565 | 0.612 |
