# Phase 1 - brand selection

Criteria pre-registered in `docs/requirements.md` 2.1 *before* this ran. Seed 20260909; 4000 sampled pairs per brand; 2,811,774 total rows (1,537,843 inbound / 1,273,931 outbound).

`c3_usable_rate` = sum of the classes admitted to the retrieval index: `self_contained`, `link_referral`, `diagnostic_ask`, `handoff_with_ask`. Excluded: `channel_switch_bare`, `bare_ack`, `fragment`, `truncated`. See `threadpilot.data.clean` for why a single deflection flag was the wrong abstraction, and `c3_proxy_validation.md` for the measured agreement of this classifier with an independent rater.

| brand | c1_usable_pairs | c3_usable_rate | self_contained | link_referral | diagnostic_ask | handoff_with_ask | channel_switch_bare | bare_ack | fragment | truncated | c2_diversity_entropy | c4_multiturn_rate | c5_escalate_pole_rate | c5_auto_pole_rate | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SpotifyCares | 43092 | 0.896 | 0.335 | 0.1585 | 0.1452 | 0.2572 | 0.0668 | 0.0067 | 0.0003 | 0.0302 | 0.7306 | 0.3728 | 0.0558 | 0.0747 | 0.7933 |
| AmazonHelp | 168814 | 0.7933 | 0.453 | 0.256 | 0.0633 | 0.021 | 0.0343 | 0.0348 | 0.1358 | 0.002 | 0.6263 | 0.5055 | 0.069 | 0.0393 | 0.6982 |
| AmericanAir | 36531 | 0.879 | 0.6633 | 0.0428 | 0.0143 | 0.1588 | 0.0867 | 0.0323 | 0.0005 | 0.0015 | 0.6959 | 0.3212 | 0.0425 | 0.0283 | 0.6297 |
| Uber_Support | 56160 | 0.7065 | 0.154 | 0.1872 | 0.0037 | 0.3615 | 0.2895 | 0.0015 | 0.0 | 0.0025 | 0.7184 | 0.289 | 0.167 | 0.0625 | 0.6064 |
| SouthwestAir | 28828 | 0.8592 | 0.6388 | 0.0432 | 0.0315 | 0.1457 | 0.0648 | 0.0522 | 0.0232 | 0.0005 | 0.673 | 0.278 | 0.0227 | 0.0377 | 0.5309 |
| VirginTrains | 27416 | 0.6925 | 0.6025 | 0.0515 | 0.02 | 0.0185 | 0.017 | 0.203 | 0.0715 | 0.016 | 0.6786 | 0.467 | 0.0515 | 0.0483 | 0.5262 |
| British_Airways | 29290 | 0.6255 | 0.4612 | 0.027 | 0.033 | 0.1042 | 0.026 | 0.0225 | 0.319 | 0.007 | 0.7467 | 0.3455 | 0.0495 | 0.057 | 0.5048 |
| AppleSupport | 106646 | 0.7073 | 0.1867 | 0.182 | 0.082 | 0.2565 | 0.2853 | 0.006 | 0.0 | 0.0015 | 0.6429 | 0.2933 | 0.0318 | 0.0535 | 0.4913 |
| comcastcares | 32921 | 0.6927 | 0.2275 | 0.0075 | 0.012 | 0.4457 | 0.262 | 0.0158 | 0.0 | 0.0295 | 0.7285 | 0.2737 | 0.0367 | 0.0413 | 0.4907 |
| Delta | 42114 | 0.6895 | 0.5373 | 0.0235 | 0.0123 | 0.1165 | 0.0867 | 0.1113 | 0.1108 | 0.0018 | 0.6177 | 0.3215 | 0.0293 | 0.0445 | 0.3729 |
| TMobileHelp | 34215 | 0.5327 | 0.1062 | 0.0382 | 0.0073 | 0.381 | 0.454 | 0.0107 | 0.0005 | 0.002 | 0.6662 | 0.3992 | 0.0457 | 0.046 | 0.3463 |
| Tesco | 38468 | 0.407 | 0.2198 | 0.0085 | 0.072 | 0.1067 | 0.03 | 0.0123 | 0.545 | 0.0057 | 0.5851 | 0.3635 | 0.0293 | 0.0395 | 0.0833 |
