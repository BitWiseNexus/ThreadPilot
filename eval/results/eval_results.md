# Evaluation results

*Generated 2026-09-17 01:21:33. n=200 golden rows; judge on a paired subsample of 100.*

| system | accuracy | macro F1 | coverage | false auto-handle | send unedited |
|---|---|---|---|---|---|
| `trivial` | 11.0% [7.0%, 15.0%] | 0.018 | 0.0% | 0.0% | 1.0% |
| `simple_tfidf_silver` | 30.0% [24.0%, 36.0%] | 0.263 | 58.5% | 53.0% | 15.0% |
| `simple_tfidf_cluster` | 36.0% [29.5%, 43.0%] | 0.332 | 82.0% | 54.3% | 17.0% |
| `retrieval_1nn` | 36.0% [29.5%, 43.0%] | 0.332 | 78.5% | 56.0% | 48.0% |
| `pipeline_no_retr` | 80.5% [75.0%, 86.5%] | 0.790 | 0.0% | 0.0% | 52.0% |
| `pipeline_no_gates` | 80.5% [75.0%, 86.5%] | 0.790 | 100.0% | 55.0% | 78.0% |
| `pipeline` | 80.5% [75.0%, 86.5%] | 0.790 | 62.0% | 41.1% | 78.0% |

## Caveats carried with these numbers

- The golden set is NOT distribution-matched (equal cluster allocation, account_security oversampled ~8x, hard cases sought). Accuracy here does not estimate production accuracy.
- Accuracy should be read against the inter-labeller ceiling (75.0% if known), not against 100%.
- Golden labels carry ~15% measured residual noise (D37).
- The judge shares a provider with the generator and is smaller.
- Batched inference shifts ~20% of individual predictions without changing aggregate accuracy (D44).
