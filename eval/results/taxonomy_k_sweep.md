# Taxonomy: choosing k

8,000 customer messages, sentence-transformers/all-MiniLM-L6-v2, seed 20260909.

Silhouette alone tends to favour very small k on short support text, which would collapse genuinely different intents together. `balance_entropy` (1.0 = even cluster sizes) and `smallest_pct` are shown next to it because a k whose smallest cluster is <2% of traffic gives the golden set too few examples per class to measure anything.

| k | silhouette | inertia | balance_entropy | smallest_cluster | smallest_pct | largest_pct |
|---|---|---|---|---|---|---|
| 4 | 0.0653 | 5449.1 | 0.9605 | 953 | 0.1191 | 0.3059 |
| 5 | 0.0581 | 5364.8 | 0.9688 | 912 | 0.114 | 0.2921 |
| 6 | 0.0583 | 5286.3 | 0.9803 | 857 | 0.1071 | 0.2522 |
| 7 | 0.0594 | 5217.4 | 0.9877 | 797 | 0.0996 | 0.1953 |
| 8 | 0.0584 | 5164.5 | 0.9913 | 742 | 0.0927 | 0.1665 |
| 9 | 0.056 | 5118.7 | 0.9966 | 731 | 0.0914 | 0.13 |
| 10 | 0.0548 | 5078.2 | 0.9948 | 529 | 0.0661 | 0.1242 |
| 11 | 0.0555 | 5041.0 | 0.9941 | 529 | 0.0661 | 0.1185 |
| 12 | 0.0577 | 5009.8 | 0.9882 | 406 | 0.0508 | 0.1185 |
| 13 | 0.0514 | 4974.1 | 0.9777 | 177 | 0.0221 | 0.1156 |
| 14 | 0.051 | 4948.5 | 0.9783 | 175 | 0.0219 | 0.102 |
