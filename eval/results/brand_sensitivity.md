# Brand-choice sensitivity to the definition of usable precedent

The C3 classifier agrees with an independent rater at **kappa 0.374**. Per-class agreement is 0.83-0.92 where the definition is unambiguous and collapses on `truncated` (0.12) and `fragment` (0.47) - a definitional divergence, not classification noise. Rather than tune the classifier until it matches the rater, this asks whether the *decision* depends on the disagreement at all.

Same scoring weights throughout; only the set of admitted reply classes changes.

## Ranking under each definition

| definition | 1st | 2nd | 3rd | admits |
|---|---|---|---|---|
| `A_strict_answers_only` | **AmazonHelp** | VirginTrains | AmericanAir | 2 classes |
| `B_project_default` | **SpotifyCares** | AmazonHelp | AmericanAir | 4 classes |
| `C_rater_aligned` | **SpotifyCares** | AmazonHelp | AmericanAir | 5 classes |
| `D_most_inclusive` | **AmazonHelp** | SpotifyCares | British_Airways | 6 classes |

**NOT ROBUST - the winner changes by definition ({'A_strict_answers_only': 'AmazonHelp', 'B_project_default': 'SpotifyCares', 'C_rater_aligned': 'SpotifyCares', 'D_most_inclusive': 'AmazonHelp'}). The brand choice cannot be justified on this metric alone and must be reported as such.**

Brands in the top 3 under every definition: AmazonHelp
