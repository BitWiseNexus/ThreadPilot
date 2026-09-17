# ThreadPilot — Decision Log

> Non-obvious decisions, captured **as they are made**. The Phase 8 report needs 10-15 of
> these; reconstructing them from memory at the end produces sanitised fiction, so they are
> written down at the moment of choosing, including the ones that later turn out wrong.
>
> Format: what I decided · what I rejected · why · what it costs me.

---

### D1 — Python 3.12.4, not the machine default 3.14.3
**Rejected:** using the default interpreter.
**Why:** `torch` and `sentence-transformers` wheel availability on 3.14 is unreliable; a
source build would blow the 15-minute reproducibility promise on a fresh clone.
**Cost:** an evaluator on 3.14-only would have to install 3.12. Documented in the README.

### D2 — numpy brute-force cosine instead of FAISS or Chroma
**Rejected:** FAISS, Chroma.
**Why:** the index is on the order of 10^4 vectors x 384 dims. Exact search is a single
matmul, faster than building an ANN index, with zero dependency risk. It is also *exact*,
so retrieval quality is never confounded by ANN recall loss — which matters because I have
to attribute grounding failures to the right cause.
**Cost:** would not scale past ~10^6 vectors. It does not need to.

### D3 — Embed the customer message, not the brand reply
**Rejected:** indexing on reply text, or on concatenated pair text.
**Why:** the retrieval question is "who else asked this?" The payoff is the reply attached to
that neighbour. Embedding replies would retrieve on answer-similarity, which is the wrong
similarity for an unanswered incoming tweet.
**Cost:** a precedent whose customer phrasing is unusual but whose resolution is apt will be
missed. Accepted; noted as a candidate failure mode for Phase 7.

### D4 — Golden-set threads excluded from the retrieval index
**Rejected:** indexing everything.
**Why:** otherwise the retriever surfaces the literal target reply and every quality metric
is inflated to the point of fraud. This is the single most important correctness detail in
the harness.
**Cost:** a slightly smaller index. Enforced by a `pytest` assertion, not by discipline.

### D5 — Thresholds tuned on a separate silver dev set, never on the golden set
**Rejected:** tuning on the golden set (the convenient option), or holding out a golden
split (which would shrink reported n).
**Why:** lets the headline number be honestly described as held-out. Tuning on the eval set
and reporting it as performance is the most common quiet dishonesty in ML write-ups.
**Cost:** silver labels are noisier, so thresholds land slightly off the golden-tuned
optimum. A slightly-suboptimal honest number beats an optimal tuned-on-test number.

### D6 — Auto/escalate = LLM proposal + one-directional deterministic gates
**Rejected:** pure-LLM decision; pure rule-based decision.
**Why:** cost asymmetry (a false auto-handle publishes a wrong answer under the brand's
name; a false escalate costs an agent thirty seconds), auditability (a named gate beats a
model narrating its own reasoning post-hoc), and tunability (gives a real coverage/risk
knob). Gates can only *add* caution, never grant it.
**Cost:** hand-designed gates encode my blind spots, and `ALWAYS_ESCALATE` caps achievable
coverage by construction. Both disclosed in "what's misleading"; a gates-off ablation in
Phase 6 forces the rule layer to justify itself numerically.

### D7 — Judge on a different model family than the generator
**Rejected:** judging Llama output with Llama (the default, easy path).
**Why:** self-preference bias — LLM judges systematically favour text from their own family.
Groq hosts several families on the free tier, so cross-family judging costs nothing.
**Cost:** still the same provider and the same broad pretraining era, so shared bias is
reduced, not eliminated. Stated as a residual weakness rather than claimed as solved.

### D8 — Deflections and bare acknowledgements excluded from the index
**Rejected:** indexing all brand replies.
**Why:** "please DM us" is precedent we must not teach the model to imitate. A draft that
deflects would score well on tone and badly on usefulness, quietly corrupting the quality
metric.
**Cost:** discards real brand behaviour, so the index is less representative of what the
brand actually does than of what it does *when it helps*. The filter's effect on index size
is reported, not hidden.

### D9 — Report auto-handle as a coverage/risk curve, not a single number
**Rejected:** a headline safety percentage at one operating point.
**Why:** the trivial `always-escalate` baseline achieves a *perfect* false-auto-handle rate
at zero coverage. Any single-number safety metric is therefore gameable, and presenting one
would be misleading by construction.
**Cost:** harder to summarise in one line. A named operating point is given for headline
purposes with its caveat printed adjacent.

### D10 — Committed LLM response cache, with an offline replay path
**Rejected:** requiring every evaluator to hold a working API key.
**Why:** an evaluator can regenerate every reported number with no `GROQ_API_KEY` at all,
which is the strongest form of the reproducibility claim. Also makes free-tier limits a
non-issue when metrics code changes.
**Cost:** a replay is a *recording*, not a fresh run. Said plainly in the README; a
`--no-cache` flag forces live calls; a cache miss in offline mode is a hard error rather
than a silent fallback.

### D11 — `python tasks.py <cmd>` instead of a Makefile
**Rejected:** Make.
**Why:** `make` is not present on this Windows machine, and a Windows evaluator hitting a
missing `make` breaks the 15-minute promise at step one.
**Cost:** slightly unidiomatic for reviewers who expect `make eval`.

### D12 — Intent taxonomy derived by clustering, not written a priori
**Rejected:** hand-writing a plausible support taxonomy up front (faster, and nobody would
notice).
**Why:** an a-priori taxonomy is a guess about the brand's traffic, and the classifier would
then be evaluated on my guess rather than on reality. Cluster exemplars are committed so
the human naming step is auditable.
**Cost:** slower, and cluster boundaries do not always land on clean human concepts —
forcing documented merge decisions in Phase 2.

### D13 — Brand selection criteria pre-registered before EDA
**Rejected:** looking at the data first, then writing up the justification.
**Why:** post-hoc criteria are rationalisation. `requirements.md` section 2.1 fixes the five
criteria and section 2.2 records a falsifiable hypothesis (AppleSupport fails on
channel-deflection; an airline wins) before any measurement.
**Cost:** I may be publicly wrong. That is the entire point of writing it down first.

### D14 — Golden set built before the pipeline
**Rejected:** building the pipeline first, then labelling (the natural order).
**Why:** if labels come after the system, the system is unconsciously built to the test and
the eval measures nothing.
**Cost:** labelling against a taxonomy before seeing pipeline outputs means some labelling
guidance has to be written blind, and Phase 3 notes will record where that hurt.

### D15 — No UI
**Rejected:** a Streamlit demo.
**Why:** the brief says the proof is worth more than the system. A demo produces nothing an
evaluator can check; the same hours in failure analysis and judge validation produce
evidence.
**Cost:** the auto/escalate decision is less immediately tangible. Mitigated by `rich` CLI
output that always shows retrieved precedent next to the draft it justifies.


### D16 — Kaggle auth via the modern `access_token`, download via `kagglehub`
**Rejected:** legacy `kaggle.json` with a `username`+`key` pair; shelling out to the `kaggle` CLI via subprocess.
**Why:** Kaggle now labels `kaggle.json` "Legacy API Credentials" and issues `KGAT_`-prefixed
tokens to `~/.kaggle/access_token` instead. `kagglehub.dataset_download()` reads that token
natively, is pure Python (no subprocess, no exit-code parsing), and returns a local path.
I initially assumed the legacy pair was the safe default and was corrected by reading the
docs rather than trusting recall — noted here because "verify the environment, do not
remember it" is a repeatable lesson, not a one-off.
**Cost:** `kagglehub` caches to `~/.cache/kagglehub/` rather than `data/raw/`, so the script
has to locate and link/copy the CSV into place. Handled with a three-tier fallback
(kagglehub -> `kaggle` CLI -> detect a manually-downloaded file) so a fresh clone works
regardless of which auth an evaluator has.

### D17 — `torch` / `sentence-transformers` are OPTIONAL dependencies
**Rejected:** one `requirements.txt` containing torch (the obvious layout).
**Why:** this is the decision that actually saves the 15-minute reproducibility promise.
`torch` is a ~200MB wheel and the single largest cost in a fresh-clone setup. But an
evaluator reproducing the *reported numbers* never needs to embed anything — they need the
committed embedding cache, the committed retrieval index, and the committed LLM response
cache. So the core `requirements.txt` stays light (pandas, numpy, scikit-learn, groq, rich,
dotenv, tenacity) and `requirements-embed.txt` adds torch + sentence-transformers only for
rebuilding embeddings from scratch. The offline eval path therefore needs **neither torch
nor an API key**.
**Cost:** two requirement files to keep in sync, and a committed binary artifact (the
embedding matrix) that a reviewer cannot read by eye. Mitigated by a script that rebuilds it
and a test asserting the committed matrix matches a freshly-computed one for a fixed sample
of rows.

### D18 — `requirements.in` (loose, hand-edited) + `requirements.txt` (frozen lock)
**Rejected:** hand-pinning exact versions from memory.
**Why:** I have no network access from the tool environment, so any exact version I typed
would be a guess dressed up as a pin. Instead: loose constraints in `requirements.in`,
install, then `pip freeze` the result into `requirements.txt`. The lock then reflects what
actually resolved and ran, which is the only pin worth having.
**Cost:** one extra step during setup, and the lock is platform-specific (Windows/cp312).
Noted in the README.


### D21 — REVERSAL of D8: a handoff that names what it needs IS usable precedent
**Rejected:** my own earlier D8, which excluded every channel-switch reply from the
retrieval index.
**Why:** D8 was falsified by measurement, not by argument. The v1 reply classifier scored
**Cohen kappa 0.20** against an independent LLM rater, and the single largest disagreement
cell was exactly this: on `channel_switch` replies my classifier said 0% usable while the
rater said 75% usable. Reading the cases, the rater was right. "Can you DM us your booking
reference, email address and phone number?" names precisely what the brand needs and is the
brand's genuine handling pattern for account-specific issues. Excluding it discarded real
precedent. Only a *bare* switch ("feel free to DM us anytime") contains no answer.
So `HANDOFF_WITH_ASK` is now admitted and `CHANNEL_SWITCH_BARE` is not.
**Bonus that fell out of it:** a message the brand historically took private is a message a
human handled, so handoff precedent is a *positive escalation signal* rather than noise.
`ReplyKind.implies_escalation` feeds gate G-handoff in Phase 4. The v1 design threw away a
useful feature along with the data.
**Cost:** the index now contains replies that do not themselves answer the question, so a
grounded draft may legitimately be "we need X to look into this". That is the brand's real
behaviour, and conflating it with a refusal to help would be the bigger error.

### D22 — Usability is a content signal, not a length threshold
**Rejected:** the v1 rule "body under 50 characters means empty pleasantry".
**Why:** it was the dominant source of v1 classification error. It discarded
"Has the order been marked as dispatched yet?" (43 chars, a useful diagnostic question) and
"You can check your subscriptions here <link>" (37 chars, a complete answer). Usefulness
does not correlate with length in support replies, where the best answers are often the
shortest. v2 detects instruction/directive/question/info-request signals and applies no
length gate above a 12-character floor.
**Cost:** more regexes to maintain, and they encode English-language assumptions - a real
limitation, since the dataset contains German, French and Spanish replies. Reported as a
known bias rather than papered over.

### D23 — Truncation means "ends mid-clause", not "ends without a full stop"
**Rejected:** flagging any long reply lacking terminal punctuation.
**Why:** two bugs compounded here and together they reordered the brand table, which is
the clearest illustration in this project of why unvalidated heuristics are dangerous.
(a) `_body()` stripped trailing punctuation before truncation was tested, so the test could
never see a terminal "?" or "." and fired on *everything* past the length threshold.
(b) `SIGNATURE_RE` handled only `^AS` / `*KittyG` / `- Tom`, missing SpotifyCares' `/AY` and
comcastcares' `-DR` (all-caps initials); unstripped signatures leave the body ending in
letters, which then reads as truncation.
Result: SpotifyCares 53% and comcastcares 54% "truncated" against AmazonHelp's 4%.
**This changed the project's headline decision.** Before the fix, brand selection picked
AmazonHelp (score .755) with SpotifyCares 9th (.405). After the fix, SpotifyCares is 1st
(.793) and AmazonHelp 2nd (.698). A regex detail, not a judgement about support data, had
been driving the choice.
v3 requires all three of: at/above the length threshold, no terminal punctuation, AND a
dangling final function word or trailing comma. Truncation rates are now 0.2%-3%.
**Cost:** the dangling-word list is English-only, so non-English truncation is under-detected.
Regression-tested in `tests/test_clean.py` with the real examples that broke it.

### D24 — All parts of a split reply are excluded, pending reassembly
**Rejected:** admitting part 1 of an N-part reply as a complete answer.
**Why:** "Kindly report this to our support team 1/3" is the opening of an answer, not an
answer. v1 admitted part 1 and excluded part 2+, which is the worst of both worlds: it kept
the incomplete openings and discarded the substance.
**Cost:** real precedent is lost, unevenly by brand - Tesco (54% fragments) and
British_Airways (32%) are penalised hard, and both drop in the brand table as a result. The
right fix is reassembling consecutive same-author replies into whole messages, which is a
Phase 4 task. Excluding is the conservative interim choice and its per-brand cost is
reported in the brand table rather than hidden.


### D25 — The composite score cannot decide the brand; robustness decides it
**Rejected:** taking the top of the composite table at face value (which would have meant
SpotifyCares); and the alternative of tuning the C3 classifier until it agreed with the
rater.
**Why:** the C3 classifier reached kappa 0.374 against an independent rater. Per-class
agreement is 0.83-0.92 wherever the definition is unambiguous and collapses on exactly two
classes - `truncated` (0.12) and `fragment` (0.47). Inspecting those cases, the divergence
is **definitional**: my classifier asks "is this a complete, well-formed reply?", the rater
asks "does the visible text contain help?" Both are legitimate questions, so further
regex-tuning would just be fitting one heuristic to another - a third iteration of the same
mistake rather than a fix.
So instead of resolving the definition, I tested whether the *decision* depends on it:
`scripts/brand_sensitivity.py` re-ranks every brand under four defensible definitions of
usable precedent, holding the scoring weights fixed.
**Result: the winner is NOT robust.** Strict and most-inclusive definitions pick AmazonHelp;
the project-default and rater-aligned definitions pick SpotifyCares. So the composite score
genuinely cannot separate the top two, and reporting "SpotifyCares scored highest" as if it
were a finding would be exactly the sort of overclaiming this project exists to avoid.
**What decides it instead:** AmazonHelp is the only brand in the top 3 under all four
definitions (1st, 2nd, 2nd, 1st), whereas SpotifyCares ranks 4th under the strictest. It
also carries 168,814 usable pairs against SpotifyCares' 43,092 - a ~4x larger retrieval
index, which matters directly for a grounding-based system - and the highest multi-turn rate
(0.506) of any candidate.
**Cost:** AmazonHelp has the *lowest* intent-diversity proxy of the leading brands (0.626 vs
SpotifyCares 0.731), so the classification task will be somewhat easier and per-class metrics
somewhat less interesting. That is a real trade and it goes in the report: I bought decision
robustness and index size at the price of task difficulty. The diversity proxy is also
TF-IDF-based and may understate a large retailer whose intents differ semantically more than
lexically - checked in Phase 2 with MiniLM embeddings, where a genuinely narrow taxonomy
would show up.
**Also recorded:** the pre-registered hypothesis in requirements.md 2.2 was half right.
"AppleSupport fails C3" held (28.5% bare channel switches; 8th of 12 despite being 2nd by
volume). "An airline wins" was falsified - the best airline, AmericanAir, placed 3rd.


### D26 — The subsample is NOT redistributed; it is rebuilt deterministically
**Rejected:** committing `data/processed/subsample.parquet` (the original plan, and what
D10 implicitly assumed).
**Why:** the subsample is 8,000 real tweets derived from the Kaggle *Customer Support on
Twitter* dataset, which is **CC BY-NC-SA 4.0**. Committing it to a public repo is
redistribution. That is arguably permitted here (attributed in CITATIONS.md, non-commercial
take-home), but ShareAlike would then reach the repo, and the cleaner answer is simply not
to redistribute someone else's data to make my own setup marginally more convenient.
**What makes this cheap rather than costly:** the rebuild is deterministic from
`SEED = 20260909`, and `tests/test_subsample.py::test_rebuild_is_byte_identical` verifies
that by re-running the builder and comparing SHA-256. So an evaluator does not have to
*trust* that their rebuild matches mine - they can check it. `subsample_meta.json` stays
committed (counts and provenance, no tweet text) so the match is verifiable before anything
downstream is trusted.
**Scoped deliberately narrowly.** `cache/llm_cache.sqlite` stays committed: it stores model
*output* plus a SHA-256 *hash* of each prompt, never the prompt text, so it redistributes no
dataset content. The no-API-key reproduction path (D10) is therefore untouched - only the
"no credentials at all" path is. Reproducing now needs a free Kaggle token and one ~500MB
download; it still needs no Groq key and no torch.
**Cost, stated plainly:** the fresh-clone path gained a required download step, so the
under-15-minute claim now covers setup + rebuild + offline eval, not clone + eval alone.
Phase 9 re-times it against that definition rather than quietly re-using the old one.


### D27 — Restrict the subsample to thread-opening messages
**Rejected:** treating every (customer message, brand reply) pair as a triage unit, which
is what the Phase 1 subsample did.
**Why:** clustering exposed it. 49.8% of pairs were mid-thread customer turns - "Thanks... :)
Its delivered yesterday.", "It says on the way" - which have no intent to classify and no
reply worth drafting. Left in, they formed a large chitchat cluster that would have inflated
intent accuracy for the wrong reason: the classifier would score well by recognising
pleasantries.
This is not new scope. `requirements.md` non-goal 5 already defines the unit of work as
"triage of an inbound message, which is a single-turn decision". The subsample simply was
not enforcing what the requirements already declared, and nobody would have noticed until
the confusion matrix looked suspiciously good.
**Cost:** halves the candidate pool (168,814 -> 84,637) and means the system is never
evaluated on follow-up turns. Acceptable, because handling follow-ups was already an
explicit non-goal.

### D28 — Restrict to English, and report the excluded share
**Rejected:** keeping all languages (more data, more "realistic").
**Why:** at every k from 6 upward, the dominant clustering signal was **language, not
intent**. Two of six clusters were pure Spanish/Portuguese and French/German. A taxonomy
built on that pool would have been substantially a language classifier wearing an intent
label. The reply-classifier heuristics in `data/clean.py` are English-only too, so the
mismatch was already there.
**Cost, and it is large:** `langdetect` dropped 11,419 of 48,000 sampled candidates - ~24%
of real traffic - led by **Japanese (3,738)**, then Spanish, French, German, Portuguese and
Italian. Japanese was invisible to the hand-rolled sizing heuristic I tried first, which is
the argument for using a real detector rather than my own regexes. The taxonomy, and every
number downstream, therefore says nothing about roughly a quarter of AmazonHelp's actual
traffic. Recorded in `subsample_meta.json["funnel"]` per language and carried into the
report's limitations, not buried.
**Determinism note:** `langdetect` is randomised by default. `DetectorFactory.seed = 0` is
set at import, without which the same text can get different languages across runs and the
byte-identity guarantee would break silently.

### D29 — k chosen by interpretability; silhouette reported as uninformative
**Rejected:** selecting k by silhouette (the obvious automatic criterion, and what the
script originally did).
**Why:** silhouette ranged 0.051-0.065 across k=4..14 - a spread of 0.014. That is not a
weak signal, it is no signal: short support text has no clean geometric cluster structure,
so the criterion just rewarded the smallest k and picked k=4. Four intents is both below the
planned range and visibly too coarse. Presenting k=4 as "chosen by silhouette" would have
dressed an arbitrary outcome in a statistic.
So k=10 was chosen by reading clusters: at k=10 every cluster is nameable and distinct,
while k=9 collapses complaints-about-support-quality into a generic service bucket. The
sweep is still committed (`taxonomy_k_sweep.md`) as evidence the automatic criterion did not
discriminate.
**Cost:** the choice is a judgement call, not a computation, and a different reader might
pick 9 or 12. Mitigated by committing the full per-cluster evidence so the judgement is
reviewable.

### D30 — Clusters named from exemplars, never from term lists
**Rejected:** naming clusters from their top TF-IDF terms, which is faster and looks rigorous.
**Why:** it would have produced a fictional intent. Cluster 1's distinctive terms were
`amazon, amazonindia, amazon pay, india` - which reads unmistakably like a payments or
regional intent. Its actual messages are generalised brand rage: "Such a #PoorService given
by Amazon", "never using Amazon again", "fuck you Amazon". Named from vocabulary it would
have become `amazon_pay_india`, an intent that does not exist, and the classifier would then
have been evaluated on its ability to reproduce my mistake.
Clusters 0 and 1 were merged on that reading (both generalised complaint). Far-from-centroid
exemplars are dumped alongside near-centroid ones for the same reason: reading only cluster
centres makes every cluster look cleaner than it is.
**Cost:** slower, and it makes the taxonomy a documented human judgement rather than an
algorithmic output. That is the honest description of what it is.

### D31 — 10 intents + `other`, exceeding the planned 6-9
**Rejected:** folding `unresolved_followup` into `service_complaint` to hit the planned range.
**Why:** that merge would have created a single 26% catch-all - a junk drawer that both
inflates the majority-class baseline (making the trivial baseline harder to beat for the
wrong reason) and hides a genuinely distinct triage case. "I'm still waiting, I've contacted
you twice" needs different handling from "your service sucks".
**Cost:** roughly 18 golden examples per class at n=200 instead of ~22, so wider per-class
confidence intervals. Reported rather than avoided by choosing a convenient taxonomy. The
planned range in `requirements.md` is left as written rather than retrofitted.

### D32 — `account_security` included despite having no cluster of its own
**Rejected:** deriving intents purely from what KMeans found.
**Why:** account compromise is only ~2.5% of opener traffic, so clustering scattered it and
a purely bottom-up taxonomy would not contain it. But it is the single highest-stakes intent
in the set: "someone has hacked my account and changed the login details" is exactly the
message that must never receive an automated reply. A taxonomy with no label for it has a
hole precisely where automation is most dangerous.
**Cost:** two, both real. (a) Intent shares now sum to 1.024 rather than 1.0, because
account_security overlaps the clusters it was drawn from - left visible rather than
normalised into a tidy fiction. (b) The golden set must oversample it to measure it at all,
which makes the golden set **deliberately not distribution-matched**. That biases any
headline accuracy figure upward or downward depending on the class, and goes in the report's
"what is misleading" section rather than a footnote.

### D33 — `ALWAYS_ESCALATE` kept deliberately small (3 of 10 intents, ~30% of traffic)
**Rejected:** gating every intent that feels risky - delivery failures, order lookups,
follow-ups - which was tempting and would have covered ~70%.
**Why:** `requirements.md` section 6 already says a system that achieves safety by escalating
everything is worthless, and the trivial always-escalate baseline exists precisely to make
that failure mode impossible to hide behind. A bloated G1 would make a high safe-to-send rate
a statement about the gate rather than about the system. So G1 covers only the cases where a
wrong auto-reply does real harm: `account_security` (compromise must reach a human),
`refund_billing` (a bot must not commit money), `service_complaint` (auto-replying to "your
service sucks" risks inflaming). ~70% of traffic stays eligible for automation.
`order_investigation` was deliberately left OUT: the brand's genuine handling is "send us
your order number", which is a safe reply and real precedent (D21).
**Cost:** more traffic is exposed to a wrong auto-handle than a cautious policy would allow.
That is the intended trade, and the coverage/risk curve (D9) is what makes it measurable
rather than asserted. A test asserts G1 stays under half the taxonomy so this cannot drift
upward unnoticed.


### D34 — Two independent labellers from different model families, not one
**Rejected:** a single LLM labelling pass (the obvious, cheaper design).
**Why:** one pass produces labels with no way to distinguish a confident label from a
coin-flip. Two labellers from different families (gpt-oss and Qwen) buy three things a
single pass cannot: an inter-labeller agreement statistic that measures how hard the task
actually is; automatic surfacing of the genuinely ambiguous rows, so human review is spent
where it matters; and a check on systematic bias.
**What it found, which is the justification:** intent agreement was reasonable
(**kappa 0.721**) but the auto/escalate decision was **kappa 0.273** - and not from
randomness. gpt-oss escalated 34.3% of messages, Qwen 73.8%. On 63 rows they chose the SAME
intent and the OPPOSITE decision. A single-labeller design would have produced those exact
labels and reported nothing unusual.
**Cost:** double the API calls on a 200k-token/day budget, which is what forced the batching
work in D39.

### D35 — Labellers are told the task, never the policy
**Rejected:** giving labellers the intent dispositions (that `refund_billing` is always
escalate, etc.).
**Why:** the golden auto/escalate labels would then be a restatement of gate G1, and Phase 6
measuring G1 against them would be circular - the gate would score near-perfectly by
construction and the number would mean nothing. Labellers received intent definitions and
boundary notes only, and judged escalation per message.
A second rule alongside it: labellers never saw the brand's actual reply, only the customer
message, which is what the pipeline sees at inference. Showing the historical reply would
leak the answer into the label and import the brand's own mistakes as ground truth.
**Cost:** labellers had less context, which is part of why decision kappa was low. That is
the right trade - a reproducible-but-circular label is worth less than a contested honest one.

### D36 — Escalation rubric v2, rewritten around the REPLY rather than the message
**Rejected:** keeping the v1 rubric, or re-prompting until the two labellers agreed.
**Why:** v1's central test was "does this need private account data?", which is true of
almost every support message. It therefore did no work, and each model fell back on its own
disposition - the measured cause of the 34% vs 74% split. Re-prompting until they agreed
would have manufactured consensus without improving the definition.
v2 asks instead: *does a safe, grounded, non-overcommitting reply exist that a competent
agent would send unedited?* That is answerable. "Please share your order number and we'll
look into it" IS such a reply, so needing an order lookup no longer escalates by itself.
Six explicit tests (E1-E6) replace the vague list, and `src/threadpilot/escalation.py`
carries them.
**Not circular:** sharing a task DEFINITION with the pipeline is necessary - grading a
system against a target it was never told is a trick question. What is not shared is the
MECHANISM: G1 decides by intent, the rubric decides per message, and their disagreement is
the measurement. "How long do refunds normally take?" is `refund_billing` (G1 escalates) but
auto-handleable under the rubric, so G1 is measurably wrong on cases like it.
**Cost:** the 113 disagreements were adjudicated under v2 while the drafts were produced
under v1, so the final labels are not a clean single-rubric artifact. Re-labelling all 200
under v2 would have cost another ~2 days of quota. Disclosed rather than hidden.

### D37 — Audit a random sample of AGREED rows, and report the measured noise
**Rejected:** treating labeller agreement as confirmation.
**Why:** two LLMs from different families can agree and both be wrong, and their agreement
then looks exactly like corroboration. The only way to detect that is to inspect a sample of
the rows where they agreed. Sampled randomly, not by lowest confidence - sampling the
least-confident rows would bias the estimate upward and make it useless for the rest.
**Result: 6 of 40 audited rows were overridden -> ~15% residual label noise** in the 47 rows
accepted on agreement without individual inspection. Two of the six were intent errors where
both labellers keyed on a surface word: "Amazon fraud policy: load your money to Amazon Pay
Balance..." was labelled `account_security` by both because it contains "fraud", when it is
a sarcastic complaint about a refund - `refund_billing`.
**Cost:** the golden set ships with a stated error bar instead of an implied zero. That is
the honest description, and 15% is high enough that it caps what any metric computed on this
set can claim.

### D38 — Distinguish per-DAY quota from per-MINUTE rate limits in the retry policy
**Rejected:** one retry policy for all 429s (what tenacity was doing).
**Why:** a per-minute limit clears in seconds and should be retried; a per-day quota cannot
clear within a run, so the same policy burns six attempts and ~2 minutes of backoff per call
and then fails anyway. Measured cost of conflating them: the first labelling pass took
**5.3 hours for 400 calls**, almost entirely in backoff against an exhausted daily budget,
and still lost 28 rows. After the fix - plus a circuit breaker that stops calling a model
once its daily budget is known gone - the same 200 rows completed in **14 seconds** at 98%
cache hits.
**Cost:** the classifier keys on Groq's wording ("tokens per day", "TPD"), so a provider
rewording it would silently restore the old behaviour. Unit-tested against both phrasings.

### D39 — Batched inference with id round-trip validation and per-item fallback
**Rejected:** one call per item everywhere (simplest, least confounded).
**Why:** every labelling call repeated a ~1,200-token system prompt to classify a sub-100-token
tweet. At 200k tokens/day that capped one model at roughly 140 labelling calls per day, which
is precisely why the first run died. Batching amortises the system prompt.
**The risk, handled explicitly:** a batched response can drop, reorder or invent item ids and
still parse cleanly, which would compute metrics over the wrong subset *invisibly*. So every
item carries an explicit id, the response must return exactly that id set, and an invalid
batch falls back to per-item calls - batching can cost tokens but never lose rows. Tested,
including the reordering case.
**Cost:** possible cross-contamination between neighbouring items, which cannot be argued
away. `scripts/validate_batching.py` measures batched vs unbatched agreement on the same
rows in Phase 4, and any result produced by batching is reported next to that number.

### D40 — The taxonomy has a hole: `other` became the largest class (17%)
**Not a decision so much as a defect found by labelling, recorded rather than patched.**
`other` ended up with 34 of 200 rows, more than any named intent. Inspecting it shows it is
not a residual - it contains at least two coherent missing intents:
* **praise / positive feedback** - "Absolutely fantastic customer service ... Bravo",
  "Fantastic service from Daniela", "Thanks for resolving my issue promptly". The taxonomy
  was derived from clusters of a complaint-dominated corpus, so satisfied customers have
  nowhere to go. Both labellers repeatedly mislabelled these as `service_complaint`, which
  is defined as *dissatisfaction* - a real error caught in adjudication.
* **pre-purchase / product questions** - "When will amazon restock this item?", "Bought
  Nikon B700 & its wifi not working", checkout and promo queries. These concern products
  and purchases rather than an existing order, and no named intent covers them.
**Why not fix it now:** the taxonomy was frozen before labelling on purpose (D14, D12), and
re-deriving it after seeing the golden set would destroy that ordering and let the taxonomy
be fitted to the evaluation data. The defect is therefore reported, and appears in the
report's "what I would do next" as the highest-value single change.
**Cost:** `other` is a 17% catch-all, so the majority-class baseline is inflated and any
`other` per-class metric is close to meaningless - it is measuring a bag of unlike things.


### D41 — Drafting and the decision proposal share one LLM call
**Rejected:** a separate "should this escalate?" call after drafting.
**Why, principled:** the escalation rubric asks whether *a safe, non-overcommitting reply
exists that an agent would send unedited*. That question is about a specific reply, so asking
it in the same breath as writing the reply is the faithful form. Asking it beforehand forces
the model to guess about a draft it has not written.
**Why, practical:** each drafting call carries the retrieved precedent block, which dominates
its token cost. A separate decision call would carry it again, and the budget is 200k
tokens/day.
**What is NOT merged:** the deterministic gates stay entirely outside the call, in
`decide.py`, so the rule layer is still separable for the Phase 6 gates-off ablation. This
call produces only the model's *proposal*.
**Cost:** the model assesses its own output, which invites self-justification. Mitigated by
the gates being able to override it, and measured by how often they do.

### D42 — Gates are one-directional, and that is asserted in both directions
**Rejected:** letting a gate also grant auto_handle when signals look clean.
**Why:** a wrong escalate costs an agent thirty seconds; a wrong auto-handle publishes a
wrong answer under the brand's name. Encoding that asymmetry structurally beats trusting a
model to respect it - especially given Phase 3 measured this exact judgement at kappa 0.273
between two capable models.
**Demonstrated immediately.** The first end-to-end run: "someone has changed the email on my
account and I cannot log in" -> the model proposed **auto_handle**, gate G1 forced escalate.
That is precisely the under-escalation Phase 3 measured in this model family (it escalated
only 34.3% of messages), caught by the deterministic floor. The rule layer earned its place
on its first real input.
**Cost:** G1 caps achievable coverage at ~70% by construction. Phase 6's gates-off ablation
makes the rule layer justify itself numerically rather than on this anecdote.

### D43 — G4 keys on explicit identifiers, not on topic
**Rejected:** "does this need private data?" as the gate test.
**Why:** that is the exact formulation that broke the Phase 3 labelling - it is true of
nearly every support message, so it fires on everything and does no work. G4 therefore
matches an explicit identifier (an Amazon order id, "order no: 12345", a tracking number, or
the dataset's own PII masks), not the subject matter. Tested from both sides: it must fire on
"please check order 403-7128266-4201160" and must NOT fire on "where is my order".
**Cost:** messages that genuinely need account data but mention no identifier slip past G4.
They can still be caught by G2 (no precedent) or by the model's own proposal.

### D44 — Batching validated, and the result is "marginal", reported as such
**Rejected:** adopting batching on the token-savings argument alone.
**Why:** batching risks an item's label being influenced by its neighbours. Measured on 30
golden messages, batched (size 8) vs one-per-call:
* per-item intent agreement **80.0%** (kappa 0.749) - so roughly one in five individual
  predictions is batch-composition dependent;
* but **accuracy against the golden labels is IDENTICAL at 73.3% either way**.
So batching relocates errors rather than adding them, and every disagreement fell on rows
whose gold label was itself contested. Aggregate metrics are safe; per-item predictions carry
~20% instability. Both numbers are reported, and results produced by batching are quoted next
to them.
**Also measured at batch size 4**, where three batches failed Groq's JSON validation and fell
back to per-item calls (28 calls rather than 8). The apparent accuracy difference there
(86.7% vs 73.3%) is four items at n=30 - inside noise, and not read as a real effect. The
useful part is that the **fallback path worked and lost no rows**, which is what it exists
for. Size 8 kept, for fewer JSON failures and 4 calls instead of 28.
**Cost:** n=30 is a small validation. A larger one would cost budget that Phase 6 needs more.

### D45 — The classifier is already at the inter-labeller ceiling
**Not a decision - a measurement that reframes every Phase 6 number.**
The classifier scores **73.3%** against the golden labels. Two independent labellers from
different model families agreed with *each other* only **75.0%** of the time on the same
task. The classifier is therefore performing at roughly the ceiling that the labelling
process itself supports.
**Consequence:** reporting "73% accuracy" against an implicit 100% would badly misrepresent
the result in both directions - it understates the system (the remaining 27% is substantially
irreducible label ambiguity) and overstates what a higher number would mean (a classifier
scoring 90% here would likely be overfitting one labeller's idiosyncrasies). Phase 6 reports
accuracy **against the inter-labeller ceiling**, and the report says so before quoting any
headline figure.


### D46 — Three baselines, not two: a retrieval-only control was added
**Rejected:** the planned pair (trivial + simple) alone.
**Why:** neither answers the question an evaluator should ask first - *what does the LLM
actually add?* The system retrieves precedent AND generates. A TF-IDF baseline shares neither
step, so beating it says little. `retrieval_1nn` uses the **same index the pipeline uses** and
returns the nearest historical reply **verbatim**, with no generation at all. It differs from
`simple_tfidf_cluster` only in where the reply text comes from (they are deliberately given
the same classifier), so the pair isolates the generation step precisely.
If copying the closest historical reply scores as well as a generated one, then generation is
decoration and that is the honest finding. A baseline that can embarrass the system is worth
more than one that cannot.
**Cost:** a third baseline to run through every eval path. Cheap - no baseline makes an LLM
call, so all of them are free to refit.

### D47 — The TF-IDF baseline is fitted under TWO training regimes
**Rejected:** picking one training set and reporting a single "simple baseline" number.
**Why:** the golden set is the test set, so the only genuinely labelled non-test data is the
100-row dev_silver pool - about 9 rows per class. A baseline that weak would be a straw man.
The alternative is ~6,948 subsample rows pseudo-labelled by their Phase 2 cluster: free, far
larger, noisier, and with golden and silver threads excluded so it leaks nothing.
Which one is chosen changes the verdict, so choosing silently would be a way of rigging the
comparison. Both are reported:
* `simple_tfidf_silver` - 100 rows, 11 classes: 30.0% accuracy, macro-F1 0.263
* `simple_tfidf_cluster` - 6,948 rows, 9 classes: 36.0% accuracy, macro-F1 0.332
Together they separate two different advantages the LLM might have: *reasoning better* versus
simply *not needing labelled data at all*.
**Structural limitation, asserted in a test rather than left implicit:** the cluster-trained
classifier has only 9 classes, because `account_security` and `other` have no Phase 2 cluster.
It can therefore **never** predict the highest-stakes intent in the taxonomy. That is a real
advantage of the LLM approach and the comparison must be read with it in view.

### D48 — LogisticRegression over a calibrated LinearSVC, because the data-poor pool broke it
**Rejected:** `CalibratedClassifierCV(LinearSVC, cv=3)`, the first implementation.
**Why:** it raised on the silver pool - 100 rows across 11 classes leaves at least one class
with fewer than 3 examples, so 3-fold calibration is impossible. That is not a nuisance to
engineer around; it is a **measurement**: the realistic labelled-data regime is too small to
calibrate a classifier at all, and it is reported rather than hidden behind a workaround.
LogisticRegression gives `predict_proba` natively with no inner CV, so both training regimes
fit under identical code and their confidence thresholds stay on the same axis as the
pipeline's `TAU_CONF` - which is what makes the coverage/risk curves comparable.
**Cost:** logistic regression probabilities on tiny data are optimistic. Reported next to the
coverage curve rather than treated as calibrated truth.

### D49 — The trivial baseline confirmed the metric-gaming problem empirically
**Not a decision; the measurement the trivial baseline exists to produce.**
Measured on the golden set: **0.0% false-auto-handle rate at 0.0% coverage**, alongside 11.0%
intent accuracy and macro-F1 0.018. It is simultaneously the *safest* system by the obvious
safety metric and the *most useless* one.
This is why auto-handle is reported as a coverage/risk curve rather than a headline
percentage (D9). Any single-number safety metric ranks this system first. The number is now
in the results rather than being an argument I make in prose.
**The simple baselines fail in the opposite direction:** 53-56% false-auto-handle rates at
58-82% coverage - they automate recklessly. Neither pole is acceptable, and the curve is what
shows a system sitting between them.


### D50 — A quota stop must halt the run, never become a prediction
**The bug, and why it is the worst kind.** `batching.complete_batch` caught every exception
except `CacheMissInOfflineMode` and converted it into a per-item fallback, then into `None`,
then into `draft.FAILED`. When the daily token quota ran out mid-run, that machinery
faithfully turned one infrastructure failure into **184 of 200 rows recorded as legitimate
predictions with empty replies** - and the run exited 0. Every gate fired on them (G5 and G6
184 times each), so the output even looked internally consistent: 191 escalates, 9
auto-handles, a plausible-looking distribution. Nothing in the results said "the budget ran
out".
This is precisely the silent-corruption failure the rest of that module was written to
prevent, and I built the hole myself by writing `except Exception` around a code path whose
whole job is to distinguish real failures from infrastructure ones.
**Fix:** `DailyQuotaExhausted` now propagates through both `complete_batch` and the per-item
fallback, exactly like `CacheMissInOfflineMode`. `run_predictions.py` catches it at the top
level, reports how far it got, and exits - everything already completed is cached, so a
re-run resumes cheaply. The corrupted `pipeline.jsonl` was deleted rather than patched.
**Lesson worth keeping:** a fallback path is a place where errors get *reclassified*, and any
`except Exception` there silently reclassifies "the system is broken" as "this item is hard".
The guards I wrote for missing ids and reordered ids were the right instinct applied to the
wrong exception.

### D51 — Prompt budget is a design constraint, and it was measured not guessed
**Rejected:** leaving the drafting prompt as written and running over several days.
**Why:** the first full attempt cost ~185k tokens for drafting alone against a 200k/day cap,
and died mid-run. Measuring where the tokens went made the fix obvious rather than a matter
of taste - precedent text dominated: `Precedent.render()` emitted 2 fields x 240 chars x 5
precedents, roughly 2,400 characters per item before anything else.
Three changes, each measured: precedent rendering 240 -> 130 chars; only the top **3** of the
5 retrieved precedents go into the prompt (gate G2 still sees all 5 for max-similarity); and
the drafting prompt no longer embeds the full rubric verbatim. Drafting prompt fell 836 -> 481
tokens; a batch of 6 now costs ~2,195 input tokens where a batch of 4 cost ~2,741.
Projected pipeline cost: **~165k, inside the daily cap**.
**Cost, stated:** the drafter now sees 3 precedents rather than 5, and shorter excerpts of
each. That could plausibly reduce grounding quality. It is a real change to the system being
measured, not a free optimisation, and the no-retrieval ablation gives the reference point
for how much precedent matters at all.


### D52 — The judge failed validation, and the failure is diagnosable
**Result:** blind re-scoring of 40 pipeline replies against the judge gave **Cohen kappa
0.333** on the headline `send_unedited` binary (70% raw agreement), with the judge
systematically MORE generous on every ordinal criterion: grounded +0.90, on_intent +1.07,
voice +0.97, actionable +0.97 on a 1-5 scale. Judge says 78% sendable; blind rater says 60%.
**The pattern is not noise.** Reading the 12 disagreements, the judge justifies its scores by
FORM - "mirrors precedent", "standard support link", "matches precedent for security issues" -
while the blind rater judged SUBSTANCE - "answers nothing", "never answers whether this
channel is right", "already described the fault". The judge cannot detect a well-formed reply
that fails to address the specific customer. It rated a reply to a **break-in with police
involved** as sendable because it "matches precedent for security/delivery issues"; the reply
was "Sorry about this. Please reach out to us directly here."
**What this costs the report.** The 78% send-unedited figure cannot carry a headline. It is
reported with the kappa attached and with the blind rater's stricter 60% shown beside it.
**What survives.** The RELATIVE ordering is probably intact. If the judge merely rewarded
precedent-conformity, `retrieval_1nn` - which IS precedent, copied verbatim - would have
scored highest; it scored 48% against the pipeline's 78%. So the bias plausibly shifts levels
rather than ranks. That is an inference, not a measurement, and is labelled as one.
**Why this is the most valuable thing in the eval.** A quality number without an agreement
statistic is an unvalidated claim, and this one would have been wrong by 18 points in the
flattering direction. The study cost 40 blind re-scores and changed what the report can say.

### D53 — The no-retrieval ablation degenerates, which is itself the finding
**Observed:** `pipeline_no_retr` scores 0% coverage and a trivially perfect 0% false
auto-handle. Not a bug: with retrieval disabled `max_similarity` is 0, so gate G2 ("no
comparable precedent") fires on every row and the system collapses into always-escalate.
**Consequence:** its DECISION metrics are uninformative by construction and are not quoted as
an ablation result. The meaningful comparison is reply quality, where all 200 drafts differ
from the retrieval-grounded ones.
**Worth noting for the design:** this shows G2 is doing exactly what it was specified to do -
refusing to automate without precedent - and that the retrieval and gating layers are
coupled. An ablation that changes one silently changes the other, which is why the ablation is
reported with its mechanism rather than as a bare number.

### D54 — Judging runs in priority order, after a budget-limited run scored only baselines
**The mistake:** judging iterated `SYSTEM_ORDER`, which lists the main pipeline LAST. A
quota-limited run therefore spent its entire daily token budget scoring four baselines and
stopped before reaching the system under test - a whole day of budget producing nothing that
could answer the central question.
**Fix:** judging iterates a PRIORITY order (pipeline, ablations, then baselines) and skips
systems already judged. Under a hard budget the ordering of work is part of the design, not
an implementation detail.


### D55 — The fresh-clone pass found the reproducibility promise was false
**What Phase 9 caught.** `python -m eval.run_eval --offline` - the README's headline promise
that every reported number regenerates with no API key - **crashed on a clean checkout**:
`FileNotFoundError: no retrieval index`. It worked only on my machine, because my machine had
`data/interim/index_vectors.npy`. That directory is gitignored, and correctly so: the index
metadata carries tweet text from a CC BY-NC-SA dataset, so committing it would redistribute
the same content the subsample is withheld to avoid (D26).
**Why it survived until now.** Every previous run of the harness happened in a working tree
that already had the index. There is no way to catch that except by actually starting from a
clean tree, which is exactly what the phase exists for - and it is the difference between a
reproducibility claim and a reproducibility check.
**Fix:** the index is loaded lazily, only when a system genuinely needs judging. When all
judge results are cached - the offline case - it is never touched. Verified by re-running in a
tree with no `data/interim/`, no subsample, and `GROQ_API_KEY=""`: 7 systems reproduced in 4s.
**Measured result of the full pass (fresh venv, README steps only):** setup 119s, check 1s,
tests 12s, offline eval 32s, **total 2m 44s** against a 15-minute promise. Recorded in the
README as measurements rather than estimates.
**Honest detail that came with it:** 33 of 153 tests SKIP on a fresh clone because they need
the retrieval index or the subsample. They skip loudly rather than passing vacuously, and the
README says 120/153 rather than claiming all of them run.

---

*Further decisions are appended as Phases 1-9 proceed.*

### D19 — Generator `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`
**Rejected:** `llama-3.3-70b-versatile` (planned, and it does not exist);
`groq/compound` (agentic); `qwen3.6-27b`; using one model for both roles.
**Why:** verified against the live `/models` endpoint on 2026-09-09 — evidence committed at
`docs/groq_models_2026-09-09.json`. Groq's free tier serves **no Llama chat model at all**,
so the model this project was planned around is gone. Of the 14 available ids, most are not
usable here: Whisper (ASR), Orpheus (TTS), `llama-prompt-guard` (512-token safety
classifiers), `allam-2-7b` (4k context, Arabic-focused), and `groq/compound*` (agentic
systems with built-in tool use — nondeterministic, would confound the eval). That leaves
`gpt-oss-{120b,20b}` and `qwen3.{6,8}-27b`. `gpt-oss-120b` is the largest general model
available and takes generation; `qwen3.8-27b` takes judging because Alibaba Qwen and OpenAI
gpt-oss are genuinely different lineages, which is what D7 actually needs. `qwen3.6-27b` was
rejected for leaking `<think>` blocks into `content`; `qwen3.8-27b` does not.
**Cost:** the judge (27B) is smaller than the generator (120B), so a weaker model grades a
stronger one. Not hidden: Phase 6 scores the judge against human re-scores and reports the
agreement, and `JUDGE_MODEL_ALT` runs the same validation with `gpt-oss-120b` so the two
candidate judges are compared on agreement-with-human rather than on my preference.
The alt judge shares a family with the generator, so a high score from it is suspect by
construction and is reported as a comparison point only.

### D20 — Empty LLM content is treated as an error, never as a valid response
**Rejected:** trusting `finish_reason == "stop"`.
**Why:** found by smoke test, not by reasoning. `gpt-oss-120b` at `max_tokens=120` returned
`finish_reason="stop"` with **empty** `content`, having spent the entire budget on its
`reasoning` channel. A pipeline that accepted that would silently record blank drafts and
missing labels as legitimate outputs, and the eval would quietly measure nothing. So
`llm.py` raises on empty content and retries with a larger budget, and per-call token
budgets are set generously in `config.py`.
**Cost:** a little extra latency on the retry path, and token budgets larger than a
non-reasoning model would need. Cheap insurance against a silent, metric-corrupting failure.
