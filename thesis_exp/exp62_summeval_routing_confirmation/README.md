# Exp62: SummEval independent routing confirmation

Exp62 is the preregistered B1 confirmation experiment. It asks whether the
development-set routing pattern observed on EduBench reproduces on an
independent, genuine multi-rater ordinal dataset.

The core experiment uses SummEval's three expert ratings for **coherence** and
**fluency**. The released annotation JSONL contains candidate summaries but not
the corresponding CNN/DailyMail source articles, so **consistency** and
**relevance** are excluded unless those source articles are recovered and
separately locked. Reference summaries must never be substituted for source
articles.

Stage 0 is CPU-only and creates a source-article-grouped 70/15/15 split. Formal
training remains disabled until the protocol, input template, token limit,
four-arm implementation, five new seeds, and test-access rule are frozen.

Planned arms:

1. `direct_residual_blocked`
2. `routed_hmsa`
3. `orthogonal_only`
4. `parallel_only`

Planned seeds: 62, 63, 64, 65, 66.

