# CFSP-PO train-only feasibility research specification

Status: `DESIGN_ONLY_NOT_AUTHORIZED_FOR_GPU_TRAINING`

Version: `cfsp-po-feasibility-v2`

## 1. Relationship to the completed thesis experiments

The frozen RAR-SFT and Field-DPO/SORC-DPO experiments remain the thesis
baseline and are not modified by this extension.

The existing preference data provide two useful positive controls:

- score pairs change `score` while keeping `rationale` byte-identical;
- rationale-alignment pairs change `rationale` while keeping `score`
  byte-identical.

Their active field is therefore known from the data construction. They test
whether a field-local preference loss can optimize a field when its scope is
given, but they do not test whether the scope of a preference can be recovered
automatically. CFSP-PO addresses that missing scientific question.

Existing dev and test results must not be used to choose the CFSP-PO definition,
counterfactual construction, utility function, thresholds, or hyperparameters.
In particular, the existing test split is excluded from every feasibility
stage described below.

## 2. Scientific question

For structured generation with output fields

\[
y=(y^{(1)},\ldots,y^{(K)}),
\]

can counterfactual field swaps recover which fields are responsible for a
preference, and can the recovered responsibility be used to improve the
responsible fields while limiting drift in non-responsible fields?

This question is stronger than the current Field-DPO question:

- Field-DPO assumes the active field is supplied by the pair builder.
- CFSP-PO estimates a soft responsibility vector from counterfactual utility
  changes and uses it to scope the preference update.

The intended contribution is responsibility discovery plus scope-consistent
optimization, not another manually chosen score/rationale loss weighting.

## 3. Counterfactual Shapley responsibility

Let \(F=\{1,\ldots,K\}\). For a preferred output \(y^+\), a rejected output
\(y^-\), and every subset \(S\subseteq F\), define the mixed output:

\[
y^{S(k)}
=
\begin{cases}
y^{+(k)}, & k\in S,\\
y^{-(k)}, & k\notin S.
\end{cases}
\]

Given a frozen utility function \(U(x,y)\), the signed responsibility of field
\(k\) is its exact Shapley value:

\[
\phi_k
=
\sum_{S\subseteq F\setminus\{k\}}
\frac{|S|!\,(K-|S|-1)!}{K!}
\left[
U(x,y^{S\cup\{k\}})-U(x,y^S)
\right].
\]

For two fields this reduces to the symmetric two-context field-swap equation.
For three or more fields, all \(2^K\) mixed outputs are required so that field
interactions are averaged across every coalition context rather than only the
all-chosen and all-rejected extremes.

Negative responsibility is clipped only for constructing the optimization
weight:

\[
\widetilde \phi_k=\max(\phi_k,0),
\qquad
a_k=
\frac{\widetilde \phi_k}
{\sum_{j=1}^{K}\widetilde \phi_j+\epsilon}.
\]

The implementation must retain the unclipped \(\phi_k\) values for diagnostics.
If every \(\widetilde \phi_k=0\), the pair is `scope_unresolved` and must not be
silently converted to uniform field weights.

Here, a positive \(\phi_k\) means that field \(k\) supports the declared
preference, a negative \(\phi_k\) means that it opposes the declared preference,
and zero means that it is utility-neutral under the frozen utility. The signed
responsibility vector and the positive optimization-scope vector are therefore
different audit objects and must not be conflated.

The auditor must also report a field-interaction index derived from deviations
between conditional marginal contributions. Shapley allocates interactions;
it does not prove that fields are causally independent.

## 4. Scope-consistent preference objective

Let \(T_k(y)\) be the token positions belonging to field \(k\). Define the
length-normalized field log probability:

\[
\ell_\theta^{(k)}(y\mid x)
=
\frac{1}{|T_k(y)|}
\sum_{t\in T_k(y)}
\log \pi_\theta(y_t\mid x,y_{<t}).
\]

For each field, define the policy-versus-reference preference margin:

\[
\Delta_k
=
\left[
\ell_\theta^{(k)}(y^+\mid x)
-
\ell_\theta^{(k)}(y^-\mid x)
\right]
-
\left[
\ell_{\mathrm{ref}}^{(k)}(y^+\mid x)
-
\ell_{\mathrm{ref}}^{(k)}(y^-\mid x)
\right].
\]

The scoped preference loss is:

\[
\mathcal L_{\mathrm{scope}}
=
-
\log \sigma
\left(
\beta\sum_{k=1}^{K}a_k\Delta_k
\right).
\]

Let \(y^{\mathrm{corr}}=y^F=y^+\) be the preferred, scope-corrected
teacher-forced sequence after counterfactual coherence validation, and let
\(h_{k,t}^{\mathrm{corr}}=(x,y^{\mathrm{corr}}_{<t})\). To limit unintended
changes in fields with low positive responsibility, compare policy and
reference distributions under this same corrected prefix:

\[
\mathcal L_{\mathrm{pres}}
=
\sum_{k=1}^{K}(1-a_k)
\frac{1}{|T_k(y^{\mathrm{corr}})|}
\sum_{t\in T_k(y^{\mathrm{corr}})}
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{ref}}(\cdot\mid h_{k,t}^{\mathrm{corr}})
\;\|\;
\pi_\theta(\cdot\mid h_{k,t}^{\mathrm{corr}})
\right).
\]

For a score-only preference whose score precedes the rationale, both
distributions are therefore evaluated after the same corrected score prefix.
The method must never compare rationale distributions conditioned on different
scores and call the difference non-target drift.

The complete proposed objective is:

\[
\boxed{
\mathcal L_{\mathrm{CFSP}}
=
\mathcal L_{\mathrm{scope}}
+
\lambda_{\mathrm{pres}}\mathcal L_{\mathrm{pres}}
}
\]

No ODPO offset, dynamic beta, GRPO component, learned reward model, or
additional risk weighting is part of the feasibility experiment.

## 5. Falsifiable hypotheses

### H1: responsibility recoverability

On a controlled structured-generation benchmark with deterministic,
decomposable utility and known true field scope, counterfactual responsibility
recovers the true responsible field set.

Primary measurements:

- top-1 positive-responsibility field accuracy;
- exact positive-responsibility set accuracy;
- macro F1 over positive-responsibility fields;
- multi-label Jaccard;
- signed responsibility-vector exact recovery;
- L1 error between normalized recovered responsibility and the oracle
  responsibility vector;
- scope entropy and single-field versus multi-field stratification;
- unresolved-pair rate.

For the deterministic additive cases, exact oracle recovery is a hard
requirement rather than an average performance target.

For learned or non-additive attribution in later stages, automatic scope must
be compared with uniform, random, policy-confidence, token-difference, simple
single-swap, and oracle-scope controls before preference training is
authorized.

### H2: target-field effectiveness

When H1 holds, CFSP-PO improves preference accuracy or task utility on
preference-supporting fields relative to:

- full-output DPO;
- uniform field-weight DPO;
- CFSP-PO without the preservation term.

Manually scoped Field-DPO using oracle scope is an upper-reference control.
CFSP-PO is expected to approach it without receiving the oracle scope, not
necessarily outperform it.

### H3: non-target preservation

At comparable optimizer steps and pair exposure, CFSP-PO reduces
non-responsible field drift relative to full-output DPO and CFSP-PO without
preservation.

Drift must be reported directly using field-level token KL, exact field
retention where applicable, and task-specific non-target utility. Overall
output quality alone is not sufficient evidence for H3.

## 6. Phase A: controlled known-scope benchmark

The first benchmark is generated from train-only synthetic templates and
contains no records, reasons, IDs, or reversible mappings from the private
education dataset.

Each output contains three independently editable fields:

1. `decision`: a categorical answer determined by explicit input facts;
2. `evidence`: a canonical subset of input facts supporting the decision;
3. `style`: a deterministic surface constraint such as requested language or
   response register.

The generator must create at least these pair families:

- exactly one responsible field;
- exactly two responsible fields;
- all fields responsible;
- a preferred and rejected output differing in a nuisance field whose utility
  is equal;
- opposing field changes where one field improves and another degrades;
- a zero-total-gap pair used only to test `scope_unresolved` handling.

The frozen utility is computed from deterministic field-level rules:

\[
U(x,y)=
w_d U_d(x,y^{(d)})
+
w_e U_e(x,y^{(e)})
+
w_s U_s(x,y^{(s)}).
\]

No LLM judge, evaluator API, model-generated preference label, or education
test data may contribute to this utility.

Data are divided by independent generator seeds into train, development, and
held-out audit sets. Templates or content atoms assigned to one split must not
appear in another split.

## 7. Counterfactual validity and audit invariants

Every generated or swapped output must:

- conform to one canonical JSON schema;
- contain exactly the expected fields;
- preserve the non-swapped fields byte-for-byte;
- take the swapped field byte-for-byte from the declared source output;
- reparse to the same canonical object after serialization;
- have deterministic IDs derived from protocol version, split, seed, example
  coordinates, and pair family;
- be reproducible from the public generator without private data.

For deterministic additive utility, the auditor must verify:

\[
\sum_k \phi_k = U(x,y^+)-U(x,y^-)
\]

up to exact rational arithmetic or a declared numerical tolerance, and must
compare every recovered \(\phi_k\) with an independently computed field-level
oracle.

The benchmark builder and auditor must not share the implementation used to
compute the expected responsibility vector.

For later natural-data counterfactuals, schema validity is necessary but not
sufficient. A separately frozen coherence validator must identify mixtures
such as a low score paired with a rationale that clearly supports a high
score. Reports must include the rejection rate and evaluator-family agreement.
Such mixtures are called controlled counterfactual interventions, not literal
causal interventions on independent fields.

## 8. Decision gates and stop rules

### Gate A: construction integrity

Required before any model training:

- 100% schema-valid original and swapped outputs;
- 100% deterministic regeneration from the frozen configuration;
- exact split disjointness;
- no private education data or existing test access;
- complete source/config/report hash chain.

Failure requires fixing the generator or abandoning the affected pair family.

### Gate B: analytic responsibility recovery

Required before implementing the CFSP-PO training loss:

- 100% exact responsibility recovery on deterministic additive, non-degenerate
  cases;
- 100% correct identification of deliberately unresolved cases;
- zero counterfactual provenance failures;
- equality between total recovered responsibility and total utility gap.

If Gate B fails because the definition itself cannot recover known scope under
its stated assumptions, CFSP-PO stops. The experiment must not proceed by
tuning thresholds on downstream model results.

### Gate C: CPU implementation

Required before GPU use:

- causal-shift and field-mask tests;
- policy/reference margin tests;
- responsibility normalization and unresolved-pair tests;
- preservation-KL scope tests;
- numerical equivalence with the written equations;
- corrected-prefix preservation tests for autoregressive field dependence;
- no score/rationale-specific branches in the generic CFSP-PO loss.

### Gate D: controlled training

Only after Gates A--C pass:

- compare full-output DPO, uniform field DPO, oracle-scope Field-DPO, CFSP-PO
  without preservation, and complete CFSP-PO;
- keep base model, pair order, optimizer steps, scheduler, beta, batch
  semantics, and selection rule fixed;
- select no method using the existing education test set.

If CFSP-PO cannot outperform full-output or uniform-scope controls in target
utility while reducing non-target drift, its general method claim is rejected.
The completed RAR-SFT/Field-DPO thesis results remain valid as a separate
task-specific contribution.

## 9. Education-task follow-up boundary

The existing education score pairs and rationale pairs may later be used as
known-scope positive controls:

- score-only difference should recover score responsibility;
- rationale-only difference should recover rationale responsibility.
- parser-invalid versus schema-valid output should recover format
  responsibility.

They cannot by themselves establish automatic scope discovery because their
scope was fixed by construction.

Automatic rationale responsibility on natural model errors requires a
separately frozen, non-circular utility source. Until such a source is defined
and independently validated, rationale responsibility on real education
outputs is outside the first CFSP-PO claim.

Any education-task experiment must use train/dev only during development.
The already observed test results remain descriptive evidence for the frozen
thesis method and are not a CFSP-PO development target.

After controlled and education train/dev gates pass, the method must be tested
on at least one external structured-generation task whose fields and
deterministic task metrics differ from education scoring. This is required to
separate a general preference-scope method claim from an education-specific
engineering result.

A new untouched confirmation set, if created, is evaluated only after the
scope estimator, preservation weight, checkpoints, contrasts, metrics, and
bootstrap protocol are frozen. The already observed education test set cannot
serve this role.

## 10. Next executable milestone

Implement only:

1. the public controlled benchmark generator;
2. an independent responsibility oracle and auditor;
3. deterministic unit tests and aggregate reports.

Do not implement or launch CFSP-PO training until Gates A and B pass.
