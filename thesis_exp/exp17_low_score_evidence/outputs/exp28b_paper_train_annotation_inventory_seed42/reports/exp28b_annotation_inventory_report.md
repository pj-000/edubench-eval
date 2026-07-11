# Exp28B Paper-Train Annotation Inventory

## Status

**READY FOR PROTOCOL DRY RUN**

- paper-train rows: 2654
- blind teacher packets: 2654
- benchmark reference rows: 2654
- legacy Exp27 rows that map to paper train: 190
- legacy rows used as new targets: 0

The blind packet contains only question, answer, metric, rubric, and metadata. Original labels
and individual human scores live in a separate private reference packet.

## Protocol Subsets

- protocol demonstration references: 15 (benchmark labels; not expert-reviewed)
- protocol development: 60
- sealed qualification: 120
- full annotation pool: all 2,654 paper-train rows

The three pilot subsets use disjoint question keys. They are all drawn from paper train. Dev and
test identities are used only for leakage guards, and held-out labels are not read.

## Teacher Routing

1. The primary teacher will annotate all 2,654 rows after protocol qualification.
2. A secondary teacher is routed to low-score, high-disagreement, primary/original-conflict,
   low-confidence, and locked high-control cases.
3. Remaining high teacher conflicts receive independent model adjudication recorded as
   `model_review_silver`.
4. No model output is described as human review.
