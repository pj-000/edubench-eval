# Exp54 V2 dev authorization plan

This plan authorizes exactly one sealed 36-checkpoint dev campaign after the
V2 implementation commit and its authorization payload receive independent
review. It does not authorize test access.

## Non-activating preparation

1. Check out the exact reviewed V2 implementation commit.
2. Generate one authorization JSON that binds the reviewed commit, candidate
   report and lock, protocol, grammar, decoder, parser, sealed dev hash, the
   complete runtime closure, exact runtime/distribution identities, exact
   wheel bytes, all 36 checkpoint artifacts, batch size 16, token budget 256,
   fixed output directories, `test_allowed=false`, and one invocation per
   checkpoint.
3. Independently review the authorization bytes and SHA-256.
4. Install the exact JSON at
   `~/.config/edubench/exp54_v2_dev_authorization.json` with mode `0444`.
5. Install its reviewed digest at
   `~/.config/edubench/exp54_v2_dev_authorization.sha256.staged` with mode
   `0444`. The staged file is not an activation point.
6. Prepare private user-owned claim and output roots. The campaign claim
   directory must be empty.

No `sudo` command is required. The production runner reads neither a CLI
authorization path nor the staged digest.

## Pre-activation gate

Run `audit_v2_dev_authorization_preactivation.py`. It must return exactly:

`V2_DEV_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS`

The audit fails if the active V2 digest exists, the formal-training anchor
exists, installed authorization bytes or permissions differ, any checkpoint
or runtime binding differs, a claim already exists, or a fixed output
directory exists. The audit is read-only: it does not claim a task, create an
output, load a model, initialize CUDA, or read test.

## Unique activation point

Only after the pre-activation PASS, atomically rename in the same directory:

`exp54_v2_dev_authorization.sha256.staged`

to:

`exp54_v2_dev_authorization.sha256`

The active digest appearing at that fixed path is the only activation point.
Before it exists, the production runner fails before dev loading, model
validation/loading, output creation, or CUDA.

## Execution and abort rules

Run the fixed launcher, which invokes only S0/R1/R2/R3 × seeds 42/43/44 ×
logical epochs 1/2/3 with batch size 16 and token budget 256. Each invocation
atomically creates a unique claim before any dev/model/output work. A second
invocation of the same checkpoint fails.

If a claimed task fails, retain its claim and output evidence, remove the
active digest, and stop the campaign. Do not retry inside the same campaign.
A retry requires a new campaign, new authorization, and new review.

After all 36 tasks complete, remove the active digest and audit results.
Completion does not authorize test access or checkpoint/protocol selection.
