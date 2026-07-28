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
6. A system administrator prepares the fixed claim root
   `/var/lib/edubench/exp54-v2-dev` as a real root-owned directory with exact
   mode `0755`.
7. The administrator creates
   `/var/lib/edubench/exp54-v2-dev/<campaign_id>` as a real root-owned,
   training-group directory with exact mode `01770` and the filesystem
   append-only flag. Group read permission is required so the execution-user
   preflight can read the append-only flag and prove that the directory is
   empty. The execution user must belong to that group. The execution user
   may create a claim but cannot delete or rename an existing claim.
8. Prepare the private user-owned output root. Every fixed final output
   directory must be absent.

The production runner itself never invokes `sudo`, creates a campaign
directory, changes ownership/mode/flags, reads a CLI authorization path, or
reads the staged digest. Root-managed claim provisioning is a separate
installation prerequisite; this plan does not authorize Codex to perform it.

## Pre-activation gate

Run `audit_v2_dev_authorization_preactivation.py`. It must return exactly:

`V2_DEV_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS`

The audit fails if the active V2 digest exists, the formal-training anchor
exists, installed authorization bytes or permissions differ, any checkpoint
or runtime binding differs, claim-root/campaign owner, group, mode or
append-only state differs, a claim already exists, or a fixed output
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
authenticates hashes without parsing dev rows, atomically creates a unique
claim, reserves its output, and only then materializes dev rows from the
already authenticated in-memory payload. A concurrent loser never receives
materialized dev rows. Deleting an output directory does not restore
invocation permission, because the execution user cannot remove the
root-protected claim.

If a claimed task fails, retain its claim and output evidence, remove the
active digest, and stop the campaign. Do not retry inside the same campaign.
A retry requires a new campaign, new authorization, and new review.

After all 36 tasks complete, remove the active digest and audit results.
Completion does not authorize test access or checkpoint/protocol selection.
