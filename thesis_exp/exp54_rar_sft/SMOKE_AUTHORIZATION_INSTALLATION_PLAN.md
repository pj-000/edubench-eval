# Exp54 smoke authorization installation plan

## Current gate

This document is a plan only. It does not authorize executing any command
below. The authorization JSON, digest candidate, claim directory, output
directory, and trust anchor must remain uninstalled until the exact candidate
and this plan receive a separate review pass.

Current state:

- reviewed smoke guard commit:
  `b2e616e324ed9de7c8aaf8eb1dee5a2f0d7a53bb`
- campaign: `exp54-smoke-b2e616e-v1`
- training identity: `jpang:jpang`, UID/GID `1025:1025`
- server filesystems: XFS
- `/usr/bin/chattr` and `/usr/bin/lsattr`: available
- installed authorization: absent
- staged trust anchor: absent
- installed trust anchor: absent
- formal-training trust anchor: absent
- claim root: absent
- smoke output root: absent
- claim creation: forbidden
- model loading: forbidden
- forward/backward and optimizer steps: forbidden

## Reviewed candidate artifacts

The next review must bind the exact bytes and SHA-256 values of:

- `configs/smoke_execution_campaign_candidate.json`
- `protocol/smoke_execution_authorization_candidate.json`
- `protocol/smoke_execution_authorization_candidate.sha256`
- `audit/smoke_execution_authorization_candidate_report.json`
- `build_smoke_authorization_candidate.py`
- `audit_smoke_authorization_candidate.py`
- `audit_installed_smoke_authorization.py`
- `SMOKE_AUTHORIZATION_INSTALLATION_PLAN.md`

The authorization binds the already-frozen smoke package lock, training
configuration lock, materialized-manifest lock, smoke plan, and 20-file
runtime-source closure. It authorizes only S0/R1/R2/R3, seed 42, one optimizer
step per arm, and one invocation per arm.

## Fail-closed installation order

Only after a separate authorization-review pass may a root administrator
perform these phases. Every phase must stop on the first mismatch.

### 1. Recheck immutable inputs and absence

Verify all candidate hashes against the reviewed values. Use `lstat`-equivalent
checks to require that these paths do not exist as files, directories, or
symlinks:

```text
/etc/edubench/exp54_smoke_authorization.json
/etc/edubench/exp54_smoke_authorization.sha256.staged
/etc/edubench/exp54_smoke_authorization.sha256
/etc/edubench/exp54_authorization.sha256
/var/lib/edubench/exp54-smoke
/home/jpang/edubench-eval-exp2/thesis_exp/outputs/exp54_rar_sft/rar_v2/smoke_runs/exp54-smoke-b2e616e-v1
```

Reconfirm:

- `jpang` has UID 1025 and primary group `jpang` has GID 1025;
- the claim filesystem is XFS and supports the append-only inode flag;
- `chattr` and `lsattr` resolve to the reviewed system binaries;
- none of the four fixed final output directories exists;
- no formal or smoke trust anchor is installed.

### 2. Prepare directories while execution remains unauthorized

The authorization digest must still be absent throughout this phase.

Create:

```text
/var/lib/edubench/exp54-smoke
  owner root:jpang
  mode 0750

/var/lib/edubench/exp54-smoke/exp54-smoke-b2e616e-v1
  owner root:jpang
  mode 0770
  append-only flag enabled with chattr +a

/home/jpang/edubench-eval-exp2/thesis_exp/outputs/exp54_rar_sft/rar_v2/smoke_runs
  owner jpang:jpang
  mode 0750

/home/jpang/edubench-eval-exp2/thesis_exp/outputs/exp54_rar_sft/rar_v2/smoke_runs/exp54-smoke-b2e616e-v1
  owner jpang:jpang
  mode 0750
```

Do not create the four arm-specific final output directories. The runner must
create each one with `mkdir(exist_ok=False)` only after its arm-specific claim
has succeeded.

### 3. Install the authorization JSON, but not the trust anchor

Create `/etc/edubench` as a real `root:root` directory with mode `0755` if it
does not already exist. Install the reviewed authorization candidate through a
temporary regular file in the same directory, verify byte equality and
SHA-256, set `root:root` ownership and mode `0444`, then atomically rename it
to:

```text
/etc/edubench/exp54_smoke_authorization.json
```

At this point execution must still fail because the trusted digest path is
absent.

### 4. Install only the staged digest

Install the reviewed one-line digest candidate through a temporary regular
file in `/etc/edubench`. Require:

- exact byte equality with the reviewed digest candidate;
- a single lowercase 64-hex SHA-256 followed by one newline;
- `root:root` ownership;
- mode `0444`;
- the authorization JSON SHA-256 equals that digest;
- claim campaign directory is append-only;
- all four claim files are absent;
- all four final output directories are absent.

Atomically rename the verified temporary digest file only to:

```text
/etc/edubench/exp54_smoke_authorization.sha256.staged
```

The production smoke trust-anchor path and formal-training trust-anchor path
must both remain absent:

```text
/etc/edubench/exp54_smoke_authorization.sha256
/etc/edubench/exp54_authorization.sha256
```

The staged digest is not an activation point. The production runner never uses
the staged path because its default trusted-digest path remains the absent
production path.

### 5. Run pre-activation preflight without claiming

Run only:

```text
python -m thesis_exp.exp54_rar_sft.audit_installed_smoke_authorization
```

The auditor must return:

```text
SMOKE_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS
```

Before any other check, it hard-fails if either the active smoke trust anchor
or formal-training trust anchor exists. It then verifies exact installed
authorization bytes, staged digest bytes, root ownership and modes,
append-only claim directory, absence of all claims and final output
directories, and successful read-only authorization verification for all four
arms using the staged digest path explicitly.

It does not import or call `claim_smoke_invocation`,
`reserve_smoke_output_directory`, `verify_model_snapshot`,
`AutoModelForCausalLM`, `torch`, or CUDA.

If pre-activation preflight fails, the active trust anchor has never existed.
Stop without renaming the staged digest and do not run any smoke arm.

### 6. Activate only after preflight PASS

Only after the exact pre-activation auditor invocation returns
`SMOKE_AUTHORIZATION_PREACTIVATION_PREFLIGHT_PASS` may root atomically rename:

```text
/etc/edubench/exp54_smoke_authorization.sha256.staged
→
/etc/edubench/exp54_smoke_authorization.sha256
```

This rename is the single authorization activation point. If the atomic rename
fails, stop and do not run any smoke arm. No production runner may start before
the rename succeeds.

## Authorized smoke execution order after installation

This section is not active until the next review explicitly authorizes smoke
execution.

Run arms serially in the frozen order:

```text
S0
R1
R2
R3
```

Before each arm, verify that its claim and final output directory are absent.
After the runner returns, require:

- exactly one new `<arm>.claimed` file;
- exactly one fixed arm output directory;
- one optimizer step;
- eight events and four micro-batches;
- finite losses and positive finite pre-clipping gradient norm;
- adapter content audit pass;
- no dev/test access.

Do not delete or reset a claim. If an arm fails after claiming, stop the whole
campaign, remove the trust anchor to prevent remaining arms from starting, and
preserve the claim and output evidence. A retry requires a new campaign ID,
new authorization, and another review.

## Deactivation and rollback boundary

Before activation, a failed preflight needs no deactivation because the active
trust anchor has never existed. Root may remove the staged digest,
authorization and empty prepared directories after disabling the append-only
flag.

After any claim exists:

- remove only the trust anchor to prevent additional starts;
- never remove or alter a claim;
- never reuse an output directory;
- preserve authorization, logs, result, and adapter artifacts for audit;
- do not continue to another arm without a new explicit review decision.

Smoke completion does not authorize seed-42 formal training. Formal training
still requires a separate result audit and formal authorization.
