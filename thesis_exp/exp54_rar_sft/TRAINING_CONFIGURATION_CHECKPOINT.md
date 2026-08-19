# Exp54 frozen manifests and training-configuration checkpoint

## Authorization boundary

The exact materialized-manifest candidate reviewed at commit
`dad35890a1f0a5111ae0c69a2b60dff26700c24a` received
`MATERIALIZED_MANIFEST_PASS`. The freeze command re-hashes the reviewed
candidate report and lock, all twelve private manifests, the shared prompt
cache, both reference inventories, all schedules, donor maps and masks, the
reviewed source files, tokenizer lock, and upstream locks. It then writes an
aggregate public lock with status:

`MATERIALIZED_MANIFEST_FROZEN_TRAINING_NOT_AUTHORIZED`

No private artifact is rebuilt, copied, or edited during promotion. Any
locked hash change invalidates the freeze.

This checkpoint prepares a complete training-configuration candidate. It
does not authorize a smoke test, formal seed 42, any optimizer step, or
dev/test access.

## Model and runtime

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Immutable upstream revision:
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- Architecture: `Qwen3ForCausalLM`
- Full local snapshot: config, weight index, all three safetensor shards,
  tokenizer, tokenizer config, merges, vocabulary, and generation config are
  size- and SHA-256-locked.
- The complete 16-file regular-file listing is size-locked with aggregate
  SHA-256
  `96308d1ae2e03ab011a52371780c8f4b448930d2e393fb8682fa8aa09687064a`.
  A symlink or non-directory model root hard-fails before traversal. Added
  files and descendant symlinks also hard-fail, including nested
  `adapter_config.json`, `adapter_model.safetensors`, and
  `adapter_model.bin`.
- Every indexed tensor is checked against the header of its exact physical
  shard. Duplicate, unindexed, missing, and swapped-shard tensor names all
  hard-fail.
- Base parameters derived from safetensor headers: `4,022,468,096`
- Training hardware: exactly one visible `NVIDIA RTX A6000` per run
- Runtime: Python 3.10.19, PyTorch 2.4.0+cu121, Transformers 4.57.1,
  PEFT 0.17.1, CUDA 12.1, cuDNN 9.1.0, driver 530.30.02
- Precision: bfloat16 autocast, no TF32, no reduced-precision FP16 reduction
- Determinism: fixed Python/NumPy/PyTorch/CUDA seed, deterministic algorithms,
  deterministic cuDNN, fixed cuBLAS workspace, no data-loader workers

The server audit found four matching RTX A6000 devices. Each individual
training process must expose exactly one of them. Arms may run concurrently
on the four identical A6000s, but every arm/seed is an independent process
and independently reloads the same base snapshot.

## LoRA candidate

The historical Exp53 values are used as a fixed, pre-result candidate rather
than tuned on Exp54:

- rank 16
- alpha 32
- dropout 0.05
- no bias or additional saved modules
- causal-LM task
- target modules:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj`

Every target occurs once in each of 36 transformer layers. Safetensor shape
metadata independently yields exactly `33,030,144` trainable LoRA
parameters. The training entry point hard-fails if PEFT produces a different
trainable count. Before new LoRA injection it also rejects Transformers'
`_hf_peft_config_loaded`, any existing `peft_config`, and any parameter name
containing `lora_` or `adapter`.

## Data traversal and optimizer steps

Each frozen arm/seed manifest already contains three logical epochs:

- 2,654 rows per logical epoch
- 7,962 events per manifest
- event order is logical-epoch-major and original-row-major
- no shuffle, packing, truncation, or second physical manifest pass
- fixed right padding to 2,048 tokens

The physical manifest is therefore traversed exactly once. Configuring three
physical epochs would incorrectly produce nine logical epochs and is
explicitly rejected.

At micro-batch size 2, each logical epoch contains 1,327 micro-batches.
Gradient accumulation is four, with a mandatory flush at each logical-epoch
boundary:

- 331 complete groups of four micro-batches
- one remainder group of three micro-batches
- the remainder loss is divided by the actual group size, three
- 332 optimizer steps per logical epoch
- 996 optimizer steps total

No remainder is carried into the next logical epoch.

## Loss and optimization

The executable path directly calls the audited `blockwise_causal_loss`:

- score block: token mean within each sample, coefficient 1
- active rationale block: token mean within each sample, coefficient 1
- inactive rationale block: coefficient 0
- batch loss: mean of per-sample block sums
- prompt, JSON punctuation, boundary padding, assistant suffix, and fixed
  padding are not supervised

Optimization candidate:

- AdamW
- learning rate `1e-4`
- betas `(0.9, 0.999)`
- epsilon `1e-8`
- weight decay 0
- gradient clipping 1
- cosine schedule
- 50 warmup optimizer steps

No coefficient or hyperparameter search is permitted in the primary
experiment.

## Checkpoints and inference contract

The runner saves after logical epochs 1, 2, and 3:

- LoRA adapter
- optimizer state
- scheduler state
- CPU/CUDA/Python RNG state
- aggregate trainer state with config, manifest, and freeze hashes

Resume and output overwrite are disabled for the first formal campaign.
Evaluation does not run inside the trainer.

Generation is deterministic greedy decoding with one beam and at most 256
new tokens. Thinking mode is disabled by the frozen chat template. The parser
accepts exactly one JSON object with exactly `score` and `rationale`; score
must be a non-boolean integer from 1 to 5 and rationale must be a string.
Duplicate keys are rejected at every JSON object level.

The train-only frozen manifests prove the 256-token cap covers all
materialized assistant targets, including assistant suffix tokens. Maximum
counts are S0 11, R1 157, R2 151, and R3 151; no event exceeds 256.

## Training hard gate

The checked-in configuration says both smoke and formal training are false.
The entry point validates the frozen manifest, but then refuses before
loading model weights unless a later external authorization lock binds:

- the exact frozen-manifest lock;
- the exact audited training-configuration lock;
- the exact configuration;
- a 16-file runtime source closure including the independent authorization
  guard, training entry point, block loss, serialization contract, inference
  parser, manifest I/O, package initializers, prompt cleaning and its local
  dependencies (including `utils/__init__.py`), and shell launcher;
- the explicitly allowed arms and seeds.

Self-consistent repository JSON files are not authorization. The exact
authorization-file SHA-256 must also be installed by an external
administrator at `/etc/edubench/exp54_authorization.sha256`. The runner
requires that trust anchor to be a non-symlink regular file, owned by root,
and not writable by group or other users. Neither the CLI nor repository
configuration can override its path. The formal audit proves that a forged
lock pair with a mismatched external digest is rejected, while an exact
externally bound digest is accepted. Authorization failures occur before
model hash verification, model loading, or output-directory creation. A
future review must separately authorize smoke or formal execution.

At candidate-audit time the fixed trust-anchor path is checked with `lstat`.
The report can be emitted only when the path does not exist. Any regular
file, symlink, directory, or other object already installed at that path
causes the candidate audit to fail rather than claiming a false
not-authorized state.
