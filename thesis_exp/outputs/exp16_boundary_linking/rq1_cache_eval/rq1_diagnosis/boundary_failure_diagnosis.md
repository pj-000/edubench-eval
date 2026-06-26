# Exp16A-Diag Boundary Failure Diagnosis

This diagnostic uses saved prediction files only. It does not load checkpoints or train models.

## Analyzed Runs

- `metric_rubric` `dev`: `1107` rows
- `qmr` `dev`: `1107` rows

## Per-Label Margin Summary

| variant | split | gold | n | mean s | mean tau3 | mean tau4 | mean margin tau3 | pred>=4 | recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `metric_rubric` | dev | 1 | 19 | -5.2296 | -0.7296 | 1.3072 | -4.5000 | 5/19 (0.2632) | 0.6842 |
| `metric_rubric` | dev | 2 | 38 | 1.8338 | 0.0859 | 1.9778 | 1.7479 | 28/38 (0.7368) | 0.0000 |
| `metric_rubric` | dev | 3 | 105 | 1.3661 | -0.2776 | 1.9373 | 1.6438 | 95/105 (0.9048) | 0.0952 |
| `metric_rubric` | dev | 4 | 389 | 1.4516 | -0.5667 | 1.7320 | 2.0183 | 381/389 (0.9794) | 0.6015 |
| `metric_rubric` | dev | 5 | 556 | 2.6711 | -0.9169 | 1.0128 | 3.5881 | 556/556 (1.0000) | 0.8381 |
| `qmr` | dev | 1 | 19 | -1.1340 | 0.4946 | 2.2912 | -1.6286 | 2/19 (0.1053) | 0.4737 |
| `qmr` | dev | 2 | 38 | 1.6080 | 0.0248 | 2.0244 | 1.5832 | 27/38 (0.7105) | 0.0000 |
| `qmr` | dev | 3 | 105 | 2.0532 | 0.3806 | 3.1363 | 1.6725 | 86/105 (0.8190) | 0.1810 |
| `qmr` | dev | 4 | 389 | 2.1364 | -0.1265 | 2.8297 | 2.2629 | 373/389 (0.9589) | 0.6838 |
| `qmr` | dev | 5 | 556 | 3.0024 | -0.9656 | 1.3435 | 3.9679 | 555/556 (0.9982) | 0.8165 |

## Label-2 Failure Pattern

- `metric_rubric` `dev` label2 n=`38`, pred>=4=`28` (0.7368)；pred=1: 0, pred=2: 0, pred=3: 10, pred=4: 12, pred=5: 16.
- `qmr` `dev` label2 n=`38`, pred>=4=`27` (0.7105)；pred=1: 0, pred=2: 0, pred=3: 11, pred=4: 10, pred=5: 17.

## Low-To-High Diagnosis

- `metric_rubric` `dev`: low-to-high `33`. mean s `2.1385`, mean tau3 `-0.2939`, mean tau4 `1.6210`, mean margin_tau3 `2.4324`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '6.3155', 'tau3_gap_l2h_minus_other_low': '-0.2565', 'margin_gap': '6.5720'}.
- `qmr` `dev`: low-to-high `29`. mean s `1.9953`, mean tau3 `-0.6749`, mean tau4 `1.2160`, mean margin_tau3 `2.6702`. Diagnosis: `both` {'s_gap_l2h_minus_other_low': '2.6491', 'tau3_gap_l2h_minus_other_low': '-1.7433', 'margin_gap': '4.3924'}.

## Metric Concentration

| variant | split | metric | true low | low-to-high | label2 pred>=4 | mean low margin tau3 |
|---|---|---|---:|---:|---:|---:|
| `metric_rubric` | dev | Instruction Following & Task Completion | 7 | 7 (1.0000) | 7/7 (1.0000) | 3.3537 |
| `metric_rubric` | dev | Error Identification & Correction Precision | 5 | 5 (1.0000) | 4/4 (1.0000) | 3.2230 |
| `metric_rubric` | dev | Clarity, Simplicity & Inspiration | 4 | 4 (1.0000) | 4/4 (1.0000) | 2.3229 |
| `metric_rubric` | dev | Scenario Element Integration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.2813 |
| `metric_rubric` | dev | Basic Factual Accuracy | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.7595 |
| `metric_rubric` | dev | Content Relevance & Scope Control | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.4111 |
| `metric_rubric` | dev | Personalization, Adaptation & Learning Support | 1 | 1 (1.0000) | 1/1 (1.0000) | 0.1071 |
| `metric_rubric` | dev | Motivation, Guidance & Positive Feedback | 9 | 5 (0.5556) | 1/4 (0.2500) | 0.0484 |
| `metric_rubric` | dev | Reasoning Process Rigor | 16 | 3 (0.1875) | 3/3 (1.0000) | -5.3035 |
| `metric_rubric` | dev | Higher-Order Thinking & Skill Development | 7 | 0 (0.0000) | 0/7 (0.0000) | -1.0167 |
| `qmr` | dev | Instruction Following & Task Completion | 7 | 7 (1.0000) | 7/7 (1.0000) | 3.0981 |
| `qmr` | dev | Error Identification & Correction Precision | 5 | 5 (1.0000) | 4/4 (1.0000) | 2.8868 |
| `qmr` | dev | Clarity, Simplicity & Inspiration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.8805 |
| `qmr` | dev | Scenario Element Integration | 4 | 4 (1.0000) | 4/4 (1.0000) | 1.3833 |
| `qmr` | dev | Basic Factual Accuracy | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.8546 |
| `qmr` | dev | Content Relevance & Scope Control | 2 | 2 (1.0000) | 2/2 (1.0000) | 4.9667 |
| `qmr` | dev | Motivation, Guidance & Positive Feedback | 9 | 2 (0.2222) | 1/4 (0.2500) | -0.5352 |
| `qmr` | dev | Reasoning Process Rigor | 16 | 3 (0.1875) | 3/3 (1.0000) | -1.5266 |
| `qmr` | dev | Higher-Order Thinking & Skill Development | 7 | 0 (0.0000) | 0/7 (0.0000) | -1.4565 |
| `qmr` | dev | Personalization, Adaptation & Learning Support | 1 | 0 (0.0000) | 0/1 (0.0000) | -0.1639 |

## Boundary-Key Stability

Interpretation rule:

- `< 1e-4`: treat as numerical noise.
- `> 1e-3`: warning; batch/order/numerical effects are no longer negligible.
- `> 1e-2`: serious warning; inspect cache/export implementation and boundary reuse.

All observed boundary_key groups have near-zero tau variation within tolerance.
All observed groups are below the `1e-3` warning threshold.

### Boundary-Key Stability Quantile Summary

| variant | split | groups | unstable | warning | serious | median max diff | p95 max diff | max diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `metric_rubric` | dev | 24 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| `qmr` | dev | 224 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |

### Top-20 Most Unstable Boundary-Key Groups

| variant | split | n | max diff | tau | mean diff | p95 diff | boundary_key | examples |
|---|---|---:|---:|---|---:|---:|---|---|
| `metric_rubric` | dev | 65 | 0.000000 | tau1 | 0.000000 | 0.000000 | `134fe8be716f` | 0bf50da591b0093b48a6b1069868c7a60c1dce30, 12bcd4b81301c8060ad287927889a9a1b7c967ea, 153a4b8462651307f2ce3e7e1c711c6b567062cb |
| `metric_rubric` | dev | 55 | 0.000000 | tau1 | 0.000000 | 0.000000 | `1dcaf86669d4` | 0c474f400c448567de8227ed48f24250fd39fc03, 11016d742fce73e51065518f84a28f9aa10fcebb, 111169cd461add4443b69738a5952b0cb49aabab |
| `metric_rubric` | dev | 40 | 0.000000 | tau1 | 0.000000 | 0.000000 | `366f771fb113` | 0ce247e0de711805c743e7e99382dc8d2fb4593a, 0e02bd863554bbd67bdb44ef1f4ca3d8f75efbf9, 18c5df34b40da1e1f9cb84a5efd33a149ad0da21 |
| `metric_rubric` | dev | 30 | 0.000000 | tau1 | 0.000000 | 0.000000 | `4245ce2c3bb4` | 1aeef6741398006350b8b3301e05b3e3a4ff2b57, 2058d90d7868632a64ce73c7ce394556d4bfd060, 2213d699b1f910533c7acc3fa94e0a39be51694e |
| `metric_rubric` | dev | 50 | 0.000000 | tau1 | 0.000000 | 0.000000 | `48cf9aad290e` | 03d911f33db9c7b54892f2b1a8ca1197e41354c1, 0f4aca6d0c04bab4aae3e576f33ab84d3675934d, 14bcc38ca9ba3a632bf357542df42ec1307aba70 |
| `metric_rubric` | dev | 40 | 0.000000 | tau1 | 0.000000 | 0.000000 | `4aef082a1cbd` | 050f125c5e01db36bfdf94540457c87a0724bdff, 0aaf39ad6c02e6ae4963e2255bdf55dc096aecb8, 0e85df678d2092614f074c1c9abafdbde1b0b352 |
| `metric_rubric` | dev | 16 | 0.000000 | tau1 | 0.000000 | 0.000000 | `6049ce916c27` | 08878a554232c03876dcd910306e1aead6bc04c4, 17b2ac651081219e46d26ed2caee675d63e8e54b, 1a67eecdaf74f2088dc0d0122ffbd507d87e6ec4 |
| `metric_rubric` | dev | 115 | 0.000000 | tau1 | 0.000000 | 0.000000 | `6eabcf80dc70` | 02257ff2c77501987e43989cc8c71a017fb39dad, 0525eeb3debdf218d99f9fb489738873d8788cea, 079ea342cccf68aed2ef06f1585d82aab1079558 |
| `metric_rubric` | dev | 25 | 0.000000 | tau1 | 0.000000 | 0.000000 | `774c9426767f` | 13664f7c3034b3fe0a9e64cc4346abecb67b9c4a, 1b8a3ddece0727927a9d1d06c3ca7ebc8ab2b4d7, 1bfed681c8635373df721e395168f6b0e8c295a5 |
| `metric_rubric` | dev | 55 | 0.000000 | tau1 | 0.000000 | 0.000000 | `83f6981cfd3f` | 0be55a7eed07d9967a869a1769c844eb9a7f792e, 0cb99948c8ab495d1a97aaaa416412554b561a09, 0d1b1588f0fcf0b1b3bd1c8f17d0412a21dd7475 |
| `metric_rubric` | dev | 85 | 0.000000 | tau1 | 0.000000 | 0.000000 | `84c8dec70e40` | 03eaab12a03a161693cdae93f04e88f87c670bb7, 0468aaf6345aa42dad1d7fa61a664f6a0bbb5439, 04bfbf54a7eeef628f3799e10e300520dcee17ca |
| `metric_rubric` | dev | 44 | 0.000000 | tau1 | 0.000000 | 0.000000 | `8933d2a9b092` | 057fb10dadb0ef587ef63e8c0692fb069e747f33, 0b852b93df7d7ad0db7ec99d276d3e4c2e6d53c9, 13b1d7ece3374577cc37e22e71c8766b4225fe21 |
| `metric_rubric` | dev | 30 | 0.000000 | tau1 | 0.000000 | 0.000000 | `8db43356452b` | 0617e15353714d8e70bb076a5058c1e095a1e90f, 142769c8ad0d050d2bc1e4f722188931823e3bb2, 20dd2fbee2c251e7323d8ed8db1fd35af6b3fd7a |
| `metric_rubric` | dev | 60 | 0.000000 | tau1 | 0.000000 | 0.000000 | `91d6d96dc841` | 077f58776d4ab6a467cb6de9b411497d9d50fd50, 0795911801d8484e3269e88e83494fe544417969, 0acfae215fbf65b18d9efae29cc65c39314c9fa7 |
| `metric_rubric` | dev | 35 | 0.000000 | tau1 | 0.000000 | 0.000000 | `a085d2986a97` | 0a2b551b462f6f24d7bd7929223e546e494d419c, 0d46d17e0317c366484af3095b20c524aaee3ca2, 1a7cb84288f2aefb7dbd2d2cfe3754d6492dfbb2 |
| `metric_rubric` | dev | 75 | 0.000000 | tau1 | 0.000000 | 0.000000 | `a5cb4375a835` | 01d0b075c22a91eb1bf30d05c97ab4256ed7d960, 053de98886f60ab5ec51e900d2ec93bc42fee4d5, 09691f5315938a4adbf7895ed89748b85f462522 |
| `metric_rubric` | dev | 35 | 0.000000 | tau1 | 0.000000 | 0.000000 | `a61094f69534` | 0254a2bc0811e4f3ec351229fb7036087d9966e7, 02789727b2e117e08a4e240edba2522b63e4488e, 09f0a5e6798734f1859446596f6223faaa5d8495 |
| `metric_rubric` | dev | 38 | 0.000000 | tau1 | 0.000000 | 0.000000 | `a648a0b8b145` | 0618ee749a296e32260659ce9bd1d7d7fb33217e, 093859561cd29aefd7cfa32e2c78b6c00a38b2ff, 161556ab705771c656fc0055400df7ca28d80565 |
| `metric_rubric` | dev | 1 | 0.000000 | tau1 | 0.000000 | 0.000000 | `aa5c3bacfe2c` | 4275823dfe4094d4b8e9887797ef050094c4f2c1 |
| `metric_rubric` | dev | 65 | 0.000000 | tau1 | 0.000000 | 0.000000 | `bd23f2a81ecb` | 0361cc9784ec44f7e093970b9f81493a6df9adbc, 0b77c5abc4a8a8768d88f6f2a2e6088eecace9b7, 0e54ffac34fa7f91e6f677878bf2da9c344ba8ed |

## qmr vs metric_rubric

- `qmr` includes question text in the boundary tower; `metric_rubric` only uses metric and rubric.
- Compare dev/test low-to-high and label-2 prediction distributions above to decide whether question-specific boundaries improve stability.

## Concise Conclusion

Exp16A enforces ordered thresholds with zero monotonic violation, but the same boundary_key can
still produce non-identical tau values in saved predictions. Treat this boundary-key warning as part
of the RQ1 diagnosis, not as a solved stability result. Label-2 recall and low-to-high failures
remain the key weakness. The failure diagnosis should be read as evidence for whether the next step
should calibrate tau thresholds or improve the quality score separation before moving to risk-aware
RQ2 methods.
