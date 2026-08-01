# Exp54 RAR-SFT dev results

Checkpoint selection: maximum dev Exact, then lower MAE, then earlier epoch.

| Arm | Exact | MAE | Kendall | L2H | Recall-2 | Recall-5 | Forced close |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.6672±0.0431 | 0.3876±0.0490 | 0.5229±0.0774 | 14.67 | 0.0000 | 0.7836 | 0.35% |
| R1 | 0.7063±0.0092 | 0.3384±0.0100 | 0.5875±0.0171 | 10.33 | 0.0238 | 0.8493 | 17.57% |
| R2 | 0.6983±0.0043 | 0.3539±0.0105 | 0.5710±0.0125 | 13.00 | 0.0000 | 0.8502 | 25.45% |
| R3 | 0.7038±0.0043 | 0.3409±0.0076 | 0.5911±0.0064 | 10.33 | 0.0238 | 0.8425 | 16.37% |

The report includes deterministic rationale diagnostics. Model-based blind rationale preference is not run because the two evaluator model identities and credentials are not configured.

Forced close only truncates the tail of a rationale that reaches the fixed 256-token boundary; the score has already been emitted and is unchanged. Unequal arm-specific forced-close rates limit direct rationale-quality interpretation.

Dev accessed: yes. Test accessed: no.
