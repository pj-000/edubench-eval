# Exp10 Preflight Ablation Matrix

Training executed: no

| ablation | lambdas | force_pair_training | use_pair_training | expected_dataloader_mode | strict_module_ablation | status |
| --- | --- | ---: | ---: | --- | ---: | --- |
| full_qdpr2 | point=1.0, pair=0.05, anchor=0.5, mono=0.1 | False | True | pair | False | PASS |
| no_pair | point=1.0, pair=0.0, anchor=0.5, mono=0.1 | False | False | pointwise | False | PASS |
| no_pair_same_pair_batches | point=1.0, pair=0.0, anchor=0.5, mono=0.1 | True | True | pair | True | PASS |
| no_anchor | point=1.0, pair=0.05, anchor=0.0, mono=0.1 | False | True | pair | True | PASS |
| no_mono | point=1.0, pair=0.05, anchor=0.5, mono=0.0 | False | True | pair | True | PASS |
| point_only | point=1.0, pair=0.0, anchor=0.0, mono=0.0 | False | False | pointwise | False | PASS |
| no_point_diagnostic | point=0.0, pair=0.05, anchor=0.5, mono=0.1 | False | True | pair | False | PASS |
