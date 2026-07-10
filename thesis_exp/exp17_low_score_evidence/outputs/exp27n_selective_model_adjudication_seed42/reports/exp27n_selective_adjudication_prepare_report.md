# Exp27N Selective Model Adjudication Preparation

Exp27N applies the Exp27M locked qwen-human gap policy to the existing 361
train-only teacher-audited rows. It emits only the 54 high-risk rows that
still lack model review.

- direct accept: 173
- weighted accept: 118
- adjudication required: 70
- already reviewed: 16
- remaining blind packet: 54
- reviewer plan: one GPT-5.6Pro session for all 54
- Qwen/DeepSeek/human scores in packet: no
- API calls: 0
- GPU/training: 0
- dev/test labels used: no

The returned annotations remain model-reviewed silver. Low-confidence or
unclear rows must become review_only. Dataset construction and training stay
blocked until all 54 outputs pass schema and evidence-substring validation.
