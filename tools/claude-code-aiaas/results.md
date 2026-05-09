# Level 1 Probe Results

## Refusal (Gemma-4-31B + Qwen-3.6-27B)

| Task | Arch | Regime | AUC (mean ± std) |
|------|------|--------|------------------|
| refusal_gemma4_31b | attention | batch | 0.914 ± 0.001 |
| refusal_gemma4_31b | attention | incremental | 0.886 ± 0.004 |
| refusal_gemma4_31b | attention_4h | batch | 0.912 ± 0.003 |
| refusal_gemma4_31b | attention_4h | incremental | 0.880 ± 0.008 |
| refusal_gemma4_31b | linear | batch | 0.910 ± 0.004 |
| refusal_gemma4_31b | linear | incremental | 0.882 ± 0.005 |
| refusal_gemma4_31b | mlp | batch | 0.909 ± 0.005 |
| refusal_gemma4_31b | mlp | incremental | 0.882 ± 0.004 |
| refusal_qwen36 | attention | batch | 0.833 ± 0.002 |
| refusal_qwen36 | attention | incremental | 0.800 ± 0.008 |
| refusal_qwen36 | attention_4h | batch | 0.833 ± 0.003 |
| refusal_qwen36 | attention_4h | incremental | 0.817 ± 0.011 |
| refusal_qwen36 | linear | batch | 0.839 ± 0.008 |
| refusal_qwen36 | linear | incremental | 0.801 ± 0.023 |
| refusal_qwen36 | mlp | batch | 0.841 ± 0.009 |
| refusal_qwen36 | mlp | incremental | 0.806 ± 0.005 |