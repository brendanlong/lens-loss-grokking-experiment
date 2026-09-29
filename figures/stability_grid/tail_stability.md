| cell | seeds | tail evals >= 0.95 (mean; per seed) | seeds with a tail collapse | worst tail acc | final acc < 0.95 |
|---|---|---|---|---|---|
| 1 layer, LayerNorm | 3 | 1.00; 1.00 1.00 1.00 | 0 | 0.99 | 0 |
| 2 layers, LayerNorm | 10 | 0.90; 0.90 0.93 1.00 0.87 0.97 0.91 0.98 0.82 0.82 0.75 | 9 | 0.05 | 2 |
| 2 layers, LayerNorm, aux loss | 3 | 1.00; 1.00 1.00 1.00 | 0 | 0.99 | 0 |
| 1 layer, no LayerNorm (Nanda) | 3 | 1.00; 1.00 1.00 1.00 | 0 | 0.98 | 0 |
| 2 layers, no LayerNorm | 3 | 1.00; 1.00 0.99 1.00 | 1 | 0.61 | 0 |
| 2 layers, no LayerNorm, aux loss | 3 | 0.99; 0.99 1.00 0.99 | 1 | 0.77 | 0 |
| 2 layers, LayerNorm, dropout 0.1 | 3 | 0.93; 0.93 0.92 0.95 | 3 | 0.02 | 1 |
| 2 layers, LayerNorm, LR ×0.1 at 20k | 3 | 0.66; 1.00 0.98 0.00 | 1 | 0.48 | 1 |
| 2 layers, LayerNorm, LR annealed 20k→40k | 10 | 1.00; 1.00 1.00 0.98 1.00 1.00 1.00 1.00 1.00 1.00 1.00 | 1 | 0.41 | 0 |
