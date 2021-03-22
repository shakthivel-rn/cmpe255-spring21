| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.6979166666666666  | [[114  16] [ 42  20]] |  The selected features are ['bmi', 'pedigree', 'age'] as they have higher coefficients |
| Solution 2   | 0.7552083333333334  | [[113  17] [ 30  32]] |  The selected features are ['pregnant', 'glucose', 'skin'] as they have higher coefficients |
| Solution 3   | 0.796875  | [[118  12] [ 27  35]] |  The selected features are ['pregnant', 'glucose', 'skin', 'bmi', 'pedigree'] as they have higher coefficients |
