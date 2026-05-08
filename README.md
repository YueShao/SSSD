# SSSD

Official implementation of the paper:

"Internal Evaluation of Unsupervised Anomaly Detection Algorithms with Explanation"

![Framework of SSSD](assets/Framework.png)

## Overview

SSSD (Similar Scores, Similar Data) is an internal evaluation framework for unsupervised anomaly detection.

It is designed to evaluate whether anomaly scores are consistent with the local structure of the input data. Intuitively, data points with similar characteristics should receive similar anomaly scores. A larger SSSD value indicates a stronger violation of this score--data consistency and suggests a higher probability of unreliable detection for the corresponding data point.



## Installation

```bash
pip install -r requirements.txt
```


## Repository Structure

```text
SSSD-main/
├── README.md
├── requirements.txt
├── SSSD.py
├── notebooks/
│   ├── 01_toy_example.ipynb
│   ├── 02_synthetic_experiments.ipynb
│   └── 03_realworld_experiments.ipynb
├── dataset/
│   ├── artificial/
│   ├── realworld/
└── assets/
    └── Framework.png
```

## Notebooks

### 1. Toy Example

01_toy_example.ipynb

A minimal 2D visualization example demonstrating the intuition behind SSSD and score-data consistency.

### 2. Synthetic Experiments

02_synthetic_experiments.ipynb

Experiments on synthetic datasets for validating the effectiveness of SSSD.

### 3. Real-world Experiments

03_realworld_experiments.ipynb

Experiments on real-world anomaly detection datasets using multiple anomaly detectors.

## Using SSSD in Your Own Code

```python
from SSSD import SSSD

# X: input data, shape (n_samples, n_features)
# scores: anomaly scores, shape (n_samples,)
sssd = SSSD(X, scores, k=5, m=5)

final_sssd, final_sssd_sum, ranked_indices = sssd.get_final_SSSD()
score_sssd, score_sssd_sum, score_ranked_indices = sssd.get_Score_SSSD()
distance_sssd, distance_sssd_sum, distance_ranked_indices = sssd.get_Distance_SSSD()
```

Output interpretation:

- `final_sssd`: point-wise SSSD values in the original sample order.
- `final_sssd_sum`: aggregated SSSD value over all samples.
- `ranked_indices`: sample indices sorted by descending SSSD value. Samples ranked earlier are more likely to violate score--data consistency.


## Notes

* Larger SSSD values indicate stronger violations of score-data consistency.
* The implementation supports arbitrary anomaly detectors that output anomaly scores.