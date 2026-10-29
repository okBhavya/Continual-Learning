| Sequence | Strategy | Avg Final Return (↑) | Forgetting (↓) | Plasticity (↑) | Forward Transfer (↑) | Runs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `similar` | **Naive Fine-Tuning** | 97.4 ± 9.0 | 81.5 ± 97.1 | 151.7 ± 66.6 | 152.8 ± 89.5 | 3 |
| `similar` | **Experience Replay** | 212.9 ± 148.1 | -62.6 ± 237.0 | 171.2 ± 55.0 | 145.2 ± 95.5 | 3 |
| `similar` | **EWC** | 236.5 ± 227.0 | -64.2 ± 116.4 | 193.7 ± 149.9 | 146.3 ± 101.7 | 3 |
| `similar` | **Parseval Regularization** | 133.6 ± 5.7 | 60.4 ± 67.3 | 173.8 ± 47.1 | 168.4 ± 68.0 | 3 |
| `similar` | **CAR-SR (Proposed)** | **202.6 ± 28.6** | -23.7 ± 89.3 | 186.8 ± 60.6 | 153.8 ± 78.3 | 3 |
| `similar` | **CAR-SR (w/o Projection)** | 219.3 ± 45.0 | -30.5 ± 77.6 | 199.0 ± 43.8 | 162.4 ± 60.2 | 3 |
| `similar` | **CAR-SR (w/o Parseval)** | 187.0 ± 72.1 | -25.6 ± 190.6 | 170.0 ± 55.5 | 154.3 ± 91.4 | 3 |
| `partial` | **Naive Fine-Tuning** | 73.9 ± 35.0 | 168.4 ± 104.8 | 186.2 ± 97.8 | 55.6 ± 24.6 | 3 |
| `partial` | **Experience Replay** | 58.1 ± 10.8 | 142.3 ± 129.9 | 153.0 ± 78.6 | 61.5 ± 34.5 | 3 |
| `partial` | **EWC** | 98.0 ± 42.1 | 108.3 ± 64.7 | 170.2 ± 85.2 | 55.5 ± 30.8 | 3 |
| `partial` | **Parseval Regularization** | 41.3 ± 23.6 | 147.5 ± 23.3 | 139.7 ± 28.2 | 94.1 ± 29.7 | 3 |
| `partial` | **CAR-SR (Proposed)** | **46.4 ± 5.4** | 97.4 ± 34.3 | 111.4 ± 26.4 | 82.8 ± 35.9 | 3 |
| `partial` | **CAR-SR (w/o Projection)** | 45.8 ± 8.4 | 159.2 ± 91.1 | 152.0 ± 69.1 | 84.4 ± 29.8 | 3 |
| `partial` | **CAR-SR (w/o Parseval)** | 60.3 ± 13.2 | 122.8 ± 135.5 | 142.1 ± 93.7 | 44.9 ± 29.1 | 3 |
| `dissimilar` | **Naive Fine-Tuning** | 79.9 ± 59.6 | 182.9 ± 100.5 | 201.8 ± 73.0 | 31.5 ± 26.8 | 3 |
| `dissimilar` | **Experience Replay** | 54.1 ± 20.5 | 179.0 ± 200.2 | 173.4 ± 123.6 | 34.4 ± 22.8 | 3 |
| `dissimilar` | **EWC** | 48.2 ± 11.9 | 192.9 ± 97.1 | 176.8 ± 76.1 | 30.9 ± 26.4 | 3 |
| `dissimilar` | **Parseval Regularization** | 89.5 ± 55.9 | 108.1 ± 16.5 | 161.6 ± 56.5 | 40.2 ± 34.2 | 3 |
| `dissimilar` | **CAR-SR (Proposed)** | **25.1 ± 5.3** | 173.4 ± 23.0 | 140.7 ± 10.5 | 40.3 ± 34.2 | 3 |
| `dissimilar` | **CAR-SR (w/o Projection)** | 33.9 ± 22.9 | 164.3 ± 50.6 | 143.5 ± 56.4 | 40.2 ± 34.2 | 3 |
| `dissimilar` | **CAR-SR (w/o Parseval)** | 50.5 ± 15.5 | 220.8 ± 158.6 | 197.7 ± 90.6 | 33.7 ± 23.5 | 3 |