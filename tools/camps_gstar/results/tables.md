# g* runs: tables

One network seed (0) per truth; 1000 ECG iterations; frozen H, speeds boxed at truth +/-1%.
Random-init column: old Method 1, median of 100 networks (`outputs/set1/SUMMARY.md`).

## ECG error (relative)

| truth | g0 = truth|S | g* (LAT fit) | g* + ECG inverse | random-init |
|---|---|---|---|---|
| truth5ti_s0 | 1.86e-02 | 3.21e-03 | 6.60e-04 | 3.10e-02 |
| truth5ti_s1 | 1.57e-02 | 7.60e-03 | 7.59e-04 | 1.70e-02 |
| truth5ti_s2 | 1.21e-02 | 8.61e-03 | 1.08e-03 | 2.20e-02 |
| truth5ti_s3 | 3.57e-02 | 9.58e-03 | 1.36e-03 | 7.30e-02 |
| truth5ti_s4 | 5.01e-03 | 1.71e-03 | 4.54e-04 | 8.00e-03 |
| truth5ti_s5 | 2.00e-02 | 3.04e-03 | 6.34e-04 | 2.20e-02 |
| **median** | **1.72e-02** | **5.40e-03** | **7.10e-04** | **2.20e-02** |

## LAT correlation with truth

| truth | g0 = truth|S | g* (LAT fit) | g* + ECG inverse | random-init |
|---|---|---|---|---|
| truth5ti_s0 | 0.9989 | 0.9997 | 0.9973 | 0.7820 |
| truth5ti_s1 | 0.9989 | 0.9996 | 0.9942 | 0.6810 |
| truth5ti_s2 | 0.9994 | 0.9998 | 0.9969 | 0.8010 |
| truth5ti_s3 | 0.9986 | 0.9995 | 0.9976 | 0.5230 |
| truth5ti_s4 | 0.9996 | 0.9999 | 0.9977 | 0.8950 |
| truth5ti_s5 | 0.9981 | 0.9995 | 0.9985 | 0.5770 |
| **median** | **0.9989** | **0.9997** | **0.9974** | **0.7315** |

## LAT MAE (ms)

| truth | g0 = truth|S | g* (LAT fit) | g* + ECG inverse | random-init |
|---|---|---|---|---|
| truth5ti_s0 | 0.96 | 0.46 | 1.34 | 12.20 |
| truth5ti_s1 | 1.03 | 0.48 | 1.74 | 14.20 |
| truth5ti_s2 | 0.94 | 0.49 | 1.89 | 14.50 |
| truth5ti_s3 | 0.89 | 0.50 | 1.11 | 15.30 |
| truth5ti_s4 | 1.17 | 0.42 | 1.45 | 11.90 |
| truth5ti_s5 | 1.20 | 0.51 | 0.95 | 14.90 |
| **median** | **1.00** | **0.48** | **1.40** | **14.35** |

## Sources and root recovery (after ECG inverse)

| truth | detected sources | candidate rank of each true root (0 = exact) | start-to-g* regression max err (ms) |
|---|---|---|---|
| truth5ti_s0 | 5 | [0, 0, 0, 0, 0] | 12.2 |
| truth5ti_s1 | 6 | [0, 0, 0, 0, 0] | 12.7 |
| truth5ti_s2 | 5 | [0, 1, 0, 0, 0] | 12.0 |
| truth5ti_s3 | 5 | [0, 0, 0, 0, 0] | 8.4 |
| truth5ti_s4 | 4 | [0, 0, 0, 1, 0] | 13.9 |
| truth5ti_s5 | 5 | [0, 0, 0, 0, 0] | 10.3 |

## ECG-error reduction factors

| truth | g0 -> g* | g* -> final | random-init -> final |
|---|---|---|---|
| truth5ti_s0 | 5.8x | 4.9x | 47x |
| truth5ti_s1 | 2.1x | 10.0x | 22x |
| truth5ti_s2 | 1.4x | 7.9x | 20x |
| truth5ti_s3 | 3.7x | 7.0x | 54x |
| truth5ti_s4 | 2.9x | 3.8x | 18x |
| truth5ti_s5 | 6.6x | 4.8x | 35x |
