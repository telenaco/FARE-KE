# 6-Axis Load Cell Calibration Results

## K Matrix (Sensitivity Matrix)
This matrix represents the sensitivity of the load cell to the applied loads.

| Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|
| 0.3927 | 0.0061 | -0.0105 | 0.0021 | 0.0240 | -0.0052 |
| -0.0124 | 0.4095 | 0.0068 | 0.0057 | -0.0088 | 0.0065 |
| 0.0084 | -0.0166 | 0.4210 | 0.0092 | -0.0027 | 0.0072 |
| 0.0026 | 0.0029 | 0.0047 | 0.1796 | 0.0038 | -0.0049 |
| 0.0008 | 0.0024 | -0.0081 | 0.0050 | 0.2004 | 0.0001 |
| -0.0068 | 0.0007 | 0.0050 | 0.0031 | -0.0054 | 0.1810 |

## Inverse of K Matrix (K^-1)
The inverse matrix is used for compensating the cross-talk between different channels.

| Fx | Fy | Fz | Mx | My | Mz |
|---|---|---|---|---|---|
| 2.5465 | -0.0338 | 0.0575 | -0.0244 | -0.3035 | 0.0718 |
| 0.0763 | 2.4399 | -0.0336 | -0.0783 | 0.0965 | -0.0857 |
| -0.0489 | 0.0976 | 2.3759 | -0.1239 | 0.0420 | -0.1025 |
| -0.0333 | -0.0409 | -0.0665 | 5.5723 | -0.1013 | 0.1528 |
| -0.0128 | -0.0242 | 0.0979 | -0.1432 | 4.9941 | -0.0089 |
| 0.0970 | -0.0140 | -0.0590 | -0.0969 | 0.1386 | 5.5279 |

## Verification (K * K^-1 = Identity Matrix)
Multiplying the K matrix by its inverse should result in an identity matrix, verifying the calculations.

| I1 | I2 | I3 | I4 | I5 | I6 |
|---|---|---|---|---|---|
| 1.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0000 | 0.0000 |
| -0.0000 | 1.0000 | 0.0000 | 0.0000 | -0.0000 | -0.0000 |
| -0.0000 | -0.0000 | 1.0000 | -0.0000 | 0.0000 | -0.0000 |
| -0.0000 | 0.0000 | -0.0000 | 1.0000 | -0.0000 | -0.0000 |
| 0.0000 | 0.0000 | -0.0000 | -0.0000 | 1.0000 | -0.0000 |
| 0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Compensated Measurements
Applying the inverse matrix to the raw measurements to obtain compensated load values.

| Axis | Raw Measurement (mV/V) | Compensated Load |
|------|------------------------|------------------|
| Fx | -1.6510 | -4.49 |
| Fy | 0.6151 | 1.37 |
| Fz | 0.2501 | 0.65 |
| Mx | 1.0054 | 5.53 |
| My | 0.8402 | 4.08 |
| Mz | 0.0067 | -0.13 |

## Accuracy Metrics
Mean Absolute Error: 0.07
Max Absolute Error: 0.18
