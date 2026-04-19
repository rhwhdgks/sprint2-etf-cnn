# CNN Model Comparison — ODE Signal Quality

Metric: mean OOS Spearman rank correlation (signal_value vs actual future_return).
Higher = better mu(t) input for ODE.

| rank | model | rank_corr | pct_positive_dates | n_folds |
| --- | --- | --- | --- | --- |
| 1 | `cnn_2d_residual_small` | 0.0426 | 52.08% | 48 |
| 2 | `ensemble_top3` | 0.0375 | 53.19% | 24 |
| 3 | `cnn_1d_dilated_image_scale` | 0.0280 | 51.15% | 24 |
| 4 | `cnn_1d_cumulative_scale` | 0.0271 | 50.56% | 24 |
| 5 | `cnn_1d_attention_image_scale` | 0.0231 | 50.66% | 24 |
| 6 | `cnn_1d_multiscale_image_scale` | 0.0087 | 48.82% | 24 |
| 7 | `cnn_2d_residual_images` | 0.0072 | 49.03% | 48 |
| 8 | `cnn_2d_rendered_images` | 0.0022 | 49.51% | 24 |
| 9 | `cnn_1d_image_scale` | -0.0010 | 49.55% | 24 |

**Recommended for ODE mu(t): `cnn_2d_residual_small`**

## ODE connection
- Each model's `ode_bundle.csv` contains `{asset}_mu`, `{asset}_sigma_ii`,
  `{asset}_risk`, and all off-diagonal covariances aligned on a daily grid.
- Use `{asset}_mu` → mu(t) vector.
- Use `{asset}_sigma_ii` + `{a}_{b}_cov` → Sigma(t) matrix.
- Use `{asset}_risk` → modulate gamma(t) or penalise mu.
