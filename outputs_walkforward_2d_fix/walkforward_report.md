# Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=60
- models: cnn_2d_residual_images, cnn_1d_dilated_image_scale
- total OOS predictions: 40320

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0459 | 0.0300 | 0.0099 | 0.0099 | 0.3539 | 0.2478 | 0.5486 | 0.0016 | 0.6119 | 144.0000 | cnn_2d_residual_images | 48 |
| 0.0460 | 0.0301 | 0.0277 | 0.0277 | 0.0219 | 0.1054 | 0.5625 | -0.0011 | 0.6434 | 144.0000 | cnn_1d_dilated_image_scale | 48 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
