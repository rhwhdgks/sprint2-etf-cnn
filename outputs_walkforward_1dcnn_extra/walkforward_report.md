# Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=120
- models: cnn_1d_cumulative_scale, cnn_1d_multiscale_image_scale, cnn_1d_dilated_image_scale, cnn_1d_attention_image_scale
- total OOS predictions: 80640

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0453 | 0.0292 | 0.0271 | 0.0271 | 0.9977 | 0.5214 | 0.5903 | 0.0047 | 0.5315 | 144.0000 | cnn_1d_cumulative_scale | 24 |
| 0.0460 | 0.0301 | 0.0280 | 0.0280 | 0.6661 | 0.3544 | 0.5694 | 0.0047 | 0.6014 | 144.0000 | cnn_1d_dilated_image_scale | 24 |
| 0.0461 | 0.0301 | 0.0231 | 0.0231 | 0.3918 | 0.2681 | 0.5903 | 0.0011 | 0.6014 | 144.0000 | cnn_1d_attention_image_scale | 24 |
| 0.0457 | 0.0296 | 0.0087 | 0.0087 | 0.3424 | 0.2459 | 0.5833 | 0.0024 | 0.4615 | 144.0000 | cnn_1d_multiscale_image_scale | 24 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
