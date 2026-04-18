# Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=120
- models: logistic_cumulative_scale, logistic_image_scale, cnn_1d_image_scale, cnn_1d_attention_image_scale
- total OOS predictions: 80640

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0458 | 0.0301 | 0.0392 | 0.0392 | 0.6569 | 0.3850 | 0.5903 | 0.0036 | 0.7168 | 144.0000 | logistic_image_scale | 24 |
| 0.0461 | 0.0301 | 0.0231 | 0.0231 | 0.3918 | 0.2681 | 0.5903 | 0.0011 | 0.6014 | 144.0000 | cnn_1d_attention_image_scale | 24 |
| 0.0465 | 0.0299 | -0.0072 | -0.0072 | -0.0512 | 0.0764 | 0.5556 | -0.0007 | 0.6469 | 144.0000 | logistic_cumulative_scale | 24 |
| 0.0458 | 0.0297 | -0.0010 | -0.0010 | -0.0469 | 0.0727 | 0.5486 | -0.0017 | 0.5280 | 144.0000 | cnn_1d_image_scale | 24 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
