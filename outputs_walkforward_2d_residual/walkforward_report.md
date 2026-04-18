# Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=60
- models: cnn_2d_residual_images
- total OOS predictions: 20160

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0458 | 0.0298 | 0.0072 | 0.0072 | 0.0154 | 0.0956 | 0.5486 | -0.0015 | 0.5559 | 144.0000 | cnn_2d_residual_images | 48 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
