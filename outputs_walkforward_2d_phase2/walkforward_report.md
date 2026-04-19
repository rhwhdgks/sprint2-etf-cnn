# Walk-Forward OOS Evaluation

## Configuration
- lookback=60, horizon=20
- label_mode=regression, target=future_return
- wf_min_train_days=500  wf_val_days=60  wf_test_days=60
- models: cnn_2d_residual_small, cnn_2d_residual_wd
- total OOS predictions: 40320

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0457 | 0.0298 | 0.0057 | 0.0057 | 0.3631 | 0.2486 | 0.5833 | 0.0003 | 0.6189 | 144.0000 | cnn_2d_residual_wd | 48 |
| 0.0462 | 0.0304 | 0.0426 | 0.0426 | 0.2528 | 0.2080 | 0.5903 | 0.0027 | 0.6084 | 144.0000 | cnn_2d_residual_small | 48 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
