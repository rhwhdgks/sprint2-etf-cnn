# Walk-Forward OOS Evaluation

## Configuration
- lookback=20, horizon=20
- label_mode=regression, target=downside_like
- wf_min_train_days=500  wf_val_days=60  wf_test_days=120
- models: logistic_cumulative_scale
- total OOS predictions: 20160

## Model Comparison (Aggregated OOS)
| rmse | mae | target_rank_correlation | future_return_rank_correlation | top_k_cumulative_return | top_k_sharpe | top_k_hit_rate | top_bottom_spread_mean | turnover | n_rebalances | model_name | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0002 | 0.0000 | 0.7414 | -0.0701 | 0.0105 | 0.0465 | 0.5417 | -0.0067 | 0.3147 | 144.0000 | logistic_cumulative_scale | 24 |

## ODE Connection
- Use the best model's `signal_value` as mu(t) proxy after calibration.
- `future_return_rank_correlation` is the primary OOS quality indicator.
