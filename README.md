# sprint2-etf-cnn

7개 ETF 자산의 μ(기대수익) 시그널을 CNN으로 생성해 ODE 기반 동적 포트폴리오 최적화 스프린트에 넘겨주는 핸드오프 패키지.

## 요약

- 60일 OHLCV를 Jiang-style 이미지로 변환 → CNN 7종으로 walk-forward OOS 예측
- 평가: 24 folds × 7년 OOS × 2,880일
- 결과물: `ode_inputs_cnn/` 아래 모델별 μ·Σ·risk·R 번들 + 앙상블 + QA + 베이스라인 비교
- 자세한 해설은 [blog_cnn_lstm_handoff.md](blog_cnn_lstm_handoff.md)

## 디렉토리

```
sprint2-etf-cnn/
├── src/                             모델·데이터·평가 코드
├── etfdata.csv                      원본 ETF 일별 OHLCV
├── requirements.txt
│
├── run_walkforward.py               walk-forward OOS 실행 entrypoint
├── make_ode_inputs.py               μ calibration, rolling Σ, ODE 번들 생성
├── collect_cnn_ode_signals.py       CNN 예측 → 모델별 번들 디렉토리 생성
├── build_handoff_package.py         README/QA/ensemble 생성 (1-shot)
├── build_handoff_figures.py         발표용 figure 8종 생성
├── build_baseline_comparison.py     CNN vs logistic 베이스라인 비교 + ablation
│
├── outputs_walkforward_4model/      walk-forward 원본 예측 (CNN 4 + 2 logistic)
├── outputs_walkforward_mu/          1D CNN μ 실험
├── outputs_walkforward_1dcnn_extra/ attention/dilated/multiscale/cumulative
├── outputs_walkforward_2d_residual/ 2D CNN + ResNet
├── outputs_walkforward_risk/        risk signal용 walk-forward
│
├── ode_inputs_cnn/                  ★ 최종 핸드오프 패키지
│   ├── README.md
│   ├── HANDOFF_SUMMARY.md           팀 1페이지 브리핑
│   ├── qa_report.md / qa_report.json
│   ├── comparison.csv / comparison.md
│   ├── comparison_with_baselines.csv
│   ├── returns_daily.csv / prices_daily.csv
│   ├── figures/                     발표용 PNG 9종
│   ├── ensemble_top3/               top-3 CNN 앙상블 번들
│   └── {model_name}/                모델별 mu_daily / ode_bundle / ode_config
│
└── blog_cnn_lstm_handoff.md         CNN → LSTM 로드맵 블로그 글
```

## 빠른 재현

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. walk-forward 실행 (결과 → outputs_walkforward_*/)
python run_walkforward.py --lookback 60 --horizon 20

# 2. CNN 예측 → ODE 입력 번들
python collect_cnn_ode_signals.py \
  --pred-paths outputs_walkforward_4model/walkforward_predictions.csv \
               outputs_walkforward_1dcnn_extra/walkforward_predictions.csv \
               outputs_walkforward_2d_residual/walkforward_predictions.csv \
  --risk-path outputs_walkforward_risk/walkforward_predictions.csv \
  --output-dir ode_inputs_cnn

# 3. 핸드오프 패키지 finalize (README/QA/ensemble)
python build_handoff_package.py

# 4. 발표용 figure
python build_handoff_figures.py

# 5. 베이스라인 비교 + ablation
python build_baseline_comparison.py
```

## 다음 스프린트가 쓰는 방법

```python
import pandas as pd
from pathlib import Path

ROOT = Path("ode_inputs_cnn/ensemble_best")  # 권장; ensemble_top3는 레거시 CNN-only

bundle = pd.read_csv(ROOT / "ode_bundle.csv", parse_dates=["date"])
returns = pd.read_csv("ode_inputs_cnn/returns_daily.csv", parse_dates=["date"])

ASSETS = ["alternative", "corp_bond_ig", "developed_equity", "emerging_equity",
          "korea_equity", "short_treasury", "treasury_7_10y"]

mu = bundle[[f"{a}_mu" for a in ASSETS]].values        # (T, 7)
risk = bundle[[f"{a}_risk" for a in ASSETS]].values    # (T, 7)
```

Leakage 보증: μ calibration은 expanding 방식, Σ는 t 시점까지의 60일 window만 사용.

## 핵심 결과

| 지표 | 1위 모델 | 값 |
|---|---|---|
| OOS rank correlation | ★ `ensemble_best` (mixed) | **0.0422** |
| Top-k portfolio Sharpe | `cnn_1d_cumulative_scale` | 0.521 |
| ★ Balanced | `ensemble_best` = logistic_image + attention + cumulative | rank 0.042 / Sharpe 0.503 |
| 레거시 CNN-only 앙상블 | `ensemble_top3` | rank 0.032 / Sharpe 0.374 |

→ **CNN-only 앙상블만으로는 logistic_image 단일(0.039)을 rank corr에서 못 이김.** Logistic을 포함한 mixed 앙상블 `ensemble_best`로 천장을 뚫고 두 metric 모두 상위권 달성. 다음 단계는 시퀀스 모델(LSTM)로 정보 축 다양화 — [blog 글](blog_cnn_lstm_handoff.md) 참고.

## 참고

- 논문: An ODE-Based Dynamic Mean-Variance Portfolio Optimisation with Time-Varying Risk Aversion
- 이미지 변환: Jiang et al. 2016 (60일 OHLCV → 2D 이미지)
