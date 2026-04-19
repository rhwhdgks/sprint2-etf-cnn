# ODE-Ready Input Bundles — `ode_inputs_cnn/`

다음 sprint의 **ODE 기반 동적 포트폴리오 최적화** 프로젝트가 바로 consume 할 수 있도록
가공된 입력 데이터입니다. 본 디렉토리만으로 ODE solver 가 필요로 하는 μ(t), Σ(t),
risk(t) 및 원본 일별 수익률을 모두 얻을 수 있습니다.

## 1. 디렉토리 구조

```
ode_inputs_cnn/
├── README.md                 (이 파일)
├── comparison.csv / .md      CNN 모델별 OOS 품질 랭킹
├── qa_report.md / .json      μ/Σ/risk sanity 점검
├── returns_daily.csv         일별 로그수익률 (wide, N dates × 7 assets)
├── prices_daily.csv          일별 close price (wide, 동일 그리드)
├── ensemble_best/          ★ 권장 default — mixed family 앙상블 (logistic + 1D CNN + 2D CNN, cross-sectional rank 평균)
├── ensemble_top3/          레거시 CNN-only 앙상블 (raw-mean)
│   ├── mu_daily.csv
│   ├── risk_daily.csv
│   ├── ode_bundle.csv
│   └── ode_config.json
└── cnn_*/                    개별 CNN 모델 번들 (7개, 같은 스키마)
    ├── mu_daily.csv
    ├── risk_daily.csv
    ├── ode_bundle.csv
    └── ode_config.json
```

## 2. 기본 데이터 스펙

- **자산 (7)**: alternative, corp_bond_ig, developed_equity, emerging_equity, korea_equity, short_treasury, treasury_7_10y
- **공통 유효 날짜**: 2012-01-13 ~ 2026-04-10
- **거래일 기준 grid**: business days (주말·공휴일 제외, forward-fill 안 함)
- **horizon**: 20 영업일 (μ 예측 지평)
- **Σ rolling window**: 60 영업일

## 3. 파일별 스키마 & 단위

### `returns_daily.csv`
날짜 × 자산 일별 **로그 수익률**. 논문 Section 6가 요구하는 primary input.
- 컬럼: `date, alternative, corp_bond_ig, ..., treasury_7_10y`
- 단위: daily log return (절대 스케일), forward-fill 없음
- 용도: ODE solver 가 원한다면 자체 rolling-mean μ 를 재계산하거나, backtest 시 실제 수익률 참조

### `prices_daily.csv`
동일 그리드의 **close price**. 포트폴리오 가치 추적 / 거래비용 시뮬레이션에 사용.

### `{model}/mu_daily.csv`
| column | 설명 |
|--------|------|
| `date` | 영업일 |
| `asset` | 자산명 |
| `mu_hat_daily` | **Calibrated μ(t) — 일별 수익률 스케일 (논문이 요구)** |
| `mu_hat_horizon` | horizon-수익률 스케일 (mu_hat_daily × horizon) |
| `mu_raw_score` | **Raw CNN score (calibration 이전)** — 다른 calibration 쓰고 싶으면 여기서 출발 |
| `future_return` | 실제 실현된 horizon 수익률 (OOS 검증용, solver 입력 아님) |

Calibration 파이프라인 (`make_ode_inputs.calibrate_mu_expanding`):
1. 각 날짜 t에 교차단면 z-score: `z(t,i) = (s - mean) / std`
2. Expanding window 의 실제 horizon-수익률 표준편차 × **0.4 shrinkage**
   (alpha ≈ 40% of total return vol 가정)
3. horizon 로 나눠 daily scale 로 변환
4. **Expanding-only, leakage 없음**

### `{model}/risk_daily.csv`
| column | 설명 |
|--------|------|
| `date` | 영업일 |
| `asset` | 자산명 |
| `risk_score` | 교차단면 z-score (클수록 downside risk 높음) |
| `target` | 실제 downside 값 (OOS 검증용) |

ODE 에서의 쓰임:
- γ(t) 를 자산별로 조절: `γ_i(t) = γ_0 × exp(k × risk_score)`
- 또는 μ 에서 차감: `μ_adj(t,i) = μ(t,i) − δ × risk_score(t,i)`

### `{model}/ode_bundle.csv` (권장 consumer 진입점)
Wide 포맷. 한 행 = 한 날짜. 모든 신호가 같은 날짜 축에 정렬돼 있음.

| prefix | 의미 |
|--------|------|
| `{asset}_mu` | calibrated μ (daily scale) |
| `{asset}_mu_raw` | raw CNN score |
| `{asset}_sigma_ii` | diagonal variance (daily) |
| `{asset}_risk` | risk_score |
| `{a}_{b}_cov` | off-diagonal covariance (daily, symmetric) |

### `{model}/ode_config.json`
모델명, horizon, sigma_window, 자산 리스트, 날짜 범위, OOS 품질, calibration 파라미터.
`mu_calibration` 필드는 다음 sprint가 재현·수정할 때 참고.

## 4. 권장 consumer 코드

```python
import numpy as np
import pandas as pd
from pathlib import Path

BUNDLE_DIR = Path("ode_inputs_cnn/ensemble_best")  # ★ 권장 default (rank 0.061 / Sharpe 0.643)
# 또는 Path("ode_inputs_cnn/cnn_2d_residual_small")  # 단일 CNN 중 최고 (rank 0.043)

bundle = pd.read_csv(BUNDLE_DIR / "ode_bundle.csv", parse_dates=["date"])
assets = ['alternative', 'corp_bond_ig', 'developed_equity', 'emerging_equity', 'korea_equity', 'short_treasury', 'treasury_7_10y']

mu_mat     = bundle[[f"{a}_mu" for a in assets]].to_numpy()       # (T, 7)
sigma_diag = bundle[[f"{a}_sigma_ii" for a in assets]].to_numpy() # (T, 7)

# 7x7 covariance matrix per date
def sigma_matrix(row, assets):
    n = len(assets)
    M = np.zeros((n, n))
    for i, a in enumerate(assets):
        M[i, i] = row[f"{a}_sigma_ii"]
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if j <= i:
                continue
            col = f"{a1}_{a2}_cov" if f"{a1}_{a2}_cov" in row else f"{a2}_{a1}_cov"
            M[i, j] = M[j, i] = row[col]
    return M

# 원본 일별 수익률 (backtest 용)
returns = pd.read_csv("ode_inputs_cnn/returns_daily.csv", parse_dates=["date"])
```

## 5. 모델 선택 가이드

[comparison.md](comparison.md) 참조.

- **★ 추천 기본값**: `ensemble_best` — OOS rank corr 0.061 / Sharpe 0.643 (두 metric 동시 1위)
  - 구성: `logistic_image_scale` + `cnn_1d_cumulative_scale` + `cnn_2d_residual_small` (cross-sectional percentile rank 평균)
- **단일 CNN 중 최고**: `cnn_2d_residual_small` — rank 0.043. Ensemble 없이 단일 CNN으로 가야 할 때
- **대안 CNN-only 앙상블**: `ensemble_top3` (raw-mean) — logistic 빼고 CNN만 쓰고 싶을 때
- **실험 용도**: 그 외 CNN 번들들은 sensitivity/ablation 시 비교용

## 6. 보증

- **No lookahead**:
  - μ calibration 은 walk-forward expanding window 만 사용
  - Σ 는 rolling window (과거 60일만 사용)
  - risk_score 는 날짜별 교차단면 z-score (시간축 정보 안 씀)
- **공통 샘플**: 7 자산 전부 유효한 날짜만 사용 (NaN 없음)
- **수익률 스케일 일관**: μ, Σ 모두 daily log-return 스케일 (논문 Section 6 규약과 일치)

## 7. 다음 sprint 가 해야 할 것

1. 이 번들을 읽어 (μ, Σ, risk) 시계열 추출
2. γ(t) 스케줄 선택 (paper: `γ(t) = γ_0 × exp(−λt)` 예시, γ_0∈{1.5, 3.5})
3. Simplex projection 포함 Euler / RK4 ODE solver 돌리기
4. `returns_daily.csv` 로 realized PnL / turnover / Sharpe 평가
5. 다른 모델 번들로 바꿔가며 sensitivity 확인
