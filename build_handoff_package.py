#!/usr/bin/env python3
"""Finalize the ODE hand-off package under ode_inputs_cnn/.

Prerequisite: collect_cnn_ode_signals.py has already populated
  ode_inputs_cnn/{model}/ode_bundle.csv, mu_daily.csv, risk_daily.csv, ode_config.json
  ode_inputs_cnn/ensemble_top3/  (same structure)
  ode_inputs_cnn/comparison.{csv,md}

This script adds:
  returns_daily.csv, prices_daily.csv   daily log-return and close-price matrices
  qa_report.{md,json}                   μ/Σ/risk sanity statistics
  README.md                             schema + consumer guide
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from src.data.loader import REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample


ASSETS_EXPECTED = [
    "alternative",
    "corp_bond_ig",
    "developed_equity",
    "emerging_equity",
    "korea_equity",
    "short_treasury",
    "treasury_7_10y",
]


# ─── returns / prices matrices ────────────────────────────────────────────────

def build_price_matrices(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices_wide, returns_wide) aligned on common-sample business days."""
    _, long_panel, _, _, _ = load_etf_csv(data_path)
    common_panel, _ = restrict_common_valid_sample(
        long_panel, selected_assets=ASSETS_EXPECTED, required_fields=REQUIRED_PRICE_FIELDS
    )
    if common_panel.empty:
        raise ValueError("no common valid sample")

    close_wide = (
        common_panel.pivot(index="date", columns="asset", values="close")
        .sort_index()[ASSETS_EXPECTED]
    )
    log_ret = np.log(close_wide / close_wide.shift(1)).dropna()

    close_wide = close_wide.loc[log_ret.index]
    close_wide.index.name = "date"
    log_ret.index.name = "date"
    return close_wide.reset_index(), log_ret.reset_index()


# ─── QA ───────────────────────────────────────────────────────────────────────

def mu_stats_per_asset(bundle_dir: Path) -> pd.DataFrame:
    mu = pd.read_csv(bundle_dir / "mu_daily.csv", parse_dates=["date"])
    grouped = mu.groupby("asset")["mu_hat_daily"].agg(["mean", "std", "min", "max"]).round(8)
    return grouped


def sigma_condition_numbers(bundle_path: Path, assets: list[str]) -> dict:
    """Compute condition number of Σ(t) at each date where full 7×7 is available."""
    bundle = pd.read_csv(bundle_path, parse_dates=["date"])
    diag_cols = [f"{a}_sigma_ii" for a in assets]
    cov_cols = {}
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if j <= i:
                continue
            c1 = f"{a1}_{a2}_cov"
            c2 = f"{a2}_{a1}_cov"
            col = c1 if c1 in bundle.columns else (c2 if c2 in bundle.columns else None)
            cov_cols[(a1, a2)] = col

    usable = bundle.dropna(subset=diag_cols + [c for c in cov_cols.values() if c])
    conds: list[float] = []
    for _, row in usable.iterrows():
        sigma = np.zeros((len(assets), len(assets)))
        for i, a in enumerate(assets):
            sigma[i, i] = row[f"{a}_sigma_ii"]
        for (a1, a2), col in cov_cols.items():
            if col is None:
                continue
            i, j = assets.index(a1), assets.index(a2)
            sigma[i, j] = sigma[j, i] = row[col]
        try:
            w = np.linalg.eigvalsh(sigma)
            w = np.clip(w, 1e-20, None)
            conds.append(float(w.max() / w.min()))
        except np.linalg.LinAlgError:
            continue

    if not conds:
        return {"n_usable_dates": 0}
    arr = np.array(conds)
    return {
        "n_usable_dates": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def risk_stats(bundle_dir: Path) -> dict:
    p = bundle_dir / "risk_daily.csv"
    if not p.exists():
        return {}
    risk = pd.read_csv(p, parse_dates=["date"])
    per_asset = (
        risk.groupby("asset")["risk_score"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    return {
        "per_asset": per_asset.to_dict(orient="index"),
        "cross_sectional_std_mean": float(
            risk.groupby("date")["risk_score"].std().mean()
        ),
    }


def nan_coverage(bundle_path: Path) -> dict:
    b = pd.read_csv(bundle_path, parse_dates=["date"])
    n = len(b)
    non_date_cols = [c for c in b.columns if c != "date"]
    nan_per_col = b[non_date_cols].isna().sum()
    full_row_mask = b[non_date_cols].notna().all(axis=1)
    return {
        "total_rows": int(n),
        "fully_populated_rows": int(full_row_mask.sum()),
        "first_fully_populated_date": (
            str(b.loc[full_row_mask, "date"].min().date())
            if full_row_mask.any()
            else None
        ),
        "avg_nan_per_row": float(b[non_date_cols].isna().sum(axis=1).mean()),
        "top_nan_columns": nan_per_col.sort_values(ascending=False).head(5).to_dict(),
    }


def ensemble_agreement(output_dir: Path, top_models: list[str]) -> dict:
    """Compute pairwise correlation of mu_hat_daily across top-k model bundles."""
    frames = {}
    for m in top_models:
        p = output_dir / m / "mu_daily.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        frames[m] = df.set_index(["date", "asset"])["mu_hat_daily"]
    if len(frames) < 2:
        return {}
    joined = pd.concat(frames, axis=1)
    joined.columns = list(frames.keys())
    corr = joined.corr().round(4)
    return {"pairwise_corr": corr.to_dict()}


def build_qa(output_dir: Path, model_dirs: list[str], ensemble_name: str | None) -> tuple[dict, str]:
    assets = ASSETS_EXPECTED
    per_model: dict = {}
    for m in model_dirs:
        bundle_dir = output_dir / m
        if not bundle_dir.exists():
            continue
        bundle_path = bundle_dir / "ode_bundle.csv"
        per_model[m] = {
            "mu_stats_per_asset": mu_stats_per_asset(bundle_dir).to_dict(orient="index"),
            "sigma_condition_number": sigma_condition_numbers(bundle_path, assets),
            "risk_stats": risk_stats(bundle_dir),
            "nan_coverage": nan_coverage(bundle_path),
        }

    ensemble_info = {}
    if ensemble_name:
        ensemble_cfg_path = output_dir / ensemble_name / "ode_config.json"
        if ensemble_cfg_path.exists():
            cfg = json.loads(ensemble_cfg_path.read_text())
            members = cfg.get("ensemble_members", [])
        else:
            members = []
        if not members:
            members = [m for m in model_dirs if m != ensemble_name][:3]
        ensemble_info = ensemble_agreement(output_dir, members)
        ensemble_info["members"] = members

    summary_json = {
        "per_model": per_model,
        "ensemble_agreement_top3": ensemble_info,
    }

    lines = [
        "# ODE Input Bundle — QA Report",
        "",
        "본 리포트는 `ode_inputs_cnn/` 디렉토리 안의 모든 번들에 대해 수치 sanity 를 점검한 결과입니다.",
        "",
        "## 1. μ(t) 분포 요약 (mu_hat_daily)",
        "",
    ]
    for m, info in per_model.items():
        lines.append(f"### {m}")
        mu = pd.DataFrame(info["mu_stats_per_asset"]).T
        lines.append(mu.to_string())
        lines.append("")

    lines += [
        "## 2. Σ(t) 조건수 (full 7×7 covariance, per-date)",
        "",
        "조건수 중간값이 너무 크면 ODE 최적화에서 수치 불안정 가능성. 일반적으로 < 10^3 권장.",
        "",
    ]
    cond_rows = []
    for m, info in per_model.items():
        cond = info["sigma_condition_number"]
        cond_rows.append({"model": m, **cond})
    if cond_rows:
        lines.append(pd.DataFrame(cond_rows).to_string(index=False))
    lines.append("")

    lines += [
        "## 3. risk_score 분포",
        "",
        "교차단면 z-score 기반. 날짜별 std가 0 근처면 신호 degenerate.",
        "",
    ]
    for m, info in per_model.items():
        rs = info.get("risk_stats", {})
        if not rs:
            continue
        lines.append(
            f"- **{m}**: 날짜별 cross-sectional std 평균 = {rs.get('cross_sectional_std_mean', float('nan')):.4f}"
        )
    lines.append("")

    lines += [
        "## 4. ode_bundle NaN 커버리지",
        "",
    ]
    nan_rows = []
    for m, info in per_model.items():
        nc = info["nan_coverage"]
        nan_rows.append({
            "model": m,
            "total_rows": nc["total_rows"],
            "full_rows": nc["fully_populated_rows"],
            "first_full_date": nc["first_fully_populated_date"],
        })
    if nan_rows:
        lines.append(pd.DataFrame(nan_rows).to_string(index=False))
    lines.append("")

    if ensemble_info:
        members = ensemble_info.get("members", [])
        lines += [
            "## 5. 앙상블 구성원 간 μ 상관",
            "",
            f"앙상블 멤버 ({len(members)}개, OOS rank corr 상위): {', '.join(members)}",
            "",
            "멤버 간 mu_hat_daily Pearson 상관 매트릭스.",
            "상관이 지나치게 높으면 앙상블 효과 미미, 낮으면 diversification 효과 큼.",
            "",
        ]
        corr = pd.DataFrame(ensemble_info["pairwise_corr"])
        lines.append(corr.round(4).to_string())
        lines.append("")

    return summary_json, "\n".join(lines) + "\n"


# ─── README ───────────────────────────────────────────────────────────────────

def build_readme(
    output_dir: Path,
    comparison_df: pd.DataFrame,
    best_model: str,
    ensemble_name: str | None,
    assets: list[str],
    horizon: int,
    sigma_window: int,
    data_date_range: tuple[str, str],
) -> str:
    top3 = comparison_df[comparison_df["model_name"].isin(
        [m for m in comparison_df["model_name"] if not str(m).startswith("ensemble_")]
    )].head(3)["model_name"].tolist()

    return dedent(f"""
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
    ├── {ensemble_name or "ensemble_top3"}/          앙상블 (top-3 CNN signal 평균)
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

    - **자산 (7)**: {", ".join(assets)}
    - **공통 유효 날짜**: {data_date_range[0]} ~ {data_date_range[1]}
    - **거래일 기준 grid**: business days (주말·공휴일 제외, forward-fill 안 함)
    - **horizon**: {horizon} 영업일 (μ 예측 지평)
    - **Σ rolling window**: {sigma_window} 영업일

    ## 3. 파일별 스키마 & 단위

    ### `returns_daily.csv`
    날짜 × 자산 일별 **로그 수익률**. 논문 Section 6가 요구하는 primary input.
    - 컬럼: `date, alternative, corp_bond_ig, ..., treasury_7_10y`
    - 단위: daily log return (절대 스케일), forward-fill 없음
    - 용도: ODE solver 가 원한다면 자체 rolling-mean μ 를 재계산하거나, backtest 시 실제 수익률 참조

    ### `prices_daily.csv`
    동일 그리드의 **close price**. 포트폴리오 가치 추적 / 거래비용 시뮬레이션에 사용.

    ### `{{model}}/mu_daily.csv`
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

    ### `{{model}}/risk_daily.csv`
    | column | 설명 |
    |--------|------|
    | `date` | 영업일 |
    | `asset` | 자산명 |
    | `risk_score` | 교차단면 z-score (클수록 downside risk 높음) |
    | `target` | 실제 downside 값 (OOS 검증용) |

    ODE 에서의 쓰임:
    - γ(t) 를 자산별로 조절: `γ_i(t) = γ_0 × exp(k × risk_score)`
    - 또는 μ 에서 차감: `μ_adj(t,i) = μ(t,i) − δ × risk_score(t,i)`

    ### `{{model}}/ode_bundle.csv` (권장 consumer 진입점)
    Wide 포맷. 한 행 = 한 날짜. 모든 신호가 같은 날짜 축에 정렬돼 있음.

    | prefix | 의미 |
    |--------|------|
    | `{{asset}}_mu` | calibrated μ (daily scale) |
    | `{{asset}}_mu_raw` | raw CNN score |
    | `{{asset}}_sigma_ii` | diagonal variance (daily) |
    | `{{asset}}_risk` | risk_score |
    | `{{a}}_{{b}}_cov` | off-diagonal covariance (daily, symmetric) |

    ### `{{model}}/ode_config.json`
    모델명, horizon, sigma_window, 자산 리스트, 날짜 범위, OOS 품질, calibration 파라미터.
    `mu_calibration` 필드는 다음 sprint가 재현·수정할 때 참고.

    ## 4. 권장 consumer 코드

    ```python
    import numpy as np
    import pandas as pd
    from pathlib import Path

    BUNDLE_DIR = Path("ode_inputs_cnn/{best_model}")  # 최고 단일 모델
    # 또는 Path("ode_inputs_cnn/{ensemble_name or "ensemble_top3"}")  # 앙상블

    bundle = pd.read_csv(BUNDLE_DIR / "ode_bundle.csv", parse_dates=["date"])
    assets = {assets!r}

    mu_mat     = bundle[[f"{{a}}_mu" for a in assets]].to_numpy()       # (T, 7)
    sigma_diag = bundle[[f"{{a}}_sigma_ii" for a in assets]].to_numpy() # (T, 7)

    # 7x7 covariance matrix per date
    def sigma_matrix(row, assets):
        n = len(assets)
        M = np.zeros((n, n))
        for i, a in enumerate(assets):
            M[i, i] = row[f"{{a}}_sigma_ii"]
        for i, a1 in enumerate(assets):
            for j, a2 in enumerate(assets):
                if j <= i:
                    continue
                col = f"{{a1}}_{{a2}}_cov" if f"{{a1}}_{{a2}}_cov" in row else f"{{a2}}_{{a1}}_cov"
                M[i, j] = M[j, i] = row[col]
        return M

    # 원본 일별 수익률 (backtest 용)
    returns = pd.read_csv("ode_inputs_cnn/returns_daily.csv", parse_dates=["date"])
    ```

    ## 5. 모델 선택 가이드

    [comparison.md](comparison.md) 참조.

    - **추천 기본값**: `{best_model}` — {"앙상블 (top-3 CNN 평균)" if ensemble_name and best_model == ensemble_name else "단일 모델 중 OOS rank corr 1위"}
    - **1순위 단일 CNN**: `{top3[0] if len(top3) > 0 else "—"}`
    - **대안 앙상블**: `{ensemble_name or "ensemble_top3"}` — 개별 모델의 noise 완화, 로버스트하지만 보수적
    - **실험 용도**: 그 외 CNN 번들들은 sensitivity/ablation 시 비교용

    ## 6. 보증

    - **No lookahead**:
      - μ calibration 은 walk-forward expanding window 만 사용
      - Σ 는 rolling window (과거 {sigma_window}일만 사용)
      - risk_score 는 날짜별 교차단면 z-score (시간축 정보 안 씀)
    - **공통 샘플**: 7 자산 전부 유효한 날짜만 사용 (NaN 없음)
    - **수익률 스케일 일관**: μ, Σ 모두 daily log-return 스케일 (논문 Section 6 규약과 일치)

    ## 7. 다음 sprint 가 해야 할 것

    1. 이 번들을 읽어 (μ, Σ, risk) 시계열 추출
    2. γ(t) 스케줄 선택 (paper: `γ(t) = γ_0 × exp(−λt)` 예시, γ_0∈{{1.5, 3.5}})
    3. Simplex projection 포함 Euler / RK4 ODE solver 돌리기
    4. `returns_daily.csv` 로 realized PnL / turnover / Sharpe 평가
    5. 다른 모델 번들로 바꿔가며 sensitivity 확인
    """).strip() + "\n"


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--output-dir", default="ode_inputs_cnn")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--sigma-window", type=int, default=60)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"{output_dir} not found — run collect_cnn_ode_signals.py first")

    # 1. returns & prices matrices
    print("Building returns/prices matrices ...")
    prices, returns = build_price_matrices(args.data_path)
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    returns["date"] = pd.to_datetime(returns["date"]).dt.strftime("%Y-%m-%d")
    prices.to_csv(output_dir / "prices_daily.csv", index=False)
    returns.to_csv(output_dir / "returns_daily.csv", index=False)
    date_range = (returns["date"].min(), returns["date"].max())
    print(f"  returns: {len(returns)} dates ({date_range[0]} ~ {date_range[1]})")

    # 2. QA report (scan bundles)
    print("Building QA report ...")
    comparison_path = output_dir / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"{comparison_path} missing — run collect_cnn_ode_signals.py")
    comparison = pd.read_csv(comparison_path)
    model_dirs = [
        d.name for d in output_dir.iterdir()
        if d.is_dir() and (d / "ode_bundle.csv").exists()
    ]
    ensemble_name = next((m for m in model_dirs if m.startswith("ensemble_")), None)
    qa_json, qa_md = build_qa(output_dir, model_dirs, ensemble_name)
    (output_dir / "qa_report.md").write_text(qa_md, encoding="utf-8")
    (output_dir / "qa_report.json").write_text(json.dumps(qa_json, indent=2, default=float), encoding="utf-8")
    print(f"  qa_report.md / qa_report.json → {output_dir}/")

    # 3. README
    print("Writing README.md ...")
    best_model = str(comparison.iloc[0]["model_name"])
    readme = build_readme(
        output_dir=output_dir,
        comparison_df=comparison,
        best_model=best_model,
        ensemble_name=ensemble_name,
        assets=ASSETS_EXPECTED,
        horizon=args.horizon,
        sigma_window=args.sigma_window,
        data_date_range=date_range,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"  README.md → {output_dir}/")

    print("\n=== Hand-off package complete ===")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            print(f"  {f.name}")
        else:
            print(f"  {f.name}/")


if __name__ == "__main__":
    main()
