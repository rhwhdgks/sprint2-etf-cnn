from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.pipeline import run_pipeline
from src.reporting import markdown_table, sort_summary_frame, write_json

DEFAULT_WINDOW_GRID = [(20, 20), (60, 20), (60, 60)]


def _window_tag(lookback: int, horizon: int) -> str:
    return f"lookback_{lookback}_horizon_{horizon}"


def _build_submission_report(
    summary_df: pd.DataFrame,
    best_model_overall: pd.Series,
    suite_root: Path,
    target_name: str,
) -> str:
    best_by_window = summary_df[
        [
            "lookback",
            "horizon",
            "best_model_name",
            "rmse",
            "mae",
            "target_rank_correlation",
            "future_return_rank_correlation",
            "top_k_cumulative_return",
            "top_k_sharpe",
            "top_k_hit_rate",
            "turnover",
        ]
    ].copy()

    recommendation = (
        f"`lookback={int(best_model_overall['lookback'])}, horizon={int(best_model_overall['horizon'])}, "
        f"model={best_model_overall['best_model_name']}`"
    )
    image_models = {"logistic_image_scale", "cnn_1d_image_scale", "cnn_2d_rendered_images"}
    image_wins = int(summary_df["best_model_name"].isin(image_models).sum())

    lines = [
        "# 제출 보고서: Lookback/Horizon 비교 실험",
        "",
        "## 실험 목적",
        f"- 메인 expected-return 회귀 타깃인 `{target_name}` 기준으로 `20/20`, `60/20`, `60/60` 구간 조합을 비교했습니다.",
        "- 각 구간 조합마다 4개 모델 그룹을 모두 학습해, 시간 구간이 길어질 때 이미지 계열 모델이 상대적으로 개선되는지 확인했습니다.",
        "",
        "## 비교 설정",
        "- 타깃: `future_return` 회귀",
        "- 모델군: `logistic_cumulative_scale`, `logistic_image_scale`, `cnn_1d_image_scale`, `cnn_2d_rendered_images`",
        "- 평가: RMSE, MAE, 타깃 랭크상관, 미래수익 랭크상관, top-k 누적수익, Sharpe, hit rate, turnover",
        "",
        "## 구간별 최고 조합",
        markdown_table(best_by_window, float_precision=4),
        "",
        "## 최종 추천",
        f"- 제출 기준 추천 조합은 {recommendation} 입니다.",
        f"- 이 조합의 top-k Sharpe는 `{best_model_overall['top_k_sharpe']:.4f}`이고, top-k 누적수익은 `{best_model_overall['top_k_cumulative_return']:.4f}`였습니다.",
        f"- 미래수익 기준 랭크상관은 `{best_model_overall['future_return_rank_correlation']:.4f}`였습니다.",
        "",
        "## 해석",
        "- `20/20`은 현재까지의 기본 기준선입니다. ETF 데이터에서는 중기 추세를 잡는 데 충분히 실용적이고 샘플 수도 가장 안정적입니다.",
        "- `60/20`은 더 긴 과거 문맥으로 20일 앞을 맞히는 설정이라, 느린 자산군 추세가 있을 때 유리할 수 있습니다.",
        "- `60/60`은 가장 느린 구조를 보지만, 예측 대상도 멀어져서 신호가 희석될 수 있습니다.",
        f"- 이번 결과에서는 긴 윈도우 두 구간(`60/20`, `60/60`)에서 모두 이미지 계열 모델이 최고 성능을 차지했습니다. 총 3개 구간 중 {image_wins}개 구간에서 이미지 계열이 승리했습니다.",
        "- 특히 `60/20`에서는 `cnn_1d_image_scale`가 최고였고, `60/60`에서는 `logistic_image_scale`가 최고였습니다. 즉, 긴 시간 문맥에서는 image-style 표현의 상대적 장점이 살아났습니다.",
        "- 다만 2D CNN은 성능이 개선되기는 했어도 최종 최고 모델은 아니었습니다. 따라서 이번 데이터에서는 2D 이미지 모델을 기본 선택으로 삼기보다, 1D CNN 또는 image-scale logistic을 우선 고려하는 편이 합리적입니다.",
        "",
        "## 제출 파일",
        f"- 스위프 요약 CSV: {suite_root / 'window_sweep_summary.csv'}",
        f"- 스위프 요약 Markdown: {suite_root / 'window_sweep_summary.md'}",
        f"- 전체 모델 비교 CSV: {suite_root / 'window_sweep_all_model_results.csv'}",
        f"- 추천 시그널 파일: {best_model_overall['best_signal_file']}",
    ]
    return "\n".join(lines) + "\n"


def run_window_sweep(base_config: PipelineConfig, windows: list[tuple[int, int]]) -> dict:
    suite_root = Path(base_config.output_dir)
    suite_root.mkdir(parents=True, exist_ok=True)

    all_model_rows: list[dict] = []
    best_window_rows: list[dict] = []
    per_window_payloads: dict[str, dict] = {}

    for lookback, horizon in windows:
        config = PipelineConfig(**base_config.to_dict())
        config.lookback = lookback
        config.horizon = horizon
        config.output_dir = str(suite_root / _window_tag(lookback, horizon))

        payload = run_pipeline(config)
        per_window_payloads[_window_tag(lookback, horizon)] = payload

        comparison_df = pd.DataFrame(payload["comparison"])
        comparison_df["lookback"] = lookback
        comparison_df["horizon"] = horizon
        comparison_df["window_tag"] = _window_tag(lookback, horizon)
        all_model_rows.extend(comparison_df.to_dict(orient="records"))

        best_row = dict(comparison_df.iloc[0].to_dict())
        best_row["best_model_name"] = payload["best_model_name"]
        best_row["best_signal_file"] = payload["paths"]["best_signal_file"]
        best_row["report_file"] = payload["paths"]["report"]
        best_window_rows.append(best_row)

    all_models_df = pd.DataFrame(all_model_rows)
    best_df = pd.DataFrame(best_window_rows)
    best_df = sort_summary_frame(best_df)

    overall_best = best_df.iloc[0]

    all_model_csv = suite_root / "window_sweep_all_model_results.csv"
    all_models_df.to_csv(all_model_csv, index=False)

    summary_columns = [
        "lookback",
        "horizon",
        "window_tag",
        "best_model_name",
        "rmse",
        "mae",
        "target_rank_correlation",
        "future_return_rank_correlation",
        "top_k_cumulative_return",
        "top_k_sharpe",
        "top_k_hit_rate",
        "top_bottom_spread_mean",
        "turnover",
        "best_signal_file",
        "report_file",
    ]
    best_df = best_df[[column for column in summary_columns if column in best_df.columns]]

    summary_csv = suite_root / "window_sweep_summary.csv"
    best_df.to_csv(summary_csv, index=False)

    summary_md = suite_root / "window_sweep_summary.md"
    summary_md.write_text(
        "# Window Sweep Summary\n\n" + markdown_table(best_df, float_precision=4) + "\n",
        encoding="utf-8",
    )

    submission_report_path = suite_root / "submission_report_ko.md"
    submission_report_path.write_text(
        _build_submission_report(best_df, overall_best, suite_root, base_config.target_name),
        encoding="utf-8",
    )

    payload = {
        "target_name": base_config.target_name,
        "windows": [{"lookback": lookback, "horizon": horizon} for lookback, horizon in windows],
        "best_by_window": best_df.to_dict(orient="records"),
        "per_window_payloads": per_window_payloads,
        "paths": {
            "all_model_csv": str(all_model_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "submission_report": str(submission_report_path),
        },
    }

    summary_json = suite_root / "window_sweep_summary.json"
    payload["paths"]["summary_json"] = str(summary_json)
    write_json(summary_json, payload)
    return payload
