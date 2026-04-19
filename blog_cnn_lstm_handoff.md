# 주가 차트를 이미지로 바꿔서 CNN에 넣어봤다 — 그리고 왜 다음엔 LSTM이 필요한가

> 7개 ETF 자산으로 포트폴리오를 짤 때 "어떤 자산이 더 오를까"를 예측하는 문제에 CNN 7종을 붙였고, 실험 도중 **베이스라인에 졌고 → logistic 섞어서 이겼고 → 그래도 찝찝해서 2D CNN 재조정하니 천장이 한 번 더 뚫렸다**. 이 글은 그 궤적과, **왜 다음 단계로 LSTM을 얹는 게 자연스러운가**를 정리한 것.
>
> **최종 결과**: logistic + 1D CNN + 2D CNN (재조정판) 3-family mixed ensemble로 rank corr **0.061**, top-k Sharpe **0.643**을 동시 1위 달성. 이 번들이 다음 ODE 스프린트의 default 입력.

---

## 1. 시작 — 뭘 만들려고 했나

요즘 한 팀에서 **ODE(미분방정식) 기반 동적 포트폴리오 최적화**라는 주제를 다루고 있다. 쉽게 말하면:

> "매일매일 시장이 변하는데, 7개 자산(주식·채권·대체투자 등)에 **얼마씩 넣을지를 수식으로 풀어서 정한다**"

이 수식에는 세 가지 입력이 필요하다.

- **μ(mu)**: 각 자산이 앞으로 얼마나 오를지 (기대수익)
- **Σ(sigma)**: 자산들이 같이 움직이는 정도 (공분산)
- **R**: 실제 일별 수익률

이 중 Σ와 R은 과거 데이터로 기계적으로 계산하면 된다. 어려운 건 **μ** — "**미래** 수익을 맞춰야 하는 것"이다. 당연히 완벽하게 맞출 순 없지만, 적어도 **랭킹(어떤 게 더 오를 것 같은지)**만 잘 맞아도 포트폴리오에 도움이 된다.

나는 이 팀에서 **μ 예측 시그널을 CNN으로 만드는 파트**를 맡았다.

---

## 2. Jiang-style 이미지 변환 — 주가를 그림으로

보통 주가 예측 모델은 숫자 시퀀스를 그대로 넣는다. `[100, 102, 101, 103, ...]` 같은 식.

그런데 2016년 Jiang이라는 연구자가 이런 제안을 했다:

> "**사람 트레이더도 차트를 보고 판단한다**. 그럼 모델도 차트 이미지를 보게 하자"

그래서 60일치 OHLCV(시가·고가·저가·종가·거래량) 데이터를 **정규화된 2D 이미지**로 변환한다. 가로축은 시간, 세로축은 가격 레벨, 픽셀 값은 캔들 모양·거래량 강도 등으로 채운다.

```
숫자 sequence  →  이미지  →  CNN (컴퓨터 비전 모델)  →  μ 예측값
```

CNN은 원래 고양이·강아지 사진 구분용으로 유명한 모델이다. 그걸 주가 차트에 붙이는 게 Jiang-style의 아이디어.

---

## 3. CNN 7종 돌려본 결과

"어떤 CNN 구조가 이 문제에 제일 잘 맞을까?" 궁금해서 7가지를 준비했다:

| # | 모델 | 특징 |
|---|---|---|
| 1 | `cnn_1d_image_scale` | 기본형. 이미지를 1D로 펼쳐서 CNN |
| 2 | `cnn_1d_multiscale` | 여러 커널 크기 동시 사용 |
| 3 | `cnn_1d_dilated` | 확장 합성곱 (멀리 떨어진 픽셀도 관계) |
| 4 | `cnn_1d_attention` | Squeeze-Excitation 어텐션 |
| 5 | `cnn_1d_cumulative` | 이미지 대신 누적수익률 시퀀스 입력 |
| 6 | `cnn_2d_rendered` | 2D 캔들 이미지 |
| 7 | `cnn_2d_residual` | 2D + ResNet 블록 |

모두 **walk-forward out-of-sample**으로 평가했다. 풀어서 설명하면:

> "2015년까지 학습 → 2016년 예측 → 2016년까지 학습 → 2017년 예측..." 식으로 **미래를 본 적 없는 상태에서만 예측하게** 한 것. 금융 모델링에서 가장 흔한 함정이 "실수로 미래 데이터를 참고해서 좋은 성적이 나오는 것"(leakage)이라 이걸 엄격히 막았다.

총 **24 fold × 7년 OOS**, 평가 날짜 2,880일.

### 📊 결과

![Model comparison](ode_inputs_cnn/figures/01_model_comparison.png)

- 왼쪽: **Spearman rank correlation** — 예측 랭킹과 실제 수익 랭킹이 얼마나 비슷한지 (높을수록 좋음, 완전 랜덤이면 0)
- 오른쪽: **Top-k 포트폴리오 Sharpe ratio** — 예측 상위 2개 자산에 투자한 전략의 실전 성과

가장 눈에 띈 건 **`cnn_1d_cumulative_scale`이 Sharpe 0.52로 압도**. 그리고 2D CNN 두 개는 **거의 바닥**. (후에 이 2D CNN 바닥이 **모델 잘못이 아니라 학습 프로토콜 잘못**이었다는 게 밝혀지는데, 그건 §6-C에서.)

---

## 4. 그런데... 베이스라인에 졌다

실험 설계에 **"CNN 없는 로지스틱 회귀"** 베이스라인을 두 개 넣어뒀다.

| 모델 | 이미지 입력? | CNN? |
|---|---|---|
| `logistic_cumulative_scale` | ❌ | ❌ (진짜 단순) |
| `logistic_image_scale` | ✅ | ❌ (이미지 + 로지스틱) |

결과:

| 지표 1등 | 모델 | 값 |
|---|---|---|
| **Rank correlation** | `logistic_image_scale` 😱 | **0.0392** |
| Top-k Sharpe | `cnn_1d_cumulative_scale` | 0.521 |
| Ensemble (CNN top-3) | — | 0.0320 |

**이미지를 넣은 로지스틱 회귀가 우리 CNN들을 rank correlation에서 이겼다.** CNN의 최고(앙상블 0.032)보다도 높다.

이게 뭘 뜻하는지 더 깊게 보려고 **2×2 ablation**을 그렸다:

![Ablation](ode_inputs_cnn/figures/09_ablation_image_vs_cnn.png)

- 🟦 이미지 없음 + 로지스틱: **−0.0072** (꽝)
- 🟥 이미지 + 로지스틱: **+0.0392** (최고)
- 🟧 이미지 없음 + CNN: +0.0271
- 🟧 이미지 + CNN: +0.0280

### 해석

1. **"이미지 변환" 자체가 lift의 대부분을 만든다** (−0.007 → +0.039)
2. CNN은 이미지 위에 얹혀도 rank corr를 개선하지 못한다 (0.039 → 0.028로 오히려 감소)
3. **하지만 포트폴리오 Sharpe에서는 CNN이 이긴다** (logistic 0.39 < CNN 0.52)

즉, CNN의 가치는 "**점예측 정확도**"가 아니라 "**선택·랭킹을 통한 포트폴리오 구성력**"에 있다. 다른 metric에서 각각 챔피언이 다르다는 것이 이번 실험의 핵심 인사이트.

---

## 5. 앙상블로 조금 더 — CNN끼리만 섞으면 천장에 막힌다

"단일 모델이 부족하면 여러 개 합치자"는 고전적 대응. Top-3 CNN의 raw score 평균을 낸 앙상블을 만들었다.

- Rank corr: **0.032** (단일 최고 0.028보다 ↑)
- Top-k Sharpe: **0.374** (logistic_image 0.385에 근접)
- Positive-fold 비율: 52.1%

![Rolling rank corr](ode_inputs_cnn/figures/06_rolling_rank_corr.png)

시간에 따른 안정성을 보면 앙상블(굵은 빨강)이 단일 모델들보다 덜 튄다. 2020년 팬데믹 같은 regime shift 구간에서도 빠르게 회복.

**하지만 CNN끼리의 앙상블로는 `logistic_image_scale` 0.039도 못 넘는다.** CNN 가문 안에서만 섞으면 이 천장이 있다.

---

## 6. 왜 CNN끼리의 앙상블은 천장에 막혔나

모델 간 raw score 상관을 찍어보면:

![Model correlation](ode_inputs_cnn/figures/05_model_raw_correlation.png)

CNN 1D 계열끼리 **0.5~0.6 상관**. 비슷한 정보를 다른 방식으로 표현하고 있을 뿐이다. 앙상블의 힘은 "**상관 낮은 신호**를 합칠 때" 나오는데, CNN 구성원들은 이미 너무 닮았다.

---

## 6-B. 천장 뚫기 — logistic을 앙상블에 태우자

위 관찰에서 바로 나오는 가설: **CNN이 아닌 다른 family를 섞으면 상관이 낮아져서 천장 위로 올라갈 수 있다.** 가장 가까이 있는 비CNN 모델이 `logistic_image_scale`. 9개 모델 전체에서 size 2~4 조합을 전수 탐색했다 (raw 평균 · cross-sectional rank 평균 두 방식).

결과(Phase 1) — **`ensemble_best` v1** = `logistic_image_scale` + `cnn_1d_attention_image_scale` + `cnn_1d_cumulative_scale` (rank 평균): rank corr **0.0422**, Sharpe **0.503**. 단독 logistic_image(0.039) 천장 돌파.

즉 **"CNN 앙상블에 logistic을 태우기"** 만으로도 두 metric을 동시에 상위로 올렸다. 뼈아픈 교훈: **단독 family로 경쟁하기보다, 다른 family를 멤버로 초대하는 게 더 효과적**이라는 것.

---

## 6-C. 다시 들어가본 2D CNN — overfit이 아니라 "덜 배워서 underperform"이었다

여기서 멈추기엔 찝찝한 구석이 있었다: **2D CNN이 개별 성능 0.007로 처참했던 이유**. 처음엔 "데이터 부족으로 overfit"이 가장 그럴듯해 보였다. 확인하려고 단일 fold로 학습곡선을 찍어봤다:

![2D CNN loss curves](ode_inputs_cnn/figures/10_2d_cnn_loss_curves.png)

- 2D residual: **epoch 18에서 최적 val loss**. 우리 기본 설정은 **8 epoch + patience 2** — train이 끝나기도 전에 조기 종료
- 1D dilated: epoch 5에서 최적 → 1D는 8 epoch로 충분
- val loss가 꾸준히 내려가는 모양 → overfit이 아니라 **그냥 덜 배운 상태**

즉 문제는 overfit이 아니라 **undertrain + overparam**이었다. Phase 2로 2D만 재훈련 (30 epoch + patience 5):

| 변형 | params | 설정 | rank corr | Sharpe |
|---|---|---|---|---|
| 원본 `cnn_2d_residual_images` | 60K | 8 ep, wd 1e-4 | 0.007 | 0.10 |
| `cnn_2d_residual_wd` (wd만 강화) | 60K | 30 ep, **wd 5e-4** | 0.006 | 0.25 |
| ★ `cnn_2d_residual_small` | **23K** | 30 ep, **wd 5e-4, dropout 0.2** | **0.043** | 0.21 |

**핵심**: capacity(1/3 축소) + strong wd + dropout을 **모두** 걸어야 유효. WD만 강화하면 오히려 망가짐. **`cnn_2d_residual_small`이 단일 CNN 중 rank corr 1위**.

이 새 2D를 Phase 3 ensemble 전수 탐색에 넣으니 다시 천장이 뚫림 — **`ensemble_best` v2** = `logistic_image_scale` + `cnn_1d_cumulative_scale` + `cnn_2d_residual_small`:

| 지표 | v1 (Phase 1) | v2 (Phase 3) | 변화 |
|---|---|---|---|
| OOS rank corr | 0.042 | **0.061** | +45% |
| Top-k Sharpe | 0.503 | **0.643** | +28% |
| 두 metric 동시 1위 | ✓ | ✓ | 여전히 유일 |

세 멤버 모두 **다른 family** — logistic (선형) + 1D CNN (no-image) + 2D CNN (image). Correlation이 최소화되면서 ensemble 효과가 최대화된 케이스.

그럼 다음 질문 — **CNN이 놓치는 축은 뭘까?** 힌트가 하나 있다:

> **Jiang-style 이미지는 시간을 "공간"으로 바꿔 넣는다.** CNN은 2D 패턴을 학습하지만, **"이 시점 다음에 저 시점이 온다"는 명시적 순서 정보는 흐려진다.**

---

## 7. 그래서 다음엔 LSTM — 왜 같은 함정에 안 빠지는가

LSTM(Long Short-Term Memory)은 시퀀스 전용으로 태어난 모델이다. 핵심 특징:

- **명시적 시간 순서 처리**: t 시점 정보를 t+1 시점으로 "전달"하는 게이트 구조
- **메모리 게이트**: 오래된 정보 중 "기억할 것"과 "잊을 것"을 학습으로 고름
- **regime 변화에 민감**: 패턴이 바뀌는 구간을 포착하기 유리

CNN이 "사진 한 장을 보고 판단하는 사람"이라면, LSTM은 "**연속 장면을 이어보며 맥락을 쌓는 사람**"이다.

### 기대하는 것

1. **CNN과 정보 축이 다르다** → 상관이 낮을 것 (CNN-CNN 0.5 vs CNN-LSTM 예상 < 0.3)
2. **상관이 낮으면 앙상블 폭발력이 커진다** — 방금 `ensemble_best`가 보여준 것처럼, mixed family가 천장을 뚫는다. LSTM은 이 믹스를 한 단계 더 다양화하는 카드
3. **발표 스토리가 완성된다**: "이미지(CNN = 공간) + 시퀀스(LSTM = 시간) + 선형(logistic = 평균장)의 **3-way 상보성**"이라는 해석

### 실패할 수도 있는 지점

- LSTM도 결국 주가 시계열 특유의 노이즈엔 약할 수 있다
- 학습 데이터가 적으면 overfitting 위험 (CNN보다 데이터 탐식)
- 우리의 `logistic_cumulative`(시퀀스+단순모델)가 이미 실패(-0.007)한 점을 고려하면, 시퀀스 입력 자체가 어려운 과제일 수도

이런 리스크를 인정하고, LSTM이 **개별 성능 0.02~0.03**만 내도 앙상블 기여로는 충분하다는 실용주의로 간다.

---

## 8. 정리

### ✅ 지금까지 얻은 것
- 8개 CNN 아키텍처의 OOS 성능 맵 (+ Phase 2에서 2D 재조정판 포함)
- 이미지 변환과 CNN 기여를 분리한 ablation
- Rank corr와 portfolio Sharpe가 **다른 챔피언**을 가리키다가, 최종적으로 한 조합이 **동시 1위**가 되는 과정 추적
- CNN-only 앙상블의 한계 (0.039 천장) → logistic 섞어 1차 돌파 (0.042) → 2D CNN 재조정 후 2차 돌파 (0.061)
- 2D CNN underperform의 원인 규명: overfit이 아니라 "capacity 과다 + 학습량 부족"
- ODE 스프린트가 바로 집어 쓸 수 있는 μ·Σ·R·risk 번들 — `ensemble_best` v2 기준 rank 0.061 / Sharpe 0.643

### 🎯 다음에 할 것
- LSTM 합류: walk-forward 포맷 통일해서 같은 평가 그리드에 올리기
- `ensemble_best`에 LSTM을 추가해서 3-way mixed ensemble로 더 올릴 수 있는지 검증
- 성공 시 ODE 스프린트에 `ensemble_cnn_lstm_logistic` 번들 전달

### 🧠 세 줄 교훈
> **① "새 모델이 최고 성능을 낸다"는 이야기는 반드시 비CNN 베이스라인과 함께 비교해야 설득력이 있다.** logistic을 베이스로 두지 않았다면 "CNN이 잘 된다"고 말하고 넘어갔을 것.
>
> **② 앙상블은 "같은 family 여럿"보다 "다른 family 혼합"이 더 효과적.** CNN끼리만 섞어서는 천장을 못 뚫었지만, logistic + 1D CNN + 2D CNN의 3-family 혼합으로 0.032 → 0.061로 두 번 뚫렸다. LSTM도 같은 논리로 합류시키는 것.
>
> **③ "underperform"의 원인을 단정짓지 말고 학습곡선부터 찍어라.** 2D CNN을 "데이터 부족"으로 포기할 뻔했지만, 학습곡선을 찍어보니 **단지 일찍 멈춰서**였다. Capacity 줄이고 학습 시간 늘리니 단독 CNN 중 1위로 올라왔고, ensemble 천장도 같이 뚫렸다.

---

## 부록: 용어 정리

- **OOS (Out-of-Sample)**: 모델이 학습에 쓰지 않은 기간에 대한 예측
- **Walk-forward**: 시간 순서대로 학습→예측→학습→예측을 반복하는 평가 방식
- **Rank correlation (Spearman)**: 예측 랭킹과 실제 랭킹의 일치도 (−1 ~ +1)
- **Top-k Sharpe**: 예측 상위 k개 자산에 투자한 전략의 위험조정 수익률
- **ODE**: 미분방정식. 여기선 포트폴리오 비중의 시간 변화를 수식으로 풀기 위한 도구
- **μ, Σ**: 각각 기대수익과 공분산 — 포트폴리오 이론의 두 핵심 입력

---

*스프린트 저장소의 실제 숫자·figure·코드는 `ode_inputs_cnn/HANDOFF_SUMMARY.md`에서 확인 가능.*
