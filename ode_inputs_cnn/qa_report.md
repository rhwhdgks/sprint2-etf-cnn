# ODE Input Bundle — QA Report

본 리포트는 `ode_inputs_cnn/` 디렉토리 안의 모든 번들에 대해 수치 sanity 를 점검한 결과입니다.

## 1. μ(t) 분포 요약 (mu_hat_daily)

### cnn_2d_residual_images
                      mean       std       min       max
alternative       0.000261  0.000857 -0.002146  0.002136
corp_bond_ig     -0.000195  0.000762 -0.002115  0.001901
developed_equity  0.000200  0.000729 -0.002000  0.001901
emerging_equity  -0.000216  0.000724 -0.002066  0.002277
korea_equity     -0.000112  0.000793 -0.002052  0.002149
short_treasury    0.000121  0.000834 -0.002139  0.002198
treasury_7_10y   -0.000059  0.000760 -0.002064  0.002024

### cnn_2d_rendered_images
                      mean       std       min       max
alternative       0.000392  0.000792 -0.001879  0.002030
corp_bond_ig     -0.000296  0.000744 -0.001921  0.001874
developed_equity  0.000162  0.000737 -0.001977  0.002002
emerging_equity  -0.000270  0.000654 -0.001717  0.001994
korea_equity     -0.000125  0.000772 -0.002019  0.002040
short_treasury    0.000165  0.000867 -0.001994  0.002197
treasury_7_10y   -0.000028  0.000784 -0.001869  0.001990

### cnn_1d_image_scale
                      mean       std       min       max
alternative       0.000408  0.000840 -0.002126  0.002139
corp_bond_ig     -0.000186  0.000753 -0.002012  0.001862
developed_equity  0.000163  0.000697 -0.001682  0.001973
emerging_equity  -0.000219  0.000659 -0.001980  0.001718
korea_equity     -0.000162  0.000764 -0.002020  0.002089
short_treasury    0.000076  0.000873 -0.002076  0.002058
treasury_7_10y   -0.000080  0.000802 -0.001910  0.002046

### cnn_1d_dilated_image_scale
                      mean       std       min       max
alternative       0.000194  0.000805 -0.002184  0.002140
corp_bond_ig     -0.000135  0.000870 -0.002090  0.002087
developed_equity  0.000043  0.000709 -0.002043  0.001846
emerging_equity  -0.000091  0.000715 -0.002013  0.002063
korea_equity     -0.000058  0.000782 -0.001955  0.002135
short_treasury    0.000061  0.000858 -0.001915  0.002082
treasury_7_10y   -0.000014  0.000810 -0.001979  0.002142

### cnn_1d_cumulative_scale
                      mean       std       min       max
alternative       0.000491  0.001111 -0.002239  0.002242
corp_bond_ig     -0.000254  0.000501 -0.001926  0.002056
developed_equity  0.000338  0.001014 -0.002151  0.002202
emerging_equity  -0.000262  0.000489 -0.001863  0.001962
korea_equity     -0.000179  0.000609 -0.002046  0.002082
short_treasury   -0.000048  0.000655 -0.002125  0.002375
treasury_7_10y   -0.000086  0.000643 -0.001980  0.002150

### ensemble_top3
                      mean       std       min       max
alternative       0.000297  0.000779 -0.002091  0.002169
corp_bond_ig     -0.000240  0.000831 -0.002150  0.002044
developed_equity  0.000132  0.000731 -0.002228  0.001796
emerging_equity  -0.000168  0.000714 -0.002084  0.002002
korea_equity     -0.000126  0.000796 -0.001986  0.002100
short_treasury    0.000093  0.000845 -0.002042  0.002085
treasury_7_10y    0.000013  0.000770 -0.001958  0.002010

### cnn_1d_attention_image_scale
                      mean       std       min       max
alternative       0.000278  0.000779 -0.002019  0.002103
corp_bond_ig     -0.000238  0.000818 -0.002181  0.002114
developed_equity  0.000072  0.000690 -0.001911  0.001725
emerging_equity  -0.000155  0.000753 -0.002081  0.001955
korea_equity     -0.000120  0.000831 -0.001911  0.002066
short_treasury    0.000124  0.000856 -0.001964  0.002050
treasury_7_10y    0.000039  0.000748 -0.001876  0.002045

### cnn_1d_multiscale_image_scale
                      mean       std       min       max
alternative       0.000465  0.000802 -0.002002  0.002156
corp_bond_ig     -0.000277  0.000761 -0.001911  0.002086
developed_equity  0.000261  0.000648 -0.001988  0.001810
emerging_equity  -0.000259  0.000689 -0.002030  0.002025
korea_equity     -0.000188  0.000789 -0.001970  0.002077
short_treasury    0.000022  0.000823 -0.001917  0.002072
treasury_7_10y   -0.000023  0.000783 -0.001875  0.002019

## 2. Σ(t) 조건수 (full 7×7 covariance, per-date)

조건수 중간값이 너무 크면 ODE 최적화에서 수치 불안정 가능성. 일반적으로 < 10^3 권장.

                        model  n_usable_dates        mean      median          p95          max
       cnn_2d_residual_images            3521 4568.436513 2715.368396 13140.800096 26766.918156
       cnn_2d_rendered_images            3521 4568.436513 2715.368396 13140.800096 26766.918156
           cnn_1d_image_scale            3521 4568.436513 2715.368396 13140.800096 26766.918156
   cnn_1d_dilated_image_scale            3521 4568.436513 2715.368396 13140.800096 26766.918156
      cnn_1d_cumulative_scale            3521 4568.436513 2715.368396 13140.800096 26766.918156
                ensemble_top3            3521 4568.436513 2715.368396 13140.800096 26766.918156
 cnn_1d_attention_image_scale            3521 4568.436513 2715.368396 13140.800096 26766.918156
cnn_1d_multiscale_image_scale            3521 4568.436513 2715.368396 13140.800096 26766.918156

## 3. risk_score 분포

교차단면 z-score 기반. 날짜별 std가 0 근처면 신호 degenerate.

- **cnn_2d_residual_images**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_2d_rendered_images**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_1d_image_scale**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_1d_dilated_image_scale**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_1d_cumulative_scale**: 날짜별 cross-sectional std 평균 = 1.0801
- **ensemble_top3**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_1d_attention_image_scale**: 날짜별 cross-sectional std 평균 = 1.0801
- **cnn_1d_multiscale_image_scale**: 날짜별 cross-sectional std 평균 = 1.0801

## 4. ode_bundle NaN 커버리지

                        model  total_rows  full_rows first_full_date
       cnn_2d_residual_images        6617          0            None
       cnn_2d_rendered_images        6617          0            None
           cnn_1d_image_scale        6617          0            None
   cnn_1d_dilated_image_scale        6617          0            None
      cnn_1d_cumulative_scale        6617          0            None
                ensemble_top3        6617          0            None
 cnn_1d_attention_image_scale        6617          0            None
cnn_1d_multiscale_image_scale        6617          0            None

## 5. 앙상블 구성원 간 μ 상관

앙상블 멤버 (3개, OOS rank corr 상위): cnn_1d_dilated_image_scale, cnn_1d_cumulative_scale, cnn_1d_attention_image_scale

멤버 간 mu_hat_daily Pearson 상관 매트릭스.
상관이 지나치게 높으면 앙상블 효과 미미, 낮으면 diversification 효과 큼.

                              cnn_1d_dilated_image_scale  cnn_1d_cumulative_scale  cnn_1d_attention_image_scale
cnn_1d_dilated_image_scale                        1.0000                   0.1000                        0.6037
cnn_1d_cumulative_scale                           0.1000                   1.0000                        0.1205
cnn_1d_attention_image_scale                      0.6037                   0.1205                        1.0000

