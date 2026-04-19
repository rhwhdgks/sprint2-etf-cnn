# Ensemble search — top-20 per aggregation mode

## Mode: raw (sorted by OOS rank corr)

|   k | members                                                                                                     |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                                      |      0.0524 |         0.3265 |      0.5785 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small         |      0.0514 |         0.3289 |      0.5878 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small             |      0.0513 |         0.2608 |      0.3927 |      0.5833 |
|   3 | logistic_image_scale + cnn_2d_residual_images + cnn_2d_residual_small                                       |      0.0511 |         0.2514 |      0.3675 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images + cnn_2d_residual_small          |      0.0502 |         0.2778 |      0.4367 |      0.5556 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small                                   |      0.0490 |         0.2651 |      0.4052 |      0.5694 |
|   4 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small     |      0.0489 |         0.2760 |      0.4242 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_2d_residual_images + cnn_2d_residual_small        |      0.0487 |         0.3763 |      0.7151 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale |      0.0480 |         0.3760 |      0.7563 |      0.5833 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                 |      0.0478 |         0.3291 |      0.5924 |      0.5972 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                           |      0.0470 |         0.3325 |      0.5959 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images                                  |      0.0469 |         0.3647 |      0.7042 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_images  |      0.0468 |         0.3427 |      0.6254 |      0.5694 |
|   2 | logistic_image_scale + cnn_2d_residual_small                                                                |      0.0465 |         0.3444 |      0.6353 |      0.6042 |
|   4 | cnn_1d_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small               |      0.0465 |         0.2418 |      0.3312 |      0.5972 |
|   4 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images + cnn_2d_residual_small       |      0.0458 |         0.3354 |      0.6092 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_2d_residual_images + cnn_2d_residual_small                  |      0.0456 |         0.2711 |      0.4086 |      0.5625 |
|   2 | logistic_image_scale + cnn_1d_dilated_image_scale                                                           |      0.0456 |         0.3877 |      0.8002 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images        |      0.0453 |         0.3026 |      0.5075 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images             |      0.0451 |         0.2709 |      0.4086 |      0.5556 |

## Mode: raw (sorted by top-k Sharpe)

|   k | members                                                                                                             |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:--------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   1 | cnn_1d_cumulative_scale                                                                                             |      0.0271 |         0.5214 |      0.9977 |      0.5903 |
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                                      |      0.0434 |         0.4939 |      0.9576 |      0.5903 |
|   4 | cnn_1d_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale      |      0.0261 |         0.4527 |      1.0148 |      0.5833 |
|   2 | logistic_image_scale + cnn_1d_multiscale_image_scale                                                                |      0.0336 |         0.4246 |      0.7928 |      0.5833 |
|   2 | cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale                                                           |      0.0275 |         0.4127 |      0.8506 |      0.6111 |
|   4 | cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale  |      0.0230 |         0.4101 |      0.8472 |      0.6042 |
|   3 | cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale                                  |      0.0184 |         0.4092 |      0.8295 |      0.6111 |
|   4 | cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale        |      0.0232 |         0.4037 |      0.8171 |      0.5764 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                           |      0.0253 |         0.3883 |      0.7665 |      0.5972 |
|   2 | logistic_image_scale + cnn_1d_dilated_image_scale                                                                   |      0.0456 |         0.3877 |      0.8002 |      0.5903 |
|   3 | cnn_1d_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale                                      |      0.0266 |         0.3863 |      0.7561 |      0.5764 |
|   4 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale |      0.0288 |         0.3860 |      0.7703 |      0.5833 |
|   1 | logistic_image_scale                                                                                                |      0.0392 |         0.3850 |      0.6569 |      0.5903 |
|   2 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                                                |      0.0359 |         0.3850 |      0.7723 |      0.5903 |
|   3 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale                                             |      0.0416 |         0.3830 |      0.7399 |      0.5556 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale    |      0.0382 |         0.3778 |      0.7647 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_2d_residual_images + cnn_2d_residual_small                |      0.0487 |         0.3763 |      0.7151 |      0.5833 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale         |      0.0480 |         0.3760 |      0.7563 |      0.5833 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                 |      0.0320 |         0.3742 |      0.7201 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                                      |      0.0326 |         0.3698 |      0.6539 |      0.5764 |

## Mode: rank (sorted by OOS rank corr)

|   k | members                                                                                                   |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:----------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                                    |      0.0614 |         0.6302 |      1.4609 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small       |      0.0582 |         0.3940 |      0.7966 |      0.5625 |
|   2 | logistic_image_scale + cnn_2d_residual_small                                                              |      0.0570 |         0.3985 |      0.7963 |      0.6319 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small     |      0.0564 |         0.4006 |      0.8004 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small                                 |      0.0524 |         0.2471 |      0.3489 |      0.5625 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_2d_residual_small                               |      0.0518 |         0.2524 |      0.3473 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small           |      0.0504 |         0.4006 |      0.7676 |      0.5486 |
|   2 | cnn_1d_cumulative_scale + cnn_2d_residual_small                                                           |      0.0484 |         0.2261 |      0.3046 |      0.5833 |
|   2 | cnn_1d_attention_image_scale + cnn_2d_residual_small                                                      |      0.0475 |         0.3265 |      0.5665 |      0.5764 |
|   3 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small                              |      0.0470 |         0.3582 |      0.6764 |      0.5694 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                            |      0.0469 |         0.3359 |      0.6070 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small    |      0.0467 |         0.4420 |      0.9633 |      0.5833 |
|   3 | logistic_image_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small                              |      0.0464 |         0.3629 |      0.6673 |      0.5903 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale + cnn_2d_residual_small           |      0.0462 |         0.4300 |      0.8882 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small  |      0.0461 |         0.2914 |      0.4649 |      0.5556 |
|   2 | cnn_1d_dilated_image_scale + cnn_2d_residual_small                                                        |      0.0459 |         0.3515 |      0.6481 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small               |      0.0455 |         0.3195 |      0.5321 |      0.5486 |
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                            |      0.0451 |         0.4301 |      0.8001 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small |      0.0447 |         0.4244 |      0.8912 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                               |      0.0439 |         0.3893 |      0.8142 |      0.5764 |

## Mode: rank (sorted by top-k Sharpe)

|   k | members                                                                                                            |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small                                             |      0.0614 |         0.6302 |      1.4609 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                                     |      0.0348 |         0.5660 |      1.2744 |      0.5903 |
|   2 | logistic_image_scale + cnn_1d_attention_image_scale                                                                |      0.0334 |         0.5564 |      1.1311 |      0.6111 |
|   1 | cnn_1d_cumulative_scale                                                                                            |      0.0271 |         0.5214 |      0.9977 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale                                      |      0.0417 |         0.5032 |      1.0533 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small             |      0.0467 |         0.4420 |      0.9633 |      0.5833 |
|   2 | logistic_image_scale + cnn_1d_multiscale_image_scale                                                               |      0.0310 |         0.4403 |      0.9610 |      0.5972 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_multiscale_image_scale + cnn_2d_residual_small              |      0.0346 |         0.4320 |      0.9083 |      0.5903 |
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                                     |      0.0451 |         0.4301 |      0.8001 |      0.5903 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale + cnn_2d_residual_small                    |      0.0462 |         0.4300 |      0.8882 |      0.5625 |
|   4 | cnn_2d_rendered_images + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small           |      0.0273 |         0.4269 |      0.9062 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_small          |      0.0447 |         0.4244 |      0.8912 |      0.5764 |
|   4 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small            |      0.0369 |         0.4011 |      0.7630 |      0.5417 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images + cnn_2d_residual_small                    |      0.0504 |         0.4006 |      0.7676 |      0.5486 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_small              |      0.0564 |         0.4006 |      0.8004 |      0.5764 |
|   2 | logistic_image_scale + cnn_2d_residual_small                                                                       |      0.0570 |         0.3985 |      0.7963 |      0.6319 |
|   4 | cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale |      0.0236 |         0.3946 |      0.7688 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_small                |      0.0582 |         0.3940 |      0.7966 |      0.5625 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                        |      0.0439 |         0.3893 |      0.8142 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                                  |      0.0363 |         0.3890 |      0.7894 |      0.5903 |
