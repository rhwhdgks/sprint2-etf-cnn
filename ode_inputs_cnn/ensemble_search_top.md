# Ensemble search — top-20 per aggregation mode

## Mode: raw (sorted by OOS rank corr)

|   k | members                                                                                                     |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale |      0.0480 |         0.3760 |      0.7563 |      0.5833 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                 |      0.0478 |         0.3291 |      0.5924 |      0.5972 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                           |      0.0470 |         0.3325 |      0.5959 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images                                  |      0.0469 |         0.3647 |      0.7042 |      0.5764 |
|   4 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale + cnn_2d_residual_images  |      0.0468 |         0.3427 |      0.6254 |      0.5694 |
|   2 | logistic_image_scale + cnn_1d_dilated_image_scale                                                           |      0.0456 |         0.3877 |      0.8002 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images        |      0.0453 |         0.3026 |      0.5075 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images             |      0.0451 |         0.2709 |      0.4086 |      0.5556 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale            |      0.0441 |         0.2916 |      0.4735 |      0.5625 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale                                      |      0.0440 |         0.3106 |      0.5304 |      0.5764 |
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                              |      0.0434 |         0.4939 |      0.9576 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale      |      0.0425 |         0.3248 |      0.5679 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images   |      0.0418 |         0.2854 |      0.4535 |      0.5764 |
|   3 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale                                     |      0.0416 |         0.3830 |      0.7399 |      0.5556 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images                                     |      0.0415 |         0.2543 |      0.3676 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images      |      0.0412 |         0.3398 |      0.5917 |      0.5694 |
|   4 | cnn_1d_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images          |      0.0408 |         0.2334 |      0.3126 |      0.5556 |
|   3 | cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images                               |      0.0404 |         0.2223 |      0.2838 |      0.5486 |
|   2 | logistic_image_scale + cnn_2d_rendered_images                                                               |      0.0401 |         0.3597 |      0.6861 |      0.5694 |
|   1 | logistic_image_scale                                                                                        |      0.0392 |         0.3850 |      0.6569 |      0.5903 |

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
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale         |      0.0480 |         0.3760 |      0.7563 |      0.5833 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                 |      0.0320 |         0.3742 |      0.7201 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                                      |      0.0326 |         0.3698 |      0.6539 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_multiscale_image_scale                                 |      0.0296 |         0.3685 |      0.6818 |      0.5903 |

## Mode: rank (sorted by OOS rank corr)

|   k | members                                                                                                       |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:--------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                                |      0.0451 |         0.4301 |      0.8001 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                   |      0.0439 |         0.3893 |      0.8142 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale                                 |      0.0417 |         0.5032 |      1.0533 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale   |      0.0406 |         0.3057 |      0.5171 |      0.5764 |
|   1 | logistic_image_scale                                                                                          |      0.0392 |         0.3850 |      0.6569 |      0.5903 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale    |      0.0389 |         0.3760 |      0.7606 |      0.5694 |
|   2 | logistic_image_scale + cnn_1d_dilated_image_scale                                                             |      0.0383 |         0.3777 |      0.7536 |      0.5972 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale        |      0.0373 |         0.3608 |      0.6915 |      0.5625 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale          |      0.0366 |         0.3784 |      0.7618 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images          |      0.0363 |         0.3543 |      0.6679 |      0.5833 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                             |      0.0363 |         0.3890 |      0.7894 |      0.5903 |
|   3 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                           |      0.0352 |         0.2283 |      0.2936 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                                |      0.0348 |         0.5660 |      1.2744 |      0.5903 |
|   2 | cnn_1d_attention_image_scale + cnn_1d_cumulative_scale                                                        |      0.0339 |         0.2718 |      0.4164 |      0.5833 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images                                       |      0.0339 |         0.3835 |      0.7338 |      0.5625 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale |      0.0337 |         0.3480 |      0.6568 |      0.5833 |
|   2 | logistic_image_scale + cnn_1d_attention_image_scale                                                           |      0.0334 |         0.5564 |      1.1311 |      0.6111 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images        |      0.0328 |         0.3782 |      0.7467 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale                              |      0.0321 |         0.2649 |      0.3878 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_image_scale + cnn_1d_cumulative_scale                                           |      0.0320 |         0.2395 |      0.3304 |      0.5486 |

## Mode: rank (sorted by top-k Sharpe)

|   k | members                                                                                                            |   rank_corr |   top_k_sharpe |   top_k_cum |   top_k_hit |
|----:|:-------------------------------------------------------------------------------------------------------------------|------------:|---------------:|------------:|------------:|
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_multiscale_image_scale                                     |      0.0348 |         0.5660 |      1.2744 |      0.5903 |
|   2 | logistic_image_scale + cnn_1d_attention_image_scale                                                                |      0.0334 |         0.5564 |      1.1311 |      0.6111 |
|   1 | cnn_1d_cumulative_scale                                                                                            |      0.0271 |         0.5214 |      0.9977 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale                                      |      0.0417 |         0.5032 |      1.0533 |      0.5694 |
|   2 | logistic_image_scale + cnn_1d_multiscale_image_scale                                                               |      0.0310 |         0.4403 |      0.9610 |      0.5972 |
|   2 | logistic_image_scale + cnn_1d_cumulative_scale                                                                     |      0.0451 |         0.4301 |      0.8001 |      0.5903 |
|   4 | cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale |      0.0236 |         0.3946 |      0.7688 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale                                        |      0.0439 |         0.3893 |      0.8142 |      0.5764 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale                                  |      0.0363 |         0.3890 |      0.7894 |      0.5903 |
|   1 | logistic_image_scale                                                                                               |      0.0392 |         0.3850 |      0.6569 |      0.5903 |
|   3 | logistic_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images                                            |      0.0339 |         0.3835 |      0.7338 |      0.5625 |
|   4 | logistic_image_scale + logistic_cumulative_scale + cnn_1d_dilated_image_scale + cnn_1d_multiscale_image_scale      |      0.0287 |         0.3789 |      0.7900 |      0.5764 |
|   4 | logistic_image_scale + cnn_2d_rendered_images + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale               |      0.0366 |         0.3784 |      0.7618 |      0.5694 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_2d_residual_images             |      0.0328 |         0.3782 |      0.7467 |      0.5764 |
|   2 | logistic_image_scale + cnn_1d_dilated_image_scale                                                                  |      0.0383 |         0.3777 |      0.7536 |      0.5972 |
|   4 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale + cnn_1d_dilated_image_scale         |      0.0389 |         0.3760 |      0.7606 |      0.5694 |
|   4 | logistic_cumulative_scale + cnn_2d_rendered_images + cnn_1d_attention_image_scale + cnn_1d_cumulative_scale        |      0.0143 |         0.3732 |      0.7306 |      0.5694 |
|   3 | logistic_image_scale + cnn_1d_dilated_image_scale + cnn_2d_residual_images                                         |      0.0276 |         0.3710 |      0.6943 |      0.5625 |
|   3 | logistic_image_scale + cnn_1d_attention_image_scale + cnn_1d_multiscale_image_scale                                |      0.0291 |         0.3659 |      0.7098 |      0.5972 |
|   4 | logistic_image_scale + logistic_cumulative_scale + cnn_2d_rendered_images + cnn_1d_multiscale_image_scale          |      0.0052 |         0.3634 |      0.7200 |      0.5972 |
