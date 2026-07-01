# Supplementary Table S19. Compute cost, hardware and software versions

Source: `environment_report + baseline runtime`

| method                                 |   runtime_seconds |   peak_cuda_memory_mb | device                  | source              |
|:---------------------------------------|------------------:|----------------------:|:------------------------|:--------------------|
| WaveST-Gate                            |      nan          |              nan      | cuda                    | baseline_comparison |
| BayesPrism                             |       53.5089     |                0      | R/cpu                   | baseline_comparison |
| RCTD (multi)                           |      234.711      |                0      | R/cpu                   | baseline_comparison |
| SpatialDWLS/Seurat                     |     1659.74       |                0      | R/Giotto/cpu            | baseline_comparison |
| CARD                                   |      365.538      |                0      | R/cpu                   | baseline_comparison |
| reference_cosine                       |        0.0922611  |              nan      | cuda                    | baseline_comparison |
| reference_nnls                         |        2.18175    |              nan      | cuda                    | baseline_comparison |
| SpatialDWLS                            |       21.7828     |                0      | R/cpu                   | baseline_comparison |
| Tangram (rna_count_based, 1000 epochs) |        6.80472    |               65.1851 | cuda                    | baseline_comparison |
| cell2location                          |      958.99       |               42.8848 | cuda                    | baseline_comparison |
| Tangram (uniform, 1000 epochs)         |        6.35717    |               65.1851 | cuda                    | baseline_comparison |
| uniform                                |        0.00929947 |              nan      | cuda                    | baseline_comparison |
| SPOTlight                              |      128.195      |                0      | R/cpu                   | baseline_comparison |
| hardware                               |                   |            24107      | NVIDIA GeForce RTX 4090 | environment_report  |
