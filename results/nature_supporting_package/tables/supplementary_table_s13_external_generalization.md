# Supplementary Table S13. External no-retuning and matched-GT transfer

Source: `results/nature_manuscript_tables/table_6_external_generalization.csv`

| setting                              |   datasets |   spots | primary_metric              | value                           | evidence                                                                                             |
|:-------------------------------------|-----------:|--------:|:----------------------------|:--------------------------------|:-----------------------------------------------------------------------------------------------------|
| external_no_retuning                 |         10 |   30631 | mean_expression_log1p_rmse  | 0.5874                          | results/nature_external_no_retuning/external_no_retuning_summary.csv                                 |
| Rep1_no_retuning_matched_GT          |          1 |    1146 | WaveST-Gate JSD / rank      | 0.2599 / 2                      | results/nature_external_matched_gt/external_matched_gt_summary.csv                                   |
| Rep1_minimal_retuning_matched_GT     |          1 |     171 | WaveST-Gate test JSD / rank | 0.03853 / 1                     | results/nature_external_matched_gt/external_matched_gt_summary.csv                                   |
| minimal_retuning_multisample_summary |          2 |     nan | top method mean JSD +/- SD  | WaveST-Gate: 0.02573 +/- 0.0181 | results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv |
