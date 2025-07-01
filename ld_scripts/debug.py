--- Time-dependent AUC and C-index at Day 365 ---
Evaluating models at Day 365:  20%|██        | 1/5 [00:00<00:00,  6.86it/s]

Model: DeepSurv_LSTM_365DayFutureTarget_detailed_outputs
  AUC@365: 0.9544
  C-index: 0.9528
  Confusion Matrix: {'True Negative': 40849, 'False Positive': 133, 'False Negative': 4, 'True Positive': 15}
Evaluating models at Day 365:  40%|████      | 2/5 [00:00<00:00,  6.75it/s]

Model: DeepSurv_MLP_365DayFutureTarget_detailed_outputs
  AUC@365: 0.9482
  C-index: 0.9460
  Confusion Matrix: {'True Negative': 40982, 'False Positive': 0, 'False Negative': 19, 'True Positive': 0}
Evaluating models at Day 365:  60%|██████    | 3/5 [00:00<00:00,  6.98it/s]

Model: DeepSurv_RNN_365DayFutureTarget_detailed_outputs
  AUC@365: 0.9603
  C-index: 0.9595
  Confusion Matrix: {'True Negative': 40716, 'False Positive': 266, 'False Negative': 4, 'True Positive': 15}
Evaluating models at Day 365:  80%|████████  | 4/5 [00:00<00:00,  6.99it/s]

Model: DeepSurv_TCN_365DayFutureTarget_detailed_outputs
  AUC@365: 0.9224
  C-index: 0.9200
  Confusion Matrix: {'True Negative': 39742, 'False Positive': 1240, 'False Negative': 4, 'True Positive': 15}
Evaluating models at Day 365: 100%|██████████| 5/5 [00:00<00:00,  6.98it/s]

Model: DeepSurv_Transformer_365DayFutureTarget_detailed_outputs
  AUC@365: 0.9398
  C-index: 0.9374
  Confusion Matrix: {'True Negative': 40834, 'False Positive': 148, 'False Negative': 4, 'True Positive': 15}

--- Time-dependent AUC and C-index at Day 365 ---
Model: tab_DeepSurv_RNN_Tabular_365DayFutureTarget_deepsurv_outputs
  AUC@365: 0.5901
  C-index: 0.5930
  Confusion Matrix: {'True Negative': 76584, 'False Positive': 1003, 'False Negative': 171, 'True Positive': 20}
Evaluating models at Day 365:  40%|████      | 2/5 [00:00<00:00,  3.13it/s]

Model: tab_DeepSurv_LSTM_Tabular_365DayFutureTarget_deepsurv_outputs
  AUC@365: 0.6890
  C-index: 0.6890
  Confusion Matrix: {'True Negative': 74016, 'False Positive': 3571, 'False Negative': 119, 'True Positive': 72}
Evaluating models at Day 365:  60%|██████    | 3/5 [00:00<00:00,  3.03it/s]

Model: tab_DeepSurv_Transformer_Tabular_365DayFutureTarget_deepsurv_outputs
  AUC@365: 0.4687
  C-index: 0.4699
  Confusion Matrix: {'True Negative': 0, 'False Positive': 77587, 'False Negative': 0, 'True Positive': 191}
Evaluating models at Day 365:  80%|████████  | 4/5 [00:01<00:00,  3.06it/s]

Model: tab_DeepSurv_MLP_Tabular_365DayFutureTarget_deepsurv_outputs
  AUC@365: 0.5303
  C-index: 0.5333
  Confusion Matrix: {'True Negative': 77501, 'False Positive': 86, 'False Negative': 191, 'True Positive': 0}
Evaluating models at Day 365: 100%|██████████| 5/5 [00:01<00:00,  3.05it/s]

Model: tab_DeepSurv_TCN_Tabular_365DayFutureTarget_deepsurv_outputs
  AUC@365: 0.5893
  C-index: 0.5901
  Confusion Matrix: {'True Negative': 60311, 'False Positive': 17276, 'False Negative': 105, 'True Positive': 86}

--- regular AUC embedding 365 ---

Processing file: LSTM_365DayFutureTarget_detailed_outputs
Optimal threshold selected: 0.020
Metrics: 
AUROC 0.5831188173443728
AUPRC 0.15811289129461867
F1 0.16819265019440613
PPV 0.09185350929266177
Recall 0.9957920792079208
Avg Precision 0.15811289129461867
Avg Recall 0.9957920792079208
Confusion Matrix {'True Negative': 1002, 'False Positive': 39775, 'False Negative': 17, 'True Positive': 4023}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5832 [0.5729, 0.5933]
  AUPRC: 0.1582 [0.1494, 0.1673]
  F1: 0.1682 [0.1641, 0.1726]
  PPV: 0.0919 [0.0894, 0.0945]
  Recall: 0.9958 [0.9938, 0.9977]
  Avg Precision: 0.1582 [0.1494, 0.1673]
  Avg Recall: 0.9958 [0.9938, 0.9977]
  Confusion Matrix: 11204.2500 [80.0000, 39890.0000]

Processing file: MLP_365DayFutureTarget_detailed_outputs
Optimal threshold selected: 0.030
Metrics: 
AUROC 0.6337862333576222
AUPRC 0.17790583651839315
F1 0.16572999138532224
PPV 0.09035201502885003
Recall 1.0
Avg Precision 0.17790583651839315
Avg Recall 1.0
Confusion Matrix {'True Negative': 103, 'False Positive': 40674, 'False Negative': 0, 'True Positive': 4040}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.6337 [0.6238, 0.6430]
  AUPRC: 0.1783 [0.1686, 0.1888]
  F1: 0.1656 [0.1612, 0.1703]
  PPV: 0.0903 [0.0877, 0.0931]
  Recall: 1.0000 [1.0000, 1.0000]
  Avg Precision: 0.1783 [0.1686, 0.1888]
  Avg Recall: 1.0000 [1.0000, 1.0000]
  Confusion Matrix: 11204.2500 [6.0000, 40696.0000]

Processing file: RNN_365DayFutureTarget_detailed_outputs
Optimal threshold selected: 0.020
Metrics: 
AUROC 0.5741269345440074
AUPRC 0.11598568368200475
F1 0.17691789392626817
PPV 0.09798485979315492
Recall 0.9099009900990099
Avg Precision 0.11598568368200475
Avg Recall 0.9099009900990099
Confusion Matrix {'True Negative': 6937, 'False Positive': 33840, 'False Negative': 364, 'True Positive': 3676}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5742 [0.5656, 0.5830]
  AUPRC: 0.1162 [0.1104, 0.1217]
  F1: 0.1769 [0.1719, 0.1818]
  PPV: 0.0980 [0.0949, 0.1009]
  Recall: 0.9100 [0.9012, 0.9185]
  Avg Precision: 0.1162 [0.1104, 0.1217]
  Avg Recall: 0.9100 [0.9012, 0.9185]
  Confusion Matrix: 11204.2500 [629.0000, 34231.0250]

Processing file: TCN_365DayFutureTarget_detailed_outputs
Optimal threshold selected: 0.010
Metrics: 
AUROC 0.637951680924769
AUPRC 0.1397297039216007
F1 0.2060855996433348
PPV 0.13298331415420023
Recall 0.4576732673267327
Avg Precision 0.1397297039216007
Avg Recall 0.4576732673267327
Confusion Matrix {'True Negative': 28722, 'False Positive': 12055, 'False Negative': 2191, 'True Positive': 1849}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.6377 [0.6287, 0.6463]
  AUPRC: 0.1399 [0.1330, 0.1466]
  F1: 0.2061 [0.1977, 0.2139]
  PPV: 0.1330 [0.1269, 0.1387]
  Recall: 0.4577 [0.4428, 0.4728]
  Avg Precision: 0.1399 [0.1330, 0.1466]
  Avg Recall: 0.4577 [0.4428, 0.4728]
  Confusion Matrix: 11204.2500 [1217.0000, 28248.0000]

Processing file: Transformer_365DayFutureTarget_detailed_outputs
Optimal threshold selected: 0.030
Metrics: 
AUROC 0.616232098661714
AUPRC 0.153438556309844
F1 0.17925001392990472
PPV 0.1009920261191687
Recall 0.7962871287128713
Avg Precision 0.153438556309844
Avg Recall 0.7962871287128713
Confusion Matrix {'True Negative': 12140, 'False Positive': 28637, 'False Negative': 823, 'True Positive': 3217}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.6160 [0.6059, 0.6258]
  AUPRC: 0.1537 [0.1456, 0.1621]
  F1: 0.1791 [0.1739, 0.1843]
  PPV: 0.1009 [0.0977, 0.1041]
  Recall: 0.7960 [0.7838, 0.8091]
  Avg Precision: 0.1537 [0.1456, 0.1621]
  Avg Recall: 0.7960 [0.7838, 0.8091]
  Confusion Matrix: 11204.2500 [1130.0000, 29103.0000]

--- regular AUC tabular 365 ---
Processing file: tab_RNN_Tabular_365DayFutureTarget_classification_outputs
Optimal threshold selected: 0.010
Metrics: 
AUROC 0.5798407607840741
AUPRC 0.10020514077142184
F1 0.1683511575978376
PPV 0.10652537646402677
Recall 0.4012043131214116
Avg Precision 0.10020514077142184
Avg Recall 0.4012043131214116
Confusion Matrix {'True Negative': 53728, 'False Positive': 24030, 'False Negative': 4276, 'True Positive': 2865}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5799 [0.5731, 0.5865]
  AUPRC: 0.1003 [0.0972, 0.1033]
  F1: 0.1684 [0.1632, 0.1738]
  PPV: 0.1066 [0.1030, 0.1102]
  Recall: 0.4013 [0.3893, 0.4133]
  Avg Precision: 0.1003 [0.0972, 0.1033]
  Avg Recall: 0.4013 [0.3893, 0.4133]
  Confusion Matrix: 21224.7500 [2213.9750, 53289.0750]

Processing file: tab_LSTM_Tabular_365DayFutureTarget_classification_outputs
Optimal threshold selected: 0.010
Metrics: 
AUROC 0.5743288779658962
AUPRC 0.09797876632789886
F1 0.17212623589333867
PPV 0.10473701801829176
Recall 0.482705503430892
Avg Precision 0.09797876632789886
Avg Recall 0.482705503430892
Confusion Matrix {'True Negative': 48294, 'False Positive': 29464, 'False Negative': 3694, 'True Positive': 3447}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5743 [0.5681, 0.5802]
  AUPRC: 0.0981 [0.0950, 0.1011]
  F1: 0.1722 [0.1670, 0.1769]
  PPV: 0.1048 [0.1014, 0.1080]
  Recall: 0.4826 [0.4715, 0.4935]
  Avg Precision: 0.0981 [0.0950, 0.1011]
  Avg Recall: 0.4826 [0.4715, 0.4935]
  Confusion Matrix: 21224.7500 [2716.0000, 47784.0500]

Processing file: tab_Transformer_Tabular_365DayFutureTarget_classification_outputs
Optimal threshold selected: 0.000
Metrics: 
AUROC 0.505567912329651
AUPRC 0.08990272285279899
F1 0.1551716644936984
PPV 0.08411170920741116
Recall 1.0
Avg Precision 0.08990272285279899
Avg Recall 1.0
Confusion Matrix {'True Negative': 0, 'False Positive': 77758, 'False Negative': 0, 'True Positive': 7141}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5056 [0.4979, 0.5124]
  AUPRC: 0.0900 [0.0870, 0.0932]
  F1: 0.1552 [0.1523, 0.1585]
  PPV: 0.0841 [0.0824, 0.0861]
  Recall: 1.0000 [1.0000, 1.0000]
  Avg Precision: 0.0900 [0.0870, 0.0932]
  Avg Recall: 1.0000 [1.0000, 1.0000]
  Confusion Matrix: 21224.7500 [0.0000, 77758.0000]

Processing file: tab_MLP_Tabular_365DayFutureTarget_classification_outputs
Optimal threshold selected: 0.010
Metrics: 
AUROC 0.5519617849682816
AUPRC 0.09306675122616816
F1 0.16678731721442017
PPV 0.0978450786255096
Recall 0.5646268029687719
Avg Precision 0.09306675122616816
Avg Recall 0.5646268029687719
Confusion Matrix {'True Negative': 40582, 'False Positive': 37176, 'False Negative': 3109, 'True Positive': 4032}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5520 [0.5452, 0.5586]
  AUPRC: 0.0931 [0.0902, 0.0958]
  F1: 0.1667 [0.1621, 0.1712]
  PPV: 0.0978 [0.0949, 0.1006]
  Recall: 0.5645 [0.5528, 0.5771]
  Avg Precision: 0.0931 [0.0902, 0.0958]
  Avg Recall: 0.5645 [0.5528, 0.5771]
  Confusion Matrix: 21224.7500 [3409.0000, 40195.0000]

Processing file: tab_TCN_Tabular_365DayFutureTarget_classification_outputs
Optimal threshold selected: 0.010
Metrics: 
AUROC 0.5510166769752276
AUPRC 0.0952280053473488
F1 0.1618723189502902
PPV 0.09523691698626748
Recall 0.5390001400364095
Avg Precision 0.0952280053473488
Avg Recall 0.5390001400364095
Confusion Matrix {'True Negative': 41192, 'False Positive': 36566, 'False Negative': 3292, 'True Positive': 3849}
Bootstrapping with 1000 iterations using 20 workers...
Bootstrapping completed.
Bootstrapped Metrics with 95% Confidence Intervals:
  AUROC: 0.5511 [0.5448, 0.5570]
  AUPRC: 0.0953 [0.0922, 0.0983]
  F1: 0.1620 [0.1577, 0.1663]
  PPV: 0.0953 [0.0926, 0.0980]
  Recall: 0.5392 [0.5278, 0.5499]
  Avg Precision: 0.0953 [0.0922, 0.0983]
  Avg Recall: 0.5392 [0.5278, 0.5499]
  Confusion Matrix: 21224.7500 [3343.0000, 40933.0250]