EXPERIMENT 7 | extraction: cuda | heads: CPU | seeds: [42, 43, 44]
Torch: 2.14.0+cu126 | torchvision: 0.29.0+cu126 | PennyLane: 0.45.1
PASS pca_classical: forward, backward, frozen base, checkpoint round trip
PASS pca_quantum: forward, backward, frozen base, checkpoint round trip
PASS task_classical: forward, backward, frozen base, checkpoint round trip
PASS task_quantum: forward, backward, frozen base, checkpoint round trip
PASS task_frozen: forward, backward, frozen base, checkpoint round trip
PASS task_no_entanglement: forward, backward, frozen base, checkpoint round trip
PASS batched circuit, all circuit gradients, identical encoders, and zero-residual initialization
Strict JPEG validation: 100%
 6723/6723 [01:44<00:00, 61.61it/s]
JPEG audit: {'intact': 6665, 'recoverable_truncated': 58}
train 4830 images; 1231 patients
val 970 images; 264 patients
test 923 images; 264 patients
Reusing features: train (4830, 1024)
Reusing features: val (970, 1024)
Feature cache: 22.7 MiB; loading/extraction 0.0s
Frozen DenseNet feature extraction: 100%
 1/1 [00:00<00:00,  2.89it/s]
Saved E2 checkpoint metrics: {'macro_AUROC': 0.7303071009041724, 'macro_AUPRC': 0.2930734458959053}
Reproduced FP32 E2 metrics: {'macro_AUROC': 0.7307837861407264, 'macro_AUPRC': 0.29433588632475105, 'macro_AUROC_50plus': 0.7711530780574897, 'macro_AUPRC_50plus': 0.399016943438275}
PCA variance retained by 48 PCs: 0.7503564357757568
All residual arms start from the same selected linear anchor per seed.
linear_control seed=42 01/40 | AUROC 0.73231 AP 0.29483 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 02/40 | AUROC 0.73372 AP 0.29569 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 03/40 | AUROC 0.73527 AP 0.29644 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 04/40 | AUROC 0.73630 AP 0.29676 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 05/40 | AUROC 0.73723 AP 0.29838 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 06/40 | AUROC 0.73806 AP 0.29852 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 07/40 | AUROC 0.73851 AP 0.29874 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 08/40 | AUROC 0.73947 AP 0.29895 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 09/40 | AUROC 0.73990 AP 0.29886 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 10/40 | AUROC 0.74034 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 11/40 | AUROC 0.74062 AP 0.29937 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 12/40 | AUROC 0.74112 AP 0.29913 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 13/40 | AUROC 0.74130 AP 0.29933 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 14/40 | AUROC 0.74185 AP 0.29954 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 15/40 | AUROC 0.74216 AP 0.29975 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 16/40 | AUROC 0.74237 AP 0.29986 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 17/40 | AUROC 0.74257 AP 0.29989 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 18/40 | AUROC 0.74265 AP 0.29937 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 19/40 | AUROC 0.74291 AP 0.29948 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 20/40 | AUROC 0.74286 AP 0.29996 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 21/40 | AUROC 0.74301 AP 0.30003 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 22/40 | AUROC 0.74331 AP 0.30019 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 23/40 | AUROC 0.74320 AP 0.30037 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 24/40 | AUROC 0.74329 AP 0.30056 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 25/40 | AUROC 0.74314 AP 0.30061 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 26/40 | AUROC 0.74311 AP 0.30061 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 27/40 | AUROC 0.74307 AP 0.30057 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 28/40 | AUROC 0.74313 AP 0.30048 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 29/40 | AUROC 0.74325 AP 0.30065 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 30/40 | AUROC 0.74310 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 31/40 | AUROC 0.74308 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 32/40 | AUROC 0.74302 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 33/40 | AUROC 0.74314 AP 0.30073 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 34/40 | AUROC 0.74310 AP 0.30067 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 35/40 | AUROC 0.74307 AP 0.30072 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 36/40 | AUROC 0.74313 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=42 37/40 | AUROC 0.74297 AP 0.30084 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 38/40 | AUROC 0.74302 AP 0.30089 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 39/40 | AUROC 0.74303 AP 0.30097 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=42 40/40 | AUROC 0.74300 AP 0.30103 | residual 0.000 | core grad 0 | 0.1s | best
pca_classical seed=42 01/50 | AUROC 0.74289 AP 0.30115 | residual 0.013 | core grad 0 | 0.1s | best
pca_classical seed=42 02/50 | AUROC 0.74299 AP 0.30162 | residual 0.023 | core grad 0 | 0.1s | best
pca_classical seed=42 03/50 | AUROC 0.74293 AP 0.30159 | residual 0.032 | core grad 0 | 0.1s
pca_classical seed=42 04/50 | AUROC 0.74220 AP 0.30151 | residual 0.175 | core grad 0.0282 | 0.1s
pca_classical seed=42 05/50 | AUROC 0.74058 AP 0.30115 | residual 0.156 | core grad 0.043 | 0.1s
pca_classical seed=42 06/50 | AUROC 0.73911 AP 0.30087 | residual 0.218 | core grad 0.0649 | 0.1s
pca_classical seed=42 07/50 | AUROC 0.73795 AP 0.30001 | residual 0.270 | core grad 0.062 | 0.1s
pca_classical seed=42 08/50 | AUROC 0.73637 AP 0.29865 | residual 0.288 | core grad 0.084 | 0.1s
pca_classical seed=42 09/50 | AUROC 0.73627 AP 0.29859 | residual 0.323 | core grad 0.0915 | 0.1s
pca_classical seed=42 10/50 | AUROC 0.73561 AP 0.29862 | residual 0.321 | core grad 0.106 | 0.1s
pca_classical seed=42 11/50 | AUROC 0.73519 AP 0.29863 | residual 0.325 | core grad 0.078 | 0.1s
pca_classical seed=42 12/50 | AUROC 0.73515 AP 0.29869 | residual 0.335 | core grad 0.0894 | 0.1s
pca_classical seed=42 13/50 | AUROC 0.73531 AP 0.29894 | residual 0.360 | core grad 0.0851 | 0.1s
pca_classical seed=42 14/50 | AUROC 0.73515 AP 0.29933 | residual 0.353 | core grad 0.0948 | 0.1s
pca_classical seed=42 15/50 | AUROC 0.73475 AP 0.29925 | residual 0.350 | core grad 0.0961 | 0.1s
pca_quantum seed=42 01/50 | AUROC 0.74295 AP 0.30113 | residual 0.009 | core grad 0 | 2.3s | best
pca_quantum seed=42 02/50 | AUROC 0.74293 AP 0.30132 | residual 0.016 | core grad 0 | 2.3s | best
pca_quantum seed=42 03/50 | AUROC 0.74269 AP 0.30102 | residual 0.023 | core grad 0 | 2.2s
pca_quantum seed=42 04/50 | AUROC 0.74249 AP 0.30089 | residual 0.029 | core grad 0.00182 | 4.8s
pca_quantum seed=42 05/50 | AUROC 0.74250 AP 0.30081 | residual 0.036 | core grad 0.00183 | 4.8s
pca_quantum seed=42 06/50 | AUROC 0.74239 AP 0.30080 | residual 0.043 | core grad 0.00264 | 4.8s
pca_quantum seed=42 07/50 | AUROC 0.74243 AP 0.30054 | residual 0.049 | core grad 0.00245 | 4.8s
pca_quantum seed=42 08/50 | AUROC 0.74228 AP 0.30051 | residual 0.055 | core grad 0.00312 | 4.8s
pca_quantum seed=42 09/50 | AUROC 0.74210 AP 0.30036 | residual 0.060 | core grad 0.00297 | 4.8s
pca_quantum seed=42 10/50 | AUROC 0.74193 AP 0.30034 | residual 0.063 | core grad 0.0038 | 4.8s
pca_quantum seed=42 11/50 | AUROC 0.74195 AP 0.30031 | residual 0.066 | core grad 0.00344 | 4.8s
pca_quantum seed=42 12/50 | AUROC 0.74188 AP 0.30026 | residual 0.069 | core grad 0.00355 | 4.8s
pca_quantum seed=42 13/50 | AUROC 0.74179 AP 0.30034 | residual 0.072 | core grad 0.00374 | 5.0s
pca_quantum seed=42 14/50 | AUROC 0.74175 AP 0.30010 | residual 0.075 | core grad 0.00419 | 4.8s
pca_quantum seed=42 15/50 | AUROC 0.74174 AP 0.30046 | residual 0.076 | core grad 0.0043 | 5.0s
task_classical seed=42 01/50 | AUROC 0.74253 AP 0.30054 | residual 0.014 | core grad 0 | 0.1s
task_classical seed=42 02/50 | AUROC 0.74189 AP 0.30035 | residual 0.026 | core grad 0 | 0.1s
task_classical seed=42 03/50 | AUROC 0.74145 AP 0.30004 | residual 0.037 | core grad 0 | 0.1s
task_classical seed=42 04/50 | AUROC 0.73922 AP 0.29903 | residual 0.198 | core grad 0.0334 | 0.1s
task_classical seed=42 05/50 | AUROC 0.73586 AP 0.29586 | residual 0.226 | core grad 0.0539 | 0.1s
task_classical seed=42 06/50 | AUROC 0.73723 AP 0.29577 | residual 0.275 | core grad 0.0769 | 0.1s
task_classical seed=42 07/50 | AUROC 0.73713 AP 0.29748 | residual 0.303 | core grad 0.0672 | 0.1s
task_classical seed=42 08/50 | AUROC 0.73773 AP 0.29799 | residual 0.319 | core grad 0.0946 | 0.1s
task_classical seed=42 09/50 | AUROC 0.73828 AP 0.29834 | residual 0.337 | core grad 0.0888 | 0.1s
task_classical seed=42 10/50 | AUROC 0.73794 AP 0.29924 | residual 0.345 | core grad 0.0962 | 0.1s
task_classical seed=42 11/50 | AUROC 0.73783 AP 0.29893 | residual 0.341 | core grad 0.0778 | 0.1s
task_classical seed=42 12/50 | AUROC 0.73798 AP 0.29943 | residual 0.351 | core grad 0.0864 | 0.1s
task_classical seed=42 13/50 | AUROC 0.73808 AP 0.29960 | residual 0.367 | core grad 0.088 | 0.1s
task_classical seed=42 14/50 | AUROC 0.73817 AP 0.29977 | residual 0.364 | core grad 0.09 | 0.1s
task_classical seed=42 15/50 | AUROC 0.73835 AP 0.30022 | residual 0.370 | core grad 0.0847 | 0.1s
task_quantum seed=42 01/50 | AUROC 0.74257 AP 0.30102 | residual 0.011 | core grad 0 | 2.3s
task_quantum seed=42 02/50 | AUROC 0.74214 AP 0.30069 | residual 0.021 | core grad 0 | 2.3s
task_quantum seed=42 03/50 | AUROC 0.74202 AP 0.30082 | residual 0.029 | core grad 0 | 2.3s
task_quantum seed=42 04/50 | AUROC 0.74174 AP 0.30068 | residual 0.037 | core grad 0.0021 | 5.0s
task_quantum seed=42 05/50 | AUROC 0.74155 AP 0.30075 | residual 0.045 | core grad 0.00234 | 5.1s
task_quantum seed=42 06/50 | AUROC 0.74126 AP 0.30052 | residual 0.053 | core grad 0.00291 | 5.1s
task_quantum seed=42 07/50 | AUROC 0.74093 AP 0.30043 | residual 0.060 | core grad 0.00326 | 5.1s
task_quantum seed=42 08/50 | AUROC 0.74064 AP 0.30020 | residual 0.067 | core grad 0.00347 | 5.1s
task_quantum seed=42 09/50 | AUROC 0.74072 AP 0.30027 | residual 0.073 | core grad 0.00369 | 5.1s
task_quantum seed=42 10/50 | AUROC 0.74082 AP 0.30019 | residual 0.080 | core grad 0.0044 | 5.2s
task_quantum seed=42 11/50 | AUROC 0.74067 AP 0.30024 | residual 0.084 | core grad 0.00406 | 5.2s
task_quantum seed=42 12/50 | AUROC 0.74075 AP 0.30012 | residual 0.087 | core grad 0.00473 | 5.2s
task_quantum seed=42 13/50 | AUROC 0.74070 AP 0.29997 | residual 0.090 | core grad 0.00461 | 5.4s
task_quantum seed=42 14/50 | AUROC 0.74065 AP 0.29994 | residual 0.093 | core grad 0.00547 | 5.2s
task_quantum seed=42 15/50 | AUROC 0.74069 AP 0.29969 | residual 0.097 | core grad 0.00474 | 5.2s
task_frozen seed=42 01/50 | AUROC 0.74257 AP 0.30102 | residual 0.011 | core grad 0 | 2.4s
task_frozen seed=42 02/50 | AUROC 0.74214 AP 0.30069 | residual 0.021 | core grad 0 | 2.4s
task_frozen seed=42 03/50 | AUROC 0.74202 AP 0.30082 | residual 0.029 | core grad 0 | 2.4s
task_frozen seed=42 04/50 | AUROC 0.74171 AP 0.30073 | residual 0.037 | core grad 0 | 4.4s
task_frozen seed=42 05/50 | AUROC 0.74139 AP 0.30071 | residual 0.045 | core grad 0 | 4.4s
task_frozen seed=42 06/50 | AUROC 0.74113 AP 0.30045 | residual 0.052 | core grad 0 | 4.4s
task_frozen seed=42 07/50 | AUROC 0.74076 AP 0.30067 | residual 0.059 | core grad 0 | 4.4s
task_frozen seed=42 08/50 | AUROC 0.74036 AP 0.30021 | residual 0.066 | core grad 0 | 4.4s
task_frozen seed=42 09/50 | AUROC 0.74027 AP 0.30021 | residual 0.072 | core grad 0 | 4.4s
task_frozen seed=42 10/50 | AUROC 0.74025 AP 0.29996 | residual 0.075 | core grad 0 | 4.4s
task_frozen seed=42 11/50 | AUROC 0.73999 AP 0.30006 | residual 0.078 | core grad 0 | 4.4s
task_frozen seed=42 12/50 | AUROC 0.74010 AP 0.30014 | residual 0.081 | core grad 0 | 4.4s
task_frozen seed=42 13/50 | AUROC 0.74002 AP 0.30008 | residual 0.084 | core grad 0 | 4.6s
task_frozen seed=42 14/50 | AUROC 0.73992 AP 0.30007 | residual 0.086 | core grad 0 | 4.4s
task_frozen seed=42 15/50 | AUROC 0.73986 AP 0.30008 | residual 0.088 | core grad 0 | 4.4s
task_no_entanglement seed=42 01/50 | AUROC 0.74268 AP 0.30100 | residual 0.011 | core grad 0 | 2.0s
task_no_entanglement seed=42 02/50 | AUROC 0.74245 AP 0.30092 | residual 0.020 | core grad 0 | 2.0s
task_no_entanglement seed=42 03/50 | AUROC 0.74228 AP 0.30078 | residual 0.029 | core grad 0 | 2.0s
task_no_entanglement seed=42 04/50 | AUROC 0.74209 AP 0.30048 | residual 0.037 | core grad 0.00191 | 4.3s
task_no_entanglement seed=42 05/50 | AUROC 0.74193 AP 0.30061 | residual 0.045 | core grad 0.00229 | 4.3s
task_no_entanglement seed=42 06/50 | AUROC 0.74180 AP 0.30069 | residual 0.052 | core grad 0.00257 | 4.3s
task_no_entanglement seed=42 07/50 | AUROC 0.74163 AP 0.30050 | residual 0.060 | core grad 0.00311 | 4.3s
task_no_entanglement seed=42 08/50 | AUROC 0.74152 AP 0.30015 | residual 0.067 | core grad 0.00315 | 4.3s
task_no_entanglement seed=42 09/50 | AUROC 0.74151 AP 0.30015 | residual 0.074 | core grad 0.0033 | 4.3s
task_no_entanglement seed=42 10/50 | AUROC 0.74171 AP 0.30023 | residual 0.082 | core grad 0.00427 | 4.3s
task_no_entanglement seed=42 11/50 | AUROC 0.74162 AP 0.30019 | residual 0.089 | core grad 0.00434 | 4.3s
task_no_entanglement seed=42 12/50 | AUROC 0.74176 AP 0.30004 | residual 0.093 | core grad 0.00437 | 4.3s
task_no_entanglement seed=42 13/50 | AUROC 0.74172 AP 0.29988 | residual 0.096 | core grad 0.00451 | 4.3s
task_no_entanglement seed=42 14/50 | AUROC 0.74175 AP 0.29984 | residual 0.100 | core grad 0.00493 | 4.3s
task_no_entanglement seed=42 15/50 | AUROC 0.74176 AP 0.29990 | residual 0.103 | core grad 0.00475 | 4.3s
linear_control seed=43 01/40 | AUROC 0.73243 AP 0.29498 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 02/40 | AUROC 0.73390 AP 0.29578 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 03/40 | AUROC 0.73514 AP 0.29619 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 04/40 | AUROC 0.73621 AP 0.29690 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 05/40 | AUROC 0.73748 AP 0.29852 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 06/40 | AUROC 0.73825 AP 0.29865 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 07/40 | AUROC 0.73874 AP 0.29890 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 08/40 | AUROC 0.73940 AP 0.29899 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 09/40 | AUROC 0.73988 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 10/40 | AUROC 0.74025 AP 0.29908 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 11/40 | AUROC 0.74088 AP 0.29845 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 12/40 | AUROC 0.74126 AP 0.29911 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 13/40 | AUROC 0.74152 AP 0.29946 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 14/40 | AUROC 0.74180 AP 0.29960 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 15/40 | AUROC 0.74219 AP 0.29979 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 16/40 | AUROC 0.74239 AP 0.29991 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 17/40 | AUROC 0.74272 AP 0.29987 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 18/40 | AUROC 0.74271 AP 0.29945 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 19/40 | AUROC 0.74298 AP 0.29963 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 20/40 | AUROC 0.74294 AP 0.30005 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 21/40 | AUROC 0.74293 AP 0.29992 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 22/40 | AUROC 0.74333 AP 0.30006 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 23/40 | AUROC 0.74332 AP 0.30038 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 24/40 | AUROC 0.74329 AP 0.30031 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 25/40 | AUROC 0.74321 AP 0.30050 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 26/40 | AUROC 0.74329 AP 0.30056 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 27/40 | AUROC 0.74317 AP 0.30045 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 28/40 | AUROC 0.74326 AP 0.30055 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 29/40 | AUROC 0.74323 AP 0.30060 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 30/40 | AUROC 0.74324 AP 0.30045 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 31/40 | AUROC 0.74320 AP 0.30054 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 32/40 | AUROC 0.74313 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 33/40 | AUROC 0.74305 AP 0.30078 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 34/40 | AUROC 0.74296 AP 0.30087 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 35/40 | AUROC 0.74289 AP 0.30084 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 36/40 | AUROC 0.74285 AP 0.30078 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 37/40 | AUROC 0.74291 AP 0.30090 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 38/40 | AUROC 0.74284 AP 0.30082 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=43 39/40 | AUROC 0.74289 AP 0.30099 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=43 40/40 | AUROC 0.74283 AP 0.30092 | residual 0.000 | core grad 0 | 0.1s
pca_classical seed=43 01/50 | AUROC 0.74227 AP 0.30041 | residual 0.016 | core grad 0 | 0.1s
pca_classical seed=43 02/50 | AUROC 0.74189 AP 0.29979 | residual 0.028 | core grad 0 | 0.1s
pca_classical seed=43 03/50 | AUROC 0.74138 AP 0.29948 | residual 0.040 | core grad 0 | 0.1s
pca_classical seed=43 04/50 | AUROC 0.74017 AP 0.29832 | residual 0.183 | core grad 0.0344 | 0.1s
pca_classical seed=43 05/50 | AUROC 0.73875 AP 0.29768 | residual 0.237 | core grad 0.039 | 0.1s
pca_classical seed=43 06/50 | AUROC 0.73594 AP 0.29647 | residual 0.221 | core grad 0.0608 | 0.1s
pca_classical seed=43 07/50 | AUROC 0.73563 AP 0.29546 | residual 0.321 | core grad 0.0656 | 0.1s
pca_classical seed=43 08/50 | AUROC 0.73529 AP 0.29500 | residual 0.285 | core grad 0.0908 | 0.1s
pca_classical seed=43 09/50 | AUROC 0.73493 AP 0.29482 | residual 0.312 | core grad 0.0782 | 0.1s
pca_classical seed=43 10/50 | AUROC 0.73491 AP 0.29559 | residual 0.321 | core grad 0.105 | 0.1s
pca_classical seed=43 11/50 | AUROC 0.73530 AP 0.29570 | residual 0.341 | core grad 0.0885 | 0.1s
pca_classical seed=43 12/50 | AUROC 0.73460 AP 0.29603 | residual 0.331 | core grad 0.0799 | 0.1s
pca_classical seed=43 13/50 | AUROC 0.73489 AP 0.29581 | residual 0.341 | core grad 0.0945 | 0.1s
pca_classical seed=43 14/50 | AUROC 0.73489 AP 0.29645 | residual 0.336 | core grad 0.0863 | 0.1s
pca_classical seed=43 15/50 | AUROC 0.73480 AP 0.29618 | residual 0.344 | core grad 0.0968 | 0.1s
pca_quantum seed=43 01/50 | AUROC 0.74274 AP 0.30073 | residual 0.009 | core grad 0 | 2.4s
pca_quantum seed=43 02/50 | AUROC 0.74249 AP 0.30070 | residual 0.016 | core grad 0 | 2.4s
pca_quantum seed=43 03/50 | AUROC 0.74237 AP 0.30064 | residual 0.022 | core grad 0 | 2.4s
pca_quantum seed=43 04/50 | AUROC 0.74214 AP 0.30033 | residual 0.029 | core grad 0.00143 | 5.2s
pca_quantum seed=43 05/50 | AUROC 0.74201 AP 0.30016 | residual 0.036 | core grad 0.00182 | 5.2s
pca_quantum seed=43 06/50 | AUROC 0.74183 AP 0.30000 | residual 0.042 | core grad 0.00232 | 5.2s
pca_quantum seed=43 07/50 | AUROC 0.74172 AP 0.29997 | residual 0.048 | core grad 0.00262 | 5.2s
pca_quantum seed=43 08/50 | AUROC 0.74181 AP 0.30012 | residual 0.053 | core grad 0.00402 | 5.2s
pca_quantum seed=43 09/50 | AUROC 0.74173 AP 0.30038 | residual 0.060 | core grad 0.00289 | 5.2s
pca_quantum seed=43 10/50 | AUROC 0.74156 AP 0.30030 | residual 0.065 | core grad 0.0035 | 5.2s
pca_quantum seed=43 11/50 | AUROC 0.74150 AP 0.30043 | residual 0.071 | core grad 0.00363 | 5.2s
pca_quantum seed=43 12/50 | AUROC 0.74132 AP 0.30023 | residual 0.077 | core grad 0.00408 | 5.2s
pca_quantum seed=43 13/50 | AUROC 0.74105 AP 0.30031 | residual 0.083 | core grad 0.00414 | 5.4s
pca_quantum seed=43 14/50 | AUROC 0.74098 AP 0.30020 | residual 0.089 | core grad 0.00475 | 5.2s
pca_quantum seed=43 15/50 | AUROC 0.74082 AP 0.30014 | residual 0.094 | core grad 0.00456 | 5.2s
task_classical seed=43 01/50 | AUROC 0.74239 AP 0.30065 | residual 0.015 | core grad 0 | 0.1s
task_classical seed=43 02/50 | AUROC 0.74166 AP 0.30062 | residual 0.028 | core grad 0 | 0.1s
task_classical seed=43 03/50 | AUROC 0.74141 AP 0.30052 | residual 0.038 | core grad 0 | 0.1s
task_classical seed=43 04/50 | AUROC 0.74017 AP 0.29934 | residual 0.199 | core grad 0.0316 | 0.1s
task_classical seed=43 05/50 | AUROC 0.73858 AP 0.29772 | residual 0.264 | core grad 0.0354 | 0.1s
task_classical seed=43 06/50 | AUROC 0.73618 AP 0.29667 | residual 0.251 | core grad 0.0615 | 0.1s
task_classical seed=43 07/50 | AUROC 0.73712 AP 0.29670 | residual 0.317 | core grad 0.0693 | 0.1s
task_classical seed=43 08/50 | AUROC 0.73822 AP 0.29805 | residual 0.322 | core grad 0.121 | 0.1s
task_classical seed=43 09/50 | AUROC 0.73756 AP 0.29777 | residual 0.323 | core grad 0.0783 | 0.1s
task_classical seed=43 10/50 | AUROC 0.73804 AP 0.29802 | residual 0.338 | core grad 0.129 | 0.1s
task_classical seed=43 11/50 | AUROC 0.73822 AP 0.29800 | residual 0.358 | core grad 0.112 | 0.1s
task_classical seed=43 12/50 | AUROC 0.73757 AP 0.29812 | residual 0.344 | core grad 0.097 | 0.1s
task_classical seed=43 13/50 | AUROC 0.73763 AP 0.29796 | residual 0.355 | core grad 0.0968 | 0.1s
task_classical seed=43 14/50 | AUROC 0.73765 AP 0.29811 | residual 0.354 | core grad 0.101 | 0.1s
task_classical seed=43 15/50 | AUROC 0.73762 AP 0.29827 | residual 0.358 | core grad 0.0891 | 0.1s
task_quantum seed=43 01/50 | AUROC 0.74263 AP 0.30065 | residual 0.011 | core grad 0 | 2.4s
task_quantum seed=43 02/50 | AUROC 0.74249 AP 0.30074 | residual 0.020 | core grad 0 | 2.4s
task_quantum seed=43 03/50 | AUROC 0.74222 AP 0.30085 | residual 0.029 | core grad 0 | 2.4s
task_quantum seed=43 04/50 | AUROC 0.74216 AP 0.30067 | residual 0.037 | core grad 0.00185 | 5.2s
task_quantum seed=43 05/50 | AUROC 0.74220 AP 0.30076 | residual 0.045 | core grad 0.00222 | 5.2s
task_quantum seed=43 06/50 | AUROC 0.74197 AP 0.30078 | residual 0.053 | core grad 0.0027 | 5.2s
task_quantum seed=43 07/50 | AUROC 0.74193 AP 0.30085 | residual 0.060 | core grad 0.00281 | 5.2s
task_quantum seed=43 08/50 | AUROC 0.74192 AP 0.30071 | residual 0.066 | core grad 0.00373 | 5.2s
task_quantum seed=43 09/50 | AUROC 0.74194 AP 0.30061 | residual 0.073 | core grad 0.00363 | 5.2s
task_quantum seed=43 10/50 | AUROC 0.74176 AP 0.30046 | residual 0.079 | core grad 0.00427 | 5.2s
task_quantum seed=43 11/50 | AUROC 0.74181 AP 0.30034 | residual 0.086 | core grad 0.00465 | 5.2s
task_quantum seed=43 12/50 | AUROC 0.74183 AP 0.30032 | residual 0.092 | core grad 0.00465 | 5.2s
task_quantum seed=43 13/50 | AUROC 0.74174 AP 0.30023 | residual 0.095 | core grad 0.00507 | 5.4s
task_quantum seed=43 14/50 | AUROC 0.74170 AP 0.30024 | residual 0.098 | core grad 0.00478 | 5.2s
task_quantum seed=43 15/50 | AUROC 0.74175 AP 0.30014 | residual 0.101 | core grad 0.00518 | 5.2s
task_frozen seed=43 01/50 | AUROC 0.74263 AP 0.30065 | residual 0.011 | core grad 0 | 2.4s
task_frozen seed=43 02/50 | AUROC 0.74249 AP 0.30074 | residual 0.020 | core grad 0 | 2.4s
task_frozen seed=43 03/50 | AUROC 0.74222 AP 0.30085 | residual 0.029 | core grad 0 | 2.4s
task_frozen seed=43 04/50 | AUROC 0.74215 AP 0.30068 | residual 0.037 | core grad 0 | 4.4s
task_frozen seed=43 05/50 | AUROC 0.74210 AP 0.30066 | residual 0.045 | core grad 0 | 4.4s
task_frozen seed=43 06/50 | AUROC 0.74189 AP 0.30089 | residual 0.053 | core grad 0 | 4.4s
task_frozen seed=43 07/50 | AUROC 0.74170 AP 0.30083 | residual 0.059 | core grad 0 | 4.4s
task_frozen seed=43 08/50 | AUROC 0.74154 AP 0.30086 | residual 0.065 | core grad 0 | 4.4s
task_frozen seed=43 09/50 | AUROC 0.74148 AP 0.30061 | residual 0.072 | core grad 0 | 4.4s
task_frozen seed=43 10/50 | AUROC 0.74121 AP 0.30058 | residual 0.078 | core grad 0 | 4.4s
task_frozen seed=43 11/50 | AUROC 0.74113 AP 0.30021 | residual 0.084 | core grad 0 | 4.4s
task_frozen seed=43 12/50 | AUROC 0.74108 AP 0.30017 | residual 0.086 | core grad 0 | 4.4s
task_frozen seed=43 13/50 | AUROC 0.74091 AP 0.30020 | residual 0.089 | core grad 0 | 4.6s
task_frozen seed=43 14/50 | AUROC 0.74088 AP 0.30025 | residual 0.091 | core grad 0 | 4.4s
task_frozen seed=43 15/50 | AUROC 0.74078 AP 0.30030 | residual 0.094 | core grad 0 | 4.4s
task_no_entanglement seed=43 01/50 | AUROC 0.74267 AP 0.30073 | residual 0.011 | core grad 0 | 2.1s
task_no_entanglement seed=43 02/50 | AUROC 0.74275 AP 0.30067 | residual 0.019 | core grad 0 | 2.1s
task_no_entanglement seed=43 03/50 | AUROC 0.74247 AP 0.30085 | residual 0.028 | core grad 0 | 2.1s
task_no_entanglement seed=43 04/50 | AUROC 0.74249 AP 0.30074 | residual 0.036 | core grad 0.0016 | 4.4s
task_no_entanglement seed=43 05/50 | AUROC 0.74244 AP 0.30060 | residual 0.044 | core grad 0.00186 | 4.4s
task_no_entanglement seed=43 06/50 | AUROC 0.74237 AP 0.30087 | residual 0.052 | core grad 0.00239 | 4.3s
task_no_entanglement seed=43 07/50 | AUROC 0.74242 AP 0.30088 | residual 0.059 | core grad 0.00263 | 4.4s
task_no_entanglement seed=43 08/50 | AUROC 0.74238 AP 0.30075 | residual 0.066 | core grad 0.00342 | 4.3s
task_no_entanglement seed=43 09/50 | AUROC 0.74233 AP 0.30060 | residual 0.073 | core grad 0.00331 | 4.3s
task_no_entanglement seed=43 10/50 | AUROC 0.74236 AP 0.30058 | residual 0.079 | core grad 0.00379 | 4.3s
task_no_entanglement seed=43 11/50 | AUROC 0.74238 AP 0.30055 | residual 0.086 | core grad 0.00411 | 4.3s
task_no_entanglement seed=43 12/50 | AUROC 0.74232 AP 0.30034 | residual 0.090 | core grad 0.00404 | 4.3s
task_no_entanglement seed=43 13/50 | AUROC 0.74232 AP 0.30033 | residual 0.093 | core grad 0.00471 | 4.3s
task_no_entanglement seed=43 14/50 | AUROC 0.74225 AP 0.30017 | residual 0.096 | core grad 0.00421 | 4.3s
task_no_entanglement seed=43 15/50 | AUROC 0.74220 AP 0.30016 | residual 0.100 | core grad 0.00434 | 4.3s
linear_control seed=44 01/40 | AUROC 0.73220 AP 0.29497 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 02/40 | AUROC 0.73370 AP 0.29591 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 03/40 | AUROC 0.73535 AP 0.29643 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 04/40 | AUROC 0.73600 AP 0.29809 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 05/40 | AUROC 0.73704 AP 0.29843 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 06/40 | AUROC 0.73828 AP 0.29854 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 07/40 | AUROC 0.73892 AP 0.29893 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 08/40 | AUROC 0.73936 AP 0.29888 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 09/40 | AUROC 0.73982 AP 0.29888 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 10/40 | AUROC 0.74043 AP 0.29887 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 11/40 | AUROC 0.74079 AP 0.29925 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 12/40 | AUROC 0.74113 AP 0.29909 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 13/40 | AUROC 0.74141 AP 0.29935 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 14/40 | AUROC 0.74202 AP 0.29952 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 15/40 | AUROC 0.74208 AP 0.29960 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 16/40 | AUROC 0.74232 AP 0.29974 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 17/40 | AUROC 0.74256 AP 0.29985 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 18/40 | AUROC 0.74263 AP 0.29994 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 19/40 | AUROC 0.74291 AP 0.29982 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 20/40 | AUROC 0.74290 AP 0.29984 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 21/40 | AUROC 0.74308 AP 0.29994 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 22/40 | AUROC 0.74335 AP 0.30008 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 23/40 | AUROC 0.74331 AP 0.30037 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 24/40 | AUROC 0.74344 AP 0.30032 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 25/40 | AUROC 0.74330 AP 0.30055 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 26/40 | AUROC 0.74322 AP 0.30046 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 27/40 | AUROC 0.74345 AP 0.30046 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 28/40 | AUROC 0.74337 AP 0.30064 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 29/40 | AUROC 0.74313 AP 0.30032 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 30/40 | AUROC 0.74314 AP 0.30062 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 31/40 | AUROC 0.74317 AP 0.30052 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 32/40 | AUROC 0.74300 AP 0.30047 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 33/40 | AUROC 0.74310 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 34/40 | AUROC 0.74301 AP 0.30079 | residual 0.000 | core grad 0 | 0.1s | best
linear_control seed=44 35/40 | AUROC 0.74283 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 36/40 | AUROC 0.74294 AP 0.30071 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 37/40 | AUROC 0.74300 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 38/40 | AUROC 0.74295 AP 0.30069 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 39/40 | AUROC 0.74295 AP 0.30072 | residual 0.000 | core grad 0 | 0.1s
linear_control seed=44 40/40 | AUROC 0.74289 AP 0.30066 | residual 0.000 | core grad 0 | 0.1s
pca_classical seed=44 01/50 | AUROC 0.74284 AP 0.30082 | residual 0.015 | core grad 0 | 0.1s | best
pca_classical seed=44 02/50 | AUROC 0.74257 AP 0.30068 | residual 0.027 | core grad 0 | 0.1s
pca_classical seed=44 03/50 | AUROC 0.74252 AP 0.30090 | residual 0.037 | core grad 0 | 0.1s | best
pca_classical seed=44 04/50 | AUROC 0.74136 AP 0.30051 | residual 0.179 | core grad 0.0282 | 0.1s
pca_classical seed=44 05/50 | AUROC 0.74022 AP 0.29983 | residual 0.185 | core grad 0.0371 | 0.1s
pca_classical seed=44 06/50 | AUROC 0.73850 AP 0.29885 | residual 0.232 | core grad 0.0573 | 0.1s
pca_classical seed=44 07/50 | AUROC 0.73872 AP 0.29913 | residual 0.290 | core grad 0.0648 | 0.1s
pca_classical seed=44 08/50 | AUROC 0.73919 AP 0.29842 | residual 0.328 | core grad 0.0801 | 0.1s
pca_classical seed=44 09/50 | AUROC 0.73919 AP 0.29966 | residual 0.338 | core grad 0.093 | 0.1s
pca_classical seed=44 10/50 | AUROC 0.73993 AP 0.29988 | residual 0.345 | core grad 0.0881 | 0.1s
pca_classical seed=44 11/50 | AUROC 0.73961 AP 0.30013 | residual 0.360 | core grad 0.0942 | 0.1s
pca_classical seed=44 12/50 | AUROC 0.73951 AP 0.30035 | residual 0.373 | core grad 0.104 | 0.1s
pca_classical seed=44 13/50 | AUROC 0.73891 AP 0.30014 | residual 0.364 | core grad 0.0901 | 0.1s
pca_classical seed=44 14/50 | AUROC 0.73929 AP 0.30076 | residual 0.378 | core grad 0.0927 | 0.1s
pca_classical seed=44 15/50 | AUROC 0.73823 AP 0.30098 | residual 0.366 | core grad 0.0972 | 0.1s | best
pca_quantum seed=44 01/50 | AUROC 0.74289 AP 0.30077 | residual 0.010 | core grad 0 | 2.4s
pca_quantum seed=44 02/50 | AUROC 0.74269 AP 0.30078 | residual 0.016 | core grad 0 | 2.4s
pca_quantum seed=44 03/50 | AUROC 0.74262 AP 0.30084 | residual 0.023 | core grad 0 | 2.4s | best
pca_quantum seed=44 04/50 | AUROC 0.74256 AP 0.30089 | residual 0.030 | core grad 0.00165 | 5.1s | best
pca_quantum seed=44 05/50 | AUROC 0.74239 AP 0.30065 | residual 0.036 | core grad 0.00211 | 5.1s
pca_quantum seed=44 06/50 | AUROC 0.74213 AP 0.30042 | residual 0.042 | core grad 0.00266 | 5.1s
pca_quantum seed=44 07/50 | AUROC 0.74210 AP 0.30034 | residual 0.048 | core grad 0.00238 | 5.2s
pca_quantum seed=44 08/50 | AUROC 0.74199 AP 0.30023 | residual 0.054 | core grad 0.0025 | 5.1s
pca_quantum seed=44 09/50 | AUROC 0.74182 AP 0.30024 | residual 0.060 | core grad 0.00315 | 5.1s
pca_quantum seed=44 10/50 | AUROC 0.74181 AP 0.30014 | residual 0.062 | core grad 0.00345 | 5.1s
pca_quantum seed=44 11/50 | AUROC 0.74181 AP 0.30017 | residual 0.065 | core grad 0.00372 | 5.1s
pca_quantum seed=44 12/50 | AUROC 0.74181 AP 0.30027 | residual 0.068 | core grad 0.00355 | 5.1s
pca_quantum seed=44 13/50 | AUROC 0.74179 AP 0.30014 | residual 0.070 | core grad 0.00362 | 5.3s
pca_quantum seed=44 14/50 | AUROC 0.74183 AP 0.30026 | residual 0.073 | core grad 0.00433 | 5.2s
pca_quantum seed=44 15/50 | AUROC 0.74166 AP 0.30022 | residual 0.074 | core grad 0.0043 | 5.1s
task_classical seed=44 01/50 | AUROC 0.74263 AP 0.30066 | residual 0.015 | core grad 0 | 0.1s
task_classical seed=44 02/50 | AUROC 0.74203 AP 0.30051 | residual 0.026 | core grad 0 | 0.1s
task_classical seed=44 03/50 | AUROC 0.74173 AP 0.30042 | residual 0.037 | core grad 0 | 0.1s
task_classical seed=44 04/50 | AUROC 0.73937 AP 0.29946 | residual 0.195 | core grad 0.0324 | 0.1s
task_classical seed=44 05/50 | AUROC 0.73738 AP 0.29843 | residual 0.222 | core grad 0.0494 | 0.1s
task_classical seed=44 06/50 | AUROC 0.73746 AP 0.29716 | residual 0.280 | core grad 0.0605 | 0.1s
task_classical seed=44 07/50 | AUROC 0.73709 AP 0.29777 | residual 0.308 | core grad 0.0848 | 0.1s
task_classical seed=44 08/50 | AUROC 0.73765 AP 0.29744 | residual 0.332 | core grad 0.077 | 0.1s
task_classical seed=44 09/50 | AUROC 0.73772 AP 0.29824 | residual 0.358 | core grad 0.115 | 0.1s
task_classical seed=44 10/50 | AUROC 0.73834 AP 0.29785 | residual 0.358 | core grad 0.0879 | 0.1s
task_classical seed=44 11/50 | AUROC 0.73669 AP 0.29801 | residual 0.354 | core grad 0.0997 | 0.1s
task_classical seed=44 12/50 | AUROC 0.73762 AP 0.29817 | residual 0.374 | core grad 0.1 | 0.1s
task_classical seed=44 13/50 | AUROC 0.73736 AP 0.29830 | residual 0.359 | core grad 0.0886 | 0.1s
task_classical seed=44 14/50 | AUROC 0.73772 AP 0.29795 | residual 0.373 | core grad 0.106 | 0.1s
task_classical seed=44 15/50 | AUROC 0.73749 AP 0.29841 | residual 0.366 | core grad 0.091 | 0.1s
task_quantum seed=44 01/50 | AUROC 0.74279 AP 0.30072 | residual 0.011 | core grad 0 | 2.4s
task_quantum seed=44 02/50 | AUROC 0.74268 AP 0.30086 | residual 0.020 | core grad 0 | 2.4s | best
task_quantum seed=44 03/50 | AUROC 0.74242 AP 0.30071 | residual 0.029 | core grad 0 | 2.4s
task_quantum seed=44 04/50 | AUROC 0.74236 AP 0.30074 | residual 0.037 | core grad 0.00195 | 5.2s
task_quantum seed=44 05/50 | AUROC 0.74228 AP 0.30080 | residual 0.045 | core grad 0.00227 | 5.2s
task_quantum seed=44 06/50 | AUROC 0.74215 AP 0.30067 | residual 0.052 | core grad 0.00313 | 5.2s
task_quantum seed=44 07/50 | AUROC 0.74209 AP 0.30060 | residual 0.059 | core grad 0.00307 | 5.2s
task_quantum seed=44 08/50 | AUROC 0.74214 AP 0.30053 | residual 0.066 | core grad 0.0039 | 5.2s
task_quantum seed=44 09/50 | AUROC 0.74189 AP 0.30030 | residual 0.072 | core grad 0.00373 | 5.2s
task_quantum seed=44 10/50 | AUROC 0.74193 AP 0.30022 | residual 0.079 | core grad 0.00402 | 5.2s
task_quantum seed=44 11/50 | AUROC 0.74191 AP 0.30035 | residual 0.082 | core grad 0.0044 | 5.2s
task_quantum seed=44 12/50 | AUROC 0.74209 AP 0.30021 | residual 0.085 | core grad 0.00466 | 5.1s
task_quantum seed=44 13/50 | AUROC 0.74208 AP 0.29998 | residual 0.088 | core grad 0.00487 | 5.4s
task_quantum seed=44 14/50 | AUROC 0.74216 AP 0.29962 | residual 0.091 | core grad 0.00427 | 5.2s
task_quantum seed=44 15/50 | AUROC 0.74213 AP 0.29951 | residual 0.094 | core grad 0.00522 | 5.0s
task_frozen seed=44 01/50 | AUROC 0.74279 AP 0.30072 | residual 0.011 | core grad 0 | 2.3s
task_frozen seed=44 02/50 | AUROC 0.74268 AP 0.30086 | residual 0.020 | core grad 0 | 2.3s | best
task_frozen seed=44 03/50 | AUROC 0.74242 AP 0.30071 | residual 0.029 | core grad 0 | 2.3s
task_frozen seed=44 04/50 | AUROC 0.74231 AP 0.30075 | residual 0.037 | core grad 0 | 4.3s
task_frozen seed=44 05/50 | AUROC 0.74217 AP 0.30080 | residual 0.045 | core grad 0 | 4.2s
task_frozen seed=44 06/50 | AUROC 0.74193 AP 0.30079 | residual 0.052 | core grad 0 | 4.2s
task_frozen seed=44 07/50 | AUROC 0.74183 AP 0.30084 | residual 0.059 | core grad 0 | 4.2s
task_frozen seed=44 08/50 | AUROC 0.74165 AP 0.30059 | residual 0.065 | core grad 0 | 4.2s
task_frozen seed=44 09/50 | AUROC 0.74138 AP 0.30031 | residual 0.071 | core grad 0 | 4.2s
task_frozen seed=44 10/50 | AUROC 0.74123 AP 0.30045 | residual 0.078 | core grad 0 | 4.2s
task_frozen seed=44 11/50 | AUROC 0.74127 AP 0.30060 | residual 0.083 | core grad 0 | 4.2s
task_frozen seed=44 12/50 | AUROC 0.74134 AP 0.30055 | residual 0.089 | core grad 0 | 4.2s
task_frozen seed=44 13/50 | AUROC 0.74127 AP 0.30041 | residual 0.091 | core grad 0 | 4.5s
task_frozen seed=44 14/50 | AUROC 0.74112 AP 0.30031 | residual 0.094 | core grad 0 | 4.2s
task_frozen seed=44 15/50 | AUROC 0.74104 AP 0.30004 | residual 0.096 | core grad 0 | 4.2s
task_no_entanglement seed=44 01/50 | AUROC 0.74292 AP 0.30083 | residual 0.011 | core grad 0 | 2.0s | best
task_no_entanglement seed=44 02/50 | AUROC 0.74284 AP 0.30105 | residual 0.020 | core grad 0 | 2.0s | best
task_no_entanglement seed=44 03/50 | AUROC 0.74288 AP 0.30093 | residual 0.028 | core grad 0 | 2.0s
task_no_entanglement seed=44 04/50 | AUROC 0.74275 AP 0.30097 | residual 0.036 | core grad 0.00168 | 4.2s
task_no_entanglement seed=44 05/50 | AUROC 0.74262 AP 0.30106 | residual 0.044 | core grad 0.00213 | 4.2s | best
task_no_entanglement seed=44 06/50 | AUROC 0.74263 AP 0.30092 | residual 0.051 | core grad 0.00274 | 4.2s
task_no_entanglement seed=44 07/50 | AUROC 0.74261 AP 0.30085 | residual 0.058 | core grad 0.00279 | 4.2s
task_no_entanglement seed=44 08/50 | AUROC 0.74253 AP 0.30061 | residual 0.064 | core grad 0.00361 | 4.2s
task_no_entanglement seed=44 09/50 | AUROC 0.74261 AP 0.30051 | residual 0.071 | core grad 0.0033 | 4.2s
task_no_entanglement seed=44 10/50 | AUROC 0.74282 AP 0.30036 | residual 0.078 | core grad 0.00413 | 4.2s
task_no_entanglement seed=44 11/50 | AUROC 0.74278 AP 0.30028 | residual 0.081 | core grad 0.00456 | 4.2s
task_no_entanglement seed=44 12/50 | AUROC 0.74293 AP 0.30027 | residual 0.084 | core grad 0.00459 | 4.2s
task_no_entanglement seed=44 13/50 | AUROC 0.74297 AP 0.30032 | residual 0.088 | core grad 0.00429 | 4.2s
task_no_entanglement seed=44 14/50 | AUROC 0.74296 AP 0.30024 | residual 0.090 | core grad 0.00388 | 4.2s
task_no_entanglement seed=44 15/50 | AUROC 0.74301 AP 0.30010 | residual 0.093 | core grad 0.00462 | 4.2s

                    arm     seed  checkpoint_epoch  validation_macro_AUROC  validation_macro_AUPRC
experiment_2_reproduced   source               5.0                0.730784                0.294336
         linear_control       42              40.0                0.743003                0.301028
          pca_classical       42               2.0                0.742987                0.301622
            pca_quantum       42               2.0                0.742929                0.301321
         task_classical       42               0.0                0.743003                0.301028
           task_quantum       42               0.0                0.743003                0.301028
            task_frozen       42               0.0                0.743003                0.301028
   task_no_entanglement       42               0.0                0.743003                0.301028
         linear_control       43              39.0                0.742888                0.300988
          pca_classical       43               0.0                0.742888                0.300988
            pca_quantum       43               0.0                0.742888                0.300988
         task_classical       43               0.0                0.742888                0.300988
           task_quantum       43               0.0                0.742888                0.300988
            task_frozen       43               0.0                0.742888                0.300988
   task_no_entanglement       43               0.0                0.742888                0.300988
         linear_control       44              34.0                0.743007                0.300792
          pca_classical       44              15.0                0.738226                0.300976
            pca_quantum       44               4.0                0.742560                0.300890
         task_classical       44               0.0                0.743007                0.300792
           task_quantum       44               2.0                0.742678                0.300865
            task_frozen       44               2.0                0.742678                0.300865
   task_no_entanglement       44               5.0                0.742622                0.301057
         linear_control ensemble               NaN                0.742902                0.300829
          pca_classical ensemble               NaN                0.741813                0.300939
            pca_quantum ensemble               NaN                0.742614                0.301304
         task_classical ensemble               NaN                0.742902                0.300829
           task_quantum ensemble               NaN                0.742818                0.300961
            task_frozen ensemble               NaN                0.742818                0.300961
   task_no_entanglement ensemble               NaN                0.742779                0.301052

Paired macro AUPRC differences (positive favors left):
    seed                                comparison      metric  point_delta  ci_lower  ci_upper  confidence_level  bootstrap_samples_requested  bootstrap_samples_valid
      42           pca_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42  task_no_entanglement minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           pca_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43  task_no_entanglement minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           pca_quantum minus circuit_reset macro_AUPRC     0.000003 -0.000082  0.000075              0.95                          500                      493
      44          task_quantum minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           task_frozen minus circuit_reset macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44  task_no_entanglement minus circuit_reset macro_AUPRC    -0.000044 -0.000284  0.000179              0.95                          500                      493
      42           pca_quantum minus pca_classical macro_AUPRC    -0.000301 -0.001276  0.001062              0.95                          500                      493
      42         task_quantum minus task_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42            task_quantum minus pca_quantum macro_AUPRC    -0.000293 -0.000823  0.000518              0.95                          500                      493
      42            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42   task_quantum minus task_no_entanglement macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42        pca_classical minus linear_control macro_AUPRC     0.000594 -0.000566  0.001344              0.95                          500                      493
      42          pca_quantum minus linear_control macro_AUPRC     0.000293 -0.000518  0.000823              0.95                          500                      493
      42       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42         task_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42          task_frozen minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      42 task_no_entanglement minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43           pca_quantum minus pca_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43         task_quantum minus task_classical macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43            task_quantum minus pca_quantum macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43   task_quantum minus task_no_entanglement macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43        pca_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          pca_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43         task_quantum minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43          task_frozen minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      43 task_no_entanglement minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44           pca_quantum minus pca_classical macro_AUPRC    -0.000086 -0.010148  0.009225              0.95                          500                      493
      44         task_quantum minus task_classical macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44            task_quantum minus pca_quantum macro_AUPRC    -0.000025 -0.001049  0.001818              0.95                          500                      493
      44            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44   task_quantum minus task_no_entanglement macro_AUPRC    -0.000192 -0.002065  0.001202              0.95                          500                      493
      44        pca_classical minus linear_control macro_AUPRC     0.000184 -0.009434  0.010946              0.95                          500                      493
      44          pca_quantum minus linear_control macro_AUPRC     0.000098 -0.001311  0.001112              0.95                          500                      493
      44       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
      44         task_quantum minus linear_control macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44          task_frozen minus linear_control macro_AUPRC     0.000073 -0.000731  0.001669              0.95                          500                      493
      44 task_no_entanglement minus linear_control macro_AUPRC     0.000265 -0.001570  0.003165              0.95                          500                      493
ensemble           pca_quantum minus pca_classical macro_AUPRC     0.000366 -0.004877  0.004775              0.95                          500                      493
ensemble         task_quantum minus task_classical macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble            task_quantum minus pca_quantum macro_AUPRC    -0.000344 -0.001289  0.000372              0.95                          500                      493
ensemble            task_quantum minus task_frozen macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
ensemble   task_quantum minus task_no_entanglement macro_AUPRC    -0.000091 -0.000692  0.000518              0.95                          500                      493
ensemble        pca_classical minus linear_control macro_AUPRC     0.000110 -0.004074  0.005510              0.95                          500                      493
ensemble          pca_quantum minus linear_control macro_AUPRC     0.000475 -0.000203  0.001496              0.95                          500                      493
ensemble       task_classical minus linear_control macro_AUPRC     0.000000  0.000000  0.000000              0.95                          500                      493
ensemble         task_quantum minus linear_control macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble          task_frozen minus linear_control macro_AUPRC     0.000132 -0.000271  0.000691              0.95                          500                      493
ensemble task_no_entanglement minus linear_control macro_AUPRC     0.000223 -0.000495  0.001020              0.95                          500                      493
Saved to: /workspace/unzippedarchive/unzippedarchive/xray_training_experiment_7_encoding/run_20260929T182317_419531Z
Test features, predictions and metrics were not computed.