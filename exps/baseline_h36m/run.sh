# Baseline 48
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py --seed 888 --exp-name baseline.txt --layer-norm-axis spatial --with-normalization --num 48

