import os

methods = {
    'stable_rl': [2, 6],
    'prod_rl': [6],
    'log_ltn': [5],
    'focal_ltn': [1, 2, 6],
    'focal_log_ltn': [1, 2, 6]
}
for seed in range(5):  # range(5):
    for method in methods:
        for p_value in methods[method]:
            os.system(
                f"python train.py --stable-config {method} --seed {seed} --p_value {p_value}")
