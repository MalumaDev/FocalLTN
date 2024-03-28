import os

for seed in range(5):  # range(5):
    for k in [1, -1]:
        for j in [1., 0.75, .5, .25]:
            if k == -1 and j == 1.:
                continue
            # Stable Product p=2
            os.system(
                f"python script-stableproduct-ltn.py --csv-path results/stable_product_p2_{j * k}_{seed}.csv"
                f" --p 2 --seed {seed} --imbalance {j * k}")

            # Stable Product p=6
            os.system(
                f"python script-stableproduct-ltn.py --csv-path results/stable_product_p6_{j * k}_{seed}.csv"
                f" --p 6 --seed {seed} --imbalance {j * k}")

            # Stable Product p=2 with Focal Loss
            os.system(
                f"python script-stableproduct-ltn.py --csv-path results/stable_product_p2_{j * k}_{seed}_focal.csv"
                f" --p 2 --seed {seed} --imbalance {j * k} --use_focal")

            # os.system(
            #     f"python stable_product.py --csv-path results/{n_examples}/stable_product_p6_{j}_{seed}.csv"
            #     f" --n-examples-train {n_examples} --p 6 --seed {seed} --imbalance {j}")

            # Log-Product
            os.system(
                f"python script-focal-ltn.py --csv-path results/logltn_default_{j * k}_{seed}.csv"
                f" --seed {seed} --imbalance {j * k}")

            # Log-Product with Focal Loss
            os.system(
                f"python script-focal-ltn.py --csv-path results/logltn_default_{j * k}_{seed}_focal.csv"
                f" --seed {seed} --imbalance {j * k} --use_focal")
