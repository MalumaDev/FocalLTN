import os

for n_examples in [1500, 15000]:
    for seed in range(5):  # range(5):
        seed += 1300
        # Stable Product p=2
        os.system(
            f"python stable_product.py --csv-path results/{n_examples}/stable_product_p2_1_{seed}.csv"
            f" --n-examples-train {n_examples} --p 2 --seed {seed} --imbalance 1")

        # Stable Product p=6
        os.system(
            f"python stable_product.py --csv-path results/{n_examples}/stable_product_p6_1_{seed}.csv"
            f" --n-examples-train {n_examples} --p 6 --seed {seed} --imbalance 1")

        # Stable Product p=2 with Focal Loss

        # os.system(
        #     f"python stable_product.py --csv-path results/{n_examples}/stable_product_p6_{j}_{seed}.csv"
        #     f" --n-examples-train {n_examples} --p 6 --seed {seed} --imbalance {j}")

        # Log-Product
        os.system(
            f"python log_ltn.py --csv-path results/{n_examples}/logltn_default_1_{seed}.csv"
            f" --n-examples-train {n_examples} --seed {seed} --imbalance 1")

        # Log-Product with Focal Loss
        for aggr in ["mean","sum"]:
            os.system(
                f"python stable_product.py --csv-path results/{n_examples}/stable_product_p2_1_{seed}_focal_{aggr}.csv"
                f" --n-examples-train {n_examples} --p 2 --seed {seed} --imbalance 1 --use_focal --reduce_type {aggr}")

            os.system(
                f"python log_ltn.py --csv-path results/{n_examples}/logltn_default_1_{seed}_focal_mean_{aggr}.csv"
                f" --n-examples-train {n_examples} --seed {seed} --imbalance 1 --use_focal --reduce_type {aggr}")
