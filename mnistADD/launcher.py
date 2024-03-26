import os

n_examples = 1500
for seed in range(5):
    for j in [1., 0.75, .5, .25]:
        os.system(
            f"python stable_product.py --csv-path results/{n_examples}/stable_product_p2_{j}_{seed}.csv"
            f" --n-examples-train {n_examples} --p 2 --seed {seed} --imbalance {j}")
        # os.system(
        #     f"python stable_product.py --csv-path results/{n_examples}/stable_product_p6_{j}_{seed}.csv"
        #     f" --n-examples-train {n_examples} --p 6 --seed {seed} --imbalance {j}")

        os.system(
            f"python log_ltn.py --csv-path results/{n_examples}/logltn_default_{j}_{seed}.csv"
            f" --n-examples-train {n_examples} --seed {seed} --imbalance {j}")

        os.system(
            f"python log_ltn.py --csv-path results/{n_examples}/logltn_default_{j}_{seed}.csv"
            f" --n-examples-train {n_examples} --seed {seed} --imbalance {j} --use_focal")
