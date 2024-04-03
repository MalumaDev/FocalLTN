import os

for n_examples in [1000]:
    for seed in range(1):  # range(5):
        for k in [1, -1]:
            for j in [1., 0.75, .5, .25]:
                if k == -1 and j == 1.:
                    continue
                # Stable Product p=2
                imbalance = j*k
                os.system(
                    f"python main.py --out_path results/{n_examples}/stable_product_p2_{imbalance}_{seed}.csv"
                    f" -n {n_examples} -p 2 --seed {seed} --imbalance {imbalance} -t stable")

                # Stable Product p=6
                os.system(
                    f"python stable_product.py --out_path results/{n_examples}/stable_product_p6_{imbalance}_{seed}.csv"
                    f" -n {n_examples} -p 6 --seed {seed} --imbalance {imbalance} -t stable")