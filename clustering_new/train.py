import click
import pandas as pd
import tensorflow as tf
from tqdm import trange

import wandb
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

import focal
import ltn

import data


# GROUNDING
class MLP(tf.keras.Model):
    """ Model to call as P(x,class) """

    def __init__(self, n_classes, hidden_layer_sizes=(16, 16, 16)):
        super().__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        logits = self.dense_class(x)
        return logits


def get_axioms(stable_config, C, cluster, x, y, close_thr, is_greater_than, eucl_dist, p_value):
    if stable_config == "stable_rl":
        Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=6), semantics="forall")
        Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=6))

        # CONSTRAINTS
        def axioms(p_exists):
            axioms = [
                Forall(x, Exists(cluster, C([x, cluster]), p=p_exists)),
                Forall(cluster, Exists(x, C([x, cluster]), p=p_exists)),
                Forall([cluster, x, y], Implies(C([x, cluster]), C([y, cluster])),
                       mask=is_greater_than([close_thr, eucl_dist([x, y])]))
            ]

            sat_level = formula_aggregator(axioms).tensor
            return sat_level
    elif stable_config == "prod_rl":
        Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_LogProd(), semantics="forall")
        Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean())

        def axioms(p_exists):
            axioms = [
                Forall(x, Exists(cluster, C([x, cluster]), p=p_exists)),
                Forall(cluster, Exists(x, C([x, cluster]), p=p_exists)),
                Forall([cluster, x, y], Implies(C([x, cluster]), C([y, cluster])),
                       mask=is_greater_than([close_thr, eucl_dist([x, y])]))
            ]

            sat_level = formula_aggregator(axioms).tensor
            return sat_level

    elif stable_config == "log_ltn":
        And = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
        Or = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogSumExp(alpha=5))
        Forall = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_Mean(), semantics="forall")
        Exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogSumExp(alpha=5), semantics="exists")
        formula_aggregator = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Sum())

        # CONSTRAINTS
        def axioms(alpha_exists):
            axioms = [
                Forall(x, Exists(cluster, C.log([x, cluster]), alpha=alpha_exists)),
                Forall(cluster, Exists(x, C.log([x, cluster]), alpha=alpha_exists)),
                Forall([cluster, x, y],
                       Or(C.nlog([x, cluster]), C.log([y, cluster]), alpha=alpha_exists),
                       mask=is_greater_than([close_thr, eucl_dist([x, y])])),
            ]

            sat_level = formula_aggregator(axioms).tensor
            return sat_level

    elif stable_config == "focal_ltn":
        Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        Equiv = ltn.Wrapper_Connective(
            ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.And_Prod(), ltn.fuzzy_ops.Implies_Reichenbach()))
        Forall = ltn.Wrapper_Quantifier(focal.FocalAggreg(gamma=p_value, is_log=False), semantics="forall")
        Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean())

        # CONSTRAINTS
        def axioms(p_exists):
            axioms = [
                Forall(x, Exists(cluster, C([x, cluster]))),
                Forall(cluster, Exists(x, C([x, cluster]))),
                Forall([cluster, x, y], Implies(C([x, cluster]), C([y, cluster])),
                       mask=is_greater_than([close_thr, eucl_dist([x, y])]))
            ]

            sat_level = formula_aggregator(axioms).tensor
            return sat_level

    elif stable_config == "focal_log_ltn":
        And = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.And_Sum())
        Or = ltn.log.Wrapper_Connective(ltn.log.fuzzy_ops.Or_LogSumExp(alpha=5))
        Forall = ltn.log.Wrapper_Quantifier(focal.FocalAggreg(gamma=p_value, is_log=True), semantics="forall")
        Exists = ltn.log.Wrapper_Quantifier(ltn.log.fuzzy_ops.Aggreg_LogSumExp(alpha=5), semantics="exists")
        formula_aggregator = ltn.log.Wrapper_Formula_Aggregator(ltn.log.fuzzy_ops.Aggreg_Sum())

        # CONSTRAINTS
        def axioms(alpha_exists):
            axioms = [
                Forall(x, Exists(cluster, C.log([x, cluster]))),
                Forall(cluster, Exists(x, C.log([x, cluster]))),
                Forall([cluster, x, y],
                       Or(C.nlog([x, cluster]), C.log([y, cluster])),
                       mask=is_greater_than([close_thr, eucl_dist([x, y])])),
            ]

            sat_level = formula_aggregator(axioms).tensor
            return sat_level
    else:
        raise ValueError(f"Unknown stable config: {stable_config}")

    return axioms


@click.command()
@click.option('--csv-path', type=str, default="TCGA-PANCAN-HiSeq-801x20531/pca_16d.csv")
@click.option('--seed', type=int, default=1300)
@click.option('--stable-config', type=str,
              default="stable_rl")  # 'stable_rl', 'prod_rl', 'log_ltn', 'log_ltn_lse', 'focal_ltn', 'focal_log_ltn'
@click.option('--p_value', type=int, default=6)
@click.option('--lr', type=float, default=0.002)
@click.option('--epochs', type=int, default=1000)
def main(csv_path, seed, stable_config, p_value, lr, epochs):
    config = locals()
    # DATA
    tf.keras.utils.set_random_seed(seed)
    dataset = data.load_pca_data(csv_path)
    nr_of_clusters = dataset.nb_clusters
    features = dataset.features

    distances = euclidean_distances(features, features)
    close_threshold = np.percentile(distances, 5)

    logits_model = MLP(nr_of_clusters)
    if "log" in stable_config:
        C = ltn.log.Predicate.FromLogits(logits_model, activation_function="softmax", with_class_indexing=True)
    else:
        C = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
    cluster = ltn.Variable("cluster", range(nr_of_clusters))

    x = ltn.Variable("x", features)
    y = ltn.Variable("y", features)

    eucl_dist = ltn.Function.Lambda(lambda inputs: tf.expand_dims(tf.norm(inputs[0] - inputs[1], axis=1), axis=1))
    is_greater_than = ltn.Predicate.Lambda(lambda inputs: inputs[0] > inputs[1])
    close_thr = ltn.Constant(close_threshold, trainable=False)

    axioms = get_axioms(stable_config, C, cluster, x, y, close_thr, is_greater_than, eucl_dist, p_value)

    axioms(p_value)  # first call to build the graph

    # TRAINING
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    name = stable_config + "_" + str(p_value)
    if "data_ratio" in config:
        name += f"_{config['data_ratio']}"
    config["group_name"] = name
    name = f"{name}_{seed}"
    try:
        runs = wandb.Api().runs(f"grains-polito/NeSy24Clustering")
        for run in runs:
            if run.name == name:
                print(f"Run {name} already exists.")
                exit()
    except:
        pass

    run = wandb.init(
        project=f"NeSy24Clustering",
        config=config,
        name=name,
        entity="grains-polito"
    )
    epochs_fixed_schedule = 0
    p_exists = np.concatenate([[1] * epochs_fixed_schedule, np.linspace(1, 6, epochs - epochs_fixed_schedule)])

    for epoch in trange(epochs):
        with tf.GradientTape() as tape:
            loss = - axioms(p_exists[epoch])
        grads = tape.gradient(loss, logits_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))
        if epoch % 100 == 0:
            sat = axioms(p_exists[epoch])
            loss = - sat
            if "log" in stable_config or "prod_rl" in stable_config:
                sat = tf.math.exp(-sat)
            run.log({"sat": sat.numpy(), "loss": loss.numpy(), "epoch": epoch})
            # print("Epoch %d: Sat Level %.3f, Loss %.3f" % (epoch, sat, loss))
    sat = axioms(p_exists[epoch])
    loss = - sat
    if "log" in stable_config or "prod_rl" in stable_config:
        sat = tf.math.exp(-sat)
    print("Training finished at Epoch %d with Sat Level %.3f, Loss %.3f" % (epoch, sat, loss))
    run.log({"sat": sat.numpy(), "loss": loss.numpy(), "epoch": epoch})

    # EVALUATE
    predictions = tf.math.argmax(C([x, cluster]).tensor, axis=1)
    rand_score = data.adjusted_rand_score(dataset.labels, predictions)
    run.log({"eval/adjusted_rand_score": rand_score})
    pcadf = pd.DataFrame(features, columns=[f"component_{i}" for i in range(features.shape[1])])
    pcadf["predicted_cluster"] = predictions
    pcadf["true_label"] = dataset.label_names
    test_table = wandb.Table(dataframe=pcadf)

    run.log({"table_results": test_table})


if __name__ == '__main__':
    main()
