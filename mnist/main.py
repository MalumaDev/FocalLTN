import logging
from pathlib import Path
from random import randint

import click
import tensorflow as tf

import ltn
import wandb
from focal import FocalAggreg
from mnist import commons
from mnist.data import get_mnist_dataset
from mnist.model import SingleDigit

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--seed", "-s", default=0, type=int, help="Seed")
@click.option("--num_examples_per_class", "-n", default=600, type=int,
              help="Number of examples per class. All for maximum number of examples.")
@click.option("--imbalance", "-i", default=0.75, type=float)
@click.option("--p_value", "-p", default=2, type=int)
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--lr", "-l", default=1e-3, type=float)
@click.option("--out_path", "-o", default="results/test.csv", type=str)
@click.option("--use_focal", is_flag=True)
@click.option("--gamma", default=2, type=float)
@click.option("--a_type", "-t", default="stable", type=str)
@click.option("--batch_size", "-b", default=64, type=int)
@click.option("--val_batch_size", "-vb", default=256, type=int)
def main(seed, num_examples_per_class, imbalance, p_value, epochs, lr, out_path, use_focal, gamma, a_type, batch_size,
         val_batch_size):
    if seed == -1:
        seed = randint(0, 2 ** 32 - 1)

    if use_focal:
        a_type += "_focal"
    args = locals()
    tf.keras.utils.set_random_seed(seed)
    ds_train, ds_test, distribution = get_mnist_dataset(num_examples_per_class, imbalance=imbalance, batch_size=batch_size,
                                                        val_batch_size=val_batch_size)
    out_path = Path(out_path)
    if out_path.exists():
        print(f"File {out_path} already exists. Exiting.")
        exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logits_model = SingleDigit()

    Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")

    classes = [ltn.Constant(i, trainable=False) for i in range(10)]

    Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
    And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
    Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
    Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())

    if not use_focal:
        Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=p_value), semantics="forall")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=p_value))
    else:
        Forall = ltn.Wrapper_Quantifier(FocalAggreg(gamma=gamma, is_log=False), semantics="forall")
        formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Mean())

    def axioms(features, labels, training=False):
        axioms = []
        for i, c in enumerate(classes):
            if i not in labels:
                continue
            c_x = ltn.Variable(f"x_{i}", features[labels == i])
            axioms.append(Forall(c_x, Digit([c_x, c], training=training)), )

        sat_level = formula_aggregator(axioms).tensor
        # if tf.math.reduce_any(tf.math.is_nan(sat_level)):
        #     raise ValueError("NaN detected")
        return sat_level

    for features, labels in ds_test:
        print("Initial sat level %.5f" % axioms(features, labels))
        break

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def train_step(features, labels):
        # sat and update
        with tf.GradientTape() as tape:
            sat = axioms(features, labels, training=True)
            if use_focal:
                loss = - sat
            else:
                loss = 1. - sat
        gradients = tape.gradient(loss, Digit.trainable_variables)
        optimizer.apply_gradients(zip(gradients, Digit.trainable_variables))
        sat = axioms(features, labels) # compute sat without dropout
        if tf.math.reduce_any(tf.math.is_nan(sat)):
            raise ValueError("NaN detected")
        metrics_dict['train_sat_kb'](sat)
        # accuracy
        predictions = logits_model([features])
        metrics_dict['train_accuracy'](tf.one_hot(labels, 10), tf.nn.softmax(predictions))

    def test_step(features, labels):
        # sat
        sat = axioms(features, labels)
        metrics_dict['test_sat_kb'](sat)
        # accuracy
        predictions = logits_model([features])
        metrics_dict['test_accuracy'](tf.one_hot(labels, 10), tf.nn.softmax(predictions))

    metrics_dict = {
        'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
        'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
        'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
        'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
    }

    name = a_type
    if use_focal:
        name += f"{args['gamma']}"
    run = wandb.init(
        project="NeSy24-mnist",
        config=args,
        name=name + f"_{imbalance}_{num_examples_per_class}_{args['seed']}",
        entity="grains-polito"
    )

    table = wandb.Table(data=list(zip(distribution[0].tolist(), distribution[1].tolist())), columns=["class", "n_samples"])
    run.log(
        {
            "distribution_classes": wandb.plot.bar(
                table, "class", "n_samples", title="Distribution of classes"
            )
        }
    )

    commons.train(
        epochs,
        metrics_dict,
        ds_train,
        ds_test,
        train_step,
        test_step,
        csv_path=out_path,
        logger=run
    )


if __name__ == "__main__":
    main()
