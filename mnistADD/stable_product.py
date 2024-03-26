import argparse
import sys
from pathlib import Path
from random import randint

import tensorflow as tf
import wandb

import ltn
import baselines, data, commons
import argparse
import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path', type=str, default="MDadd_ltn_stable_product_p2.csv")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--n-examples-train', type=str, default="all")
    parser.add_argument('--n-examples-test', type=int, default=2500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--imbalance', type=float, default=1.)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


args = parse_args()
n_examples_train = args['n_examples_train']
n_examples_train = int(n_examples_train) if n_examples_train != "all" else "all"
n_examples_test = args['n_examples_test']
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = Path(args['csv_path'])
p_forall = args["p"]
imbalance = args['imbalance']

if csv_path.exists():
    print(f"File {csv_path} already exists. Exiting.")
    sys.exit(0)

csv_path.parent.mkdir(parents=True, exist_ok=True)

if args['seed'] == -1:
    args['seed'] = randint(0, 2**32 - 1)

tf.keras.utils.set_random_seed(args['seed'])

""" DATASET """
ds_train, ds_test = data.get_mnist_op_dataset(
    count_train=n_examples_train,
    count_test=n_examples_test,
    buffer_size=10000,
    batch_size=batch_size,
    n_operands=4,
    op=lambda args: 10 * args[0] + args[1] + 10 * args[2] + args[3],
    imbalance=imbalance
)

""" LTN MODEL AND LOSS """
### Predicates
logits_model = baselines.SingleDigit(inputs_as_a_list=True)
Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
### Variables
d1 = ltn.Variable("digits1", range(10))
d2 = ltn.Variable("digits2", range(10))
d3 = ltn.Variable("digits3", range(10))
d4 = ltn.Variable("digits4", range(10))
### Operators
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=p_forall), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(), semantics="exists")

# mask
add = ltn.Function.Lambda(lambda inputs: inputs[0] + inputs[1])
times = ltn.Function.Lambda(lambda inputs: inputs[0] * inputs[1])
ten = ltn.Constant(10, trainable=False)
equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
two_digit_number = lambda inputs: add([times([ten, inputs[0]]), inputs[1]])


def axioms(images_x1, images_x2, images_y1, images_y2, labels_z, p_schedule):
    images_x1 = ltn.Variable("x1", images_x1)
    images_x2 = ltn.Variable("x2", images_x2)
    images_y1 = ltn.Variable("y1", images_y1)
    images_y2 = ltn.Variable("y2", images_y2)
    labels_z = ltn.Variable("z", labels_z)
    axiom = Forall(
        ltn.diag(images_x1, images_x2, images_y1, images_y2, labels_z),
        Exists(
            (d1, d2, d3, d4),
            And(
                And(Digit([images_x1, d1]), Digit([images_x2, d2])),
                And(Digit([images_y1, d3]), Digit([images_y2, d4]))
            ),
            mask=equals([labels_z, add([two_digit_number([d1, d2]), two_digit_number([d3, d4])])]),
            p=p_schedule
        ),
        p=2
    )
    sat = axiom.tensor
    return sat


### Initialize layers and weights
x1, x2, y1, y2, z = next(ds_train.as_numpy_iterator())
axioms(x1, x2, y1, y2, z, 2)

""" TRAINING """

optimizer = tf.keras.optimizers.Adam(0.001)
metrics_dict = {
    'train_loss': tf.keras.metrics.Mean(name="train_loss"),
    'train_accuracy': tf.keras.metrics.Mean(name="train_accuracy"),
    'test_loss': tf.keras.metrics.Mean(name="test_loss"),
    'test_accuracy': tf.keras.metrics.Mean(name="test_accuracy")
}


def train_step(images_x1, images_x2, images_y1, images_y2, labels_z, **kwargs):
    # loss
    with tf.GradientTape() as tape:
        loss = 1. - axioms(images_x1, images_x2, images_y1, images_y2, labels_z, **kwargs)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]), axis=-1)
    predictions_x2 = tf.argmax(logits_model([images_x2]), axis=-1)
    predictions_y1 = tf.argmax(logits_model([images_y1]), axis=-1)
    predictions_y2 = tf.argmax(logits_model([images_y2]), axis=-1)
    predictions_z = 10 * predictions_x1 + predictions_x2 + 10 * predictions_y1 + predictions_y2
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


def test_step(images_x1, images_x2, images_y1, images_y2, labels_z, **kwargs):
    # loss
    loss = 1. - axioms(images_x1, images_x2, images_y1, images_y2, labels_z, **kwargs)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]), axis=-1)
    predictions_x2 = tf.argmax(logits_model([images_x2]), axis=-1)
    predictions_y1 = tf.argmax(logits_model([images_y1]), axis=-1)
    predictions_y2 = tf.argmax(logits_model([images_y2]), axis=-1)
    predictions_z = 10 * predictions_x1 + predictions_x2 + 10 * predictions_y1 + predictions_y2
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


from collections import defaultdict

scheduled_parameters = defaultdict(lambda: {})
p_min = 1
p_max = 5
p_exists_values = np.linspace(p_min, p_max, EPOCHS, dtype=np.float32)
for epoch in range(EPOCHS):
    scheduled_parameters[epoch] = {"p_schedule": tf.constant(p_exists_values[epoch])}

run = wandb.init(
    project="NeSy24",
    config=args,
    name=f"SP_{p_forall}_{imbalance}_{n_examples_train}_{args['seed']}",
    entity="grains-polito"
)

commons.train(
    EPOCHS,
    metrics_dict,
    ds_train,
    ds_test,
    train_step,
    test_step,
    csv_path=csv_path,
    scheduled_parameters=scheduled_parameters,
    logger=run
)
