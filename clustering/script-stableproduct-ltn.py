import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import ltn
from focal import FocalAggreg

from random import randint
import wandb
import data
import argparse
from pathlib import Path
import sys
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path', type=str, default="Cluster_ltn_stable_product_p2.csv")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--use_focal', action='store_true')
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--imbalance', type=float, default=0.99)
    parser.add_argument('--forall_only', action='store_true')
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
epochs = args['epochs']
imbalance = args['imbalance']
use_focal = args['use_focal']
p_forall = args["p"]
csv_path = Path(args['csv_path'])


if csv_path.exists():
    print(f"File {csv_path} already exists. Exiting.")
    sys.exit(0)

csv_path.parent.mkdir(parents=True, exist_ok=True)

if args['seed'] == -1:
    args['seed'] = randint(0, 2 ** 32 - 1)

tf.keras.utils.set_random_seed(args['seed'])

# DATA
dataset = data.load_pca_data("TCGA-PANCAN-HiSeq-801x20531/pca_16d.csv",
    imbalance=imbalance)
nr_of_clusters = dataset.nb_clusters
features = dataset.features

distances = euclidean_distances(features, features)
close_threshold = np.percentile(distances, 5)

# GROUNDING
class MLP(tf.keras.Model):
    """ Model to call as P(x,class) """
    def __init__(self, n_classes, hidden_layer_sizes=(16,16,16)):
        super().__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        
    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        logits = self.dense_class(x)
        return logits

logits_model = MLP(nr_of_clusters)
C = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
cluster_x = ltn.Variable("cluster_x", dataset.labels)
cluster_y = ltn.Variable("cluster_y", dataset.labels)
cluster = ltn.Variable("cluster", tf.one_hot(dataset.labels, nr_of_clusters))
# cluster = ltn.Variable("cluster", range(nr_of_clusters))

x = ltn.Variable("x",features)
y = ltn.Variable("y",features)

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.And_Prod(),ltn.fuzzy_ops.Implies_Reichenbach()))
if not use_focal:
    Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=p_forall), semantics="forall")
else:
    Forall = ltn.Wrapper_Quantifier(FocalAggreg(gamma=args['gamma'], alpha=5, is_log=False), semantics="forall")
    Forall_s = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_Sum(), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6),semantics="exists")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=6))


eucl_dist = ltn.Function.Lambda(lambda inputs: tf.expand_dims(tf.norm(inputs[0]-inputs[1],axis=1),axis=1))
is_greater_than = ltn.Function.Lambda(lambda inputs: inputs[0] > inputs[1])
close_thr = ltn.Constant(close_threshold, trainable=False)
equal_of = ltn.Function.Lambda(lambda inputs: inputs[0] == inputs[1])
and_func = ltn.Function.Lambda(lambda inputs: tf.logical_and(inputs[0] > 0, inputs[1] > 0))
make_pred = ltn.Predicate.Lambda(lambda input: input)

# CONSTRAINTS
def axioms(p_exists, forall_only=False, use_focal=False):
    if not forall_only:
        axioms = [
            Forall(x, Exists(cluster, C([x,cluster]),p=p_exists)),
            Forall(cluster, Exists(x, C([x,cluster]),p=p_exists)),
            Forall([cluster,x,y], Implies(C([x,cluster]),C([y,cluster])),
                mask = is_greater_than([close_thr,eucl_dist([x,y])]))
        ]
    else:
        if use_focal:
            t = Forall_s(ltn.diag(x, cluster_x),
                         Forall(ltn.diag(y, cluster_y),
                                Implies(C([x, cluster_x]), C([y, cluster_y])),
                                mask=make_pred(and_func([is_greater_than([close_thr, eucl_dist([x, y])]),
                                                         equal_of([cluster_x, cluster_y])]))))
        else:
            t = Forall(ltn.diag(x, cluster_x),
                         Forall(ltn.diag(y, cluster_y),
                                Implies(C([x, cluster_x]), C([y, cluster_y])),
                                mask=make_pred(and_func([is_greater_than([close_thr, eucl_dist([x, y])]),
                                                         equal_of([cluster_x, cluster_y])]))))
        axioms = [
            Forall(ltn.diag(x, cluster_x), C([x, cluster_x])),
            Forall(ltn.diag(y, cluster_y), C([y, cluster_y])),
            t
        ]
    sat_level = formula_aggregator(axioms).tensor
    return sat_level

axioms(p_exists=6, forall_only=args["forall_only"], use_focal=use_focal) # first call to build the graph

# TRAINING
trainable_variables = logits_model.trainable_variables
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

epochs_fixed_schedule = 0
p_exists = np.concatenate([
        [1]*epochs_fixed_schedule,
        np.linspace(1, 4, epochs-epochs_fixed_schedule)])

name = "SP"
if use_focal:
    name += f"_focal{args['gamma']}"
name_ns = name + f"_{p_forall}_{imbalance}_" + str(args["forall_only"])
args["name"] = name_ns
run = wandb.init(
    project="NeSy24Cluster",
    config=args,
    name=name_ns + f"_{args['seed']}",
    entity="grains-polito")

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = - axioms(p_exists[epoch], forall_only=args["forall_only"], use_focal=use_focal)
    grads = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, logits_model.trainable_variables))
    if epoch%100 == 0:
        sat = axioms(p_exists[epoch], forall_only=args["forall_only"], use_focal=use_focal)
        loss = 1 - sat
        print("Epoch %d: Sat Level %.3f, Loss %.3f"%(epoch, sat, loss))
        run.log({"Sat Level": sat, "Loss": loss})
sat = axioms(p_exists[epoch], forall_only=args["forall_only"], use_focal=use_focal)
loss = 1 - sat
print("Training finished at Epoch %d with Sat Level %.3f, Loss %.3f"%(epoch, sat, loss))
run.log({"Sat Level": sat, "Loss": loss})

# EVALUATE
predictions = tf.math.argmax(C([x,cluster_x]).tensor,axis=1)
print(data.adjusted_rand_score(dataset.labels,predictions)) 
data.save_pdf_predictions(features, predictions, dataset.label_names, csv_path=csv_path)
run.log({"AdjRandIdx": data.adjusted_rand_score(dataset.labels,predictions)})