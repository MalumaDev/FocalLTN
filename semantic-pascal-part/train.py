import datetime
import logging
import random
import sys

sys.path.insert(1, '../')
from typing import Any
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

import wandb
from tqdm import tqdm
import yaml
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger

logging.getLogger().setLevel(logging.INFO)

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = "config.yml"
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
if "random_seed" not in config:
    config["random_seed"] = random.randint(0, 2 ** 32 - 1)

if "workers" not in config:
    config["workers"] = 12

if "chunk_size" not in config:
    config["chunk_size"] = 100

import data_processing
import ltn.wrapper as ltnw
import ltn.utils as ltnu
import pascalpart_theory, pascalpart_theory_log
import pascalpart_domains
import evaluation

data_processing.config = config
pascalpart_domains.config = config
pascalpart_theory.config = config
pascalpart_theory_log.config = config
evaluation.config = config

data_processing.set_types()

DomGroup = tuple[str]
DomLabel = str


def log_value(name: str, value: float, step: int = None) -> None:
    if name == "step":
        raise ValueError("Metrics name cannot be 'step'.")
    wandb.log(
        {name: value}, step=step, commit=False
    )


def log_dict_of_values(
        names_to_values: dict[str, float], step: int = None
) -> None:
    wandb.log(
        names_to_values, step=step
    )


def get_dom_group(constraint: ltnw.Constraint) -> DomGroup:
    return tuple(sorted([dom.label for dom in constraint.doms_feed_dict.values()]))


def get_kwarg_to_dom_label_mapping(constraint: ltnw.Constraint) -> dict[str, str]:
    return {kwarg: dom.label for (kwarg, dom) in constraint.doms_feed_dict.items()}


def get_dom_label_to_kwarg_mapping(constraint: ltnw.Constraint) -> dict[str, str]:
    return {dom.label: kwarg for (kwarg, dom) in constraint.doms_feed_dict.items()}


def apply_key_mapping_to_dict(feed_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    return {mapping[old_key]: values for (old_key, values) in feed_dict.items()}


def train(
        theory: ltnw.Theory,
        test_datasets: evaluation.TestDatasets,
        test_loggers: list[ltnu.logging.MetricsLogger],
        epoch_range: tuple[int, int] = None,
        pretraining: bool = False, logger=None
):
    epoch_range = epoch_range if epoch_range is not None else (0, config["epochs"])
    constraints = theory.constraints if not pretraining else [cstr for cstr in theory.constraints
                                                              if cstr.label.endswith("groundtruth")]
    print(f"{len(constraints)} constraints:")
    for c in constraints:
        print(c.label)
    for epoch in tqdm(range(*epoch_range), desc="Epochs"):
        theory.constraints[0].operator_config.update_schedule(epoch)
        for _ in tqdm(range(config["training_steps_per_epoch"]), desc="Steps"):
            if pretraining:
                theory.train_step_from_domains([cstr for cstr in theory.constraints
                                                if cstr.label.endswith("groundtruth")])
            else:
                theory.train_step_from_domains()
        theory.reset_metrics()
        evaluation.test_step(theory, test_datasets=test_datasets, loggers=test_loggers,
                             step=theory.step)


if __name__ == "__main__":
    # check if run already exists
    name = f"{config['ltn_config']}"
    if config["ltn_config"] == "stable_rl":
        name += f"_{config['p_universal_quantifier']}"
    if "focal" in name and "gamma" in config:
        name += f"_{config['gamma']}"

    if "data_ratio" in config:
        name += f"_{config['data_ratio']}"

    config["group_name"] = name

    name += f"_{config['random_seed']}_test"

    # runs = wandb.Api().runs(f"grains-polito/NeSy24PascalPart_{config['data_category'].upper()}")
    # for run in runs:
    #     if run.name == name:
    #         logging.info(f"Run {name} already exists.")
    #         sys.exit()

    keras.utils.set_random_seed(config["random_seed"])
    logging.info("Loading training data.")
    pascal_data = pascalpart_domains.get_pascalpart_data()
    pascal_doms = pascalpart_domains.data_to_domains(pascal_data)

    # loggers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer_train = tf.summary.create_file_writer(
        "logging/gradient_tape/" + current_time + "/train")
    summary_writer_test = tf.summary.create_file_writer(
        "logging/gradient_tape/" + current_time + "/test")
    tfsummary_logger_train = ltnu.logging.TfSummaryLogger(summary_writer_train)
    tfsummary_logger_test = ltnu.logging.TfSummaryLogger(summary_writer_test)

    df_logger_train = ltnu.logging.DataFrameLogger()
    df_logger_test = ltnu.logging.DataFrameLogger()

    run = wandb.init(
        project=f"NeSy24PascalPart_{config['data_category'].upper()}",
        config=config,
        name=name,
        entity="grains-polito"
    )

    wandb_logger = WandbMetricsLogger()

    wandb_logger.log_value = log_value
    wandb_logger.log_dict_of_values = log_dict_of_values

    wandb_logger.commit = lambda: wandb.log({})

    train_loggers = [tfsummary_logger_train, df_logger_train, wandb_logger]
    test_loggers = [tfsummary_logger_test, df_logger_test, wandb_logger]

    # training
    use_prior_rules = (config["with_mereological_axioms"] or config["with_clustering_axioms"])
    epochs_pretraining = config["epochs_of_pretraining"] if use_prior_rules else 0
    epochs_normal_training = config["epochs"]

    logging.info("Building LTN theory.")
    if config["ltn_config"] in ["stable_rl", "prod_rl", "focal_ltn", "focal_ltn_sum"]:
        theory = pascalpart_theory.get_theory(
            class_to_id=data_processing.get_classes_to_id(),
            part_to_wholes=data_processing.get_part_to_wholes_ontologies(),
            whole_to_parts=data_processing.get_whole_to_parts_ontologies(),
            pascal_doms=pascal_doms,
            metrics_loggers=train_loggers,
        )
    else:
        theory = pascalpart_theory_log.get_theory(
            class_to_id=data_processing.get_classes_to_id(),
            part_to_wholes=data_processing.get_part_to_wholes_ontologies(),
            whole_to_parts=data_processing.get_whole_to_parts_ontologies(),
            pascal_doms=pascal_doms,
            metrics_loggers=train_loggers
        )
    logging.info("Loading testing data.")
    test_datasets = evaluation.get_test_datasets()

    if epochs_pretraining > 0:
        logging.info("----STARTING PRE-TRAINING----")
        train(theory, test_datasets, test_loggers,
              epoch_range=(0, epochs_pretraining), pretraining=True, logger=run)

    logging.info("----STARTING TRAINING----")
    train(theory, test_datasets, test_loggers,
          epoch_range=(epochs_pretraining, epochs_normal_training), logger=run)
    # save csv
    csv_prefix = config["ltn_config"]
    if config["ltn_config"] == "stable_rl":
        csv_prefix += f"_{config['p_universal_quantifier']}"
    df_logger_train.to_csv("logging/dataframes/" + csv_prefix + current_time + "train.csv")
    df_logger_test.to_csv("logging/dataframes/" + csv_prefix + current_time + "test.csv")
