from collections import defaultdict

from tqdm import tqdm


def train(
        epochs,
        metrics_dict,
        ds_train,
        ds_test,
        train_step,
        test_step,
        csv_path=None,
        scheduled_parameters=None,
        logger=None
):
    """
    Args:
        epochs: int, number of training epochs.
        metrics_dict: dict, {"metrics_label": tf.keras.metrics instance}.
        ds_train: iterable dataset, e.g. using tf.data.Dataset.
        ds_test: iterable dataset, e.g. using tf.data.Dataset.
        train_step: callable function. the arguments passed to the function
            are the itered elements of ds_train.
        test_step: callable function. the arguments passed to the function
            are the itered elements of ds_test.
        csv_path: (optional) path to create a csv file, to save the metrics.
        scheduled_parameters: (optional) a dictionary that returns kwargs for
            the train_step and test_step functions, for each epoch.
            Call using scheduled_parameters[epoch].
    """
    if scheduled_parameters is None:
        scheduled_parameters = defaultdict(lambda: {})
    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    if csv_path is not None:
        csv_file = open(csv_path, "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    pbar = tqdm(range(epochs), desc="Epoch: 0")
    for epoch in pbar:
        for metrics in metrics_dict.values():
            metrics.reset_state()

        for batch_elements in tqdm(ds_train, desc="Train steps", leave=False):
            train_step(*batch_elements, **scheduled_parameters[epoch])
        for batch_elements in tqdm(ds_test, desc="Test steps", leave=False):
            test_step(*batch_elements, **scheduled_parameters[epoch])

        metrics_results = [metrics.result() for metrics in metrics_dict.values()]
        if logger is not None:
            tmp = dict(zip(metrics_dict.keys(), metrics_results))
            for k in tmp.keys():
                tmp[k] = float(tmp[k])
            tmp["epoch"] = epoch
            logger.log(tmp)
        pbar.set_description(template.format(epoch, *metrics_results))
        if csv_path is not None:
            csv_file.write(csv_template.format(epoch, *metrics_results) + "\n")
            csv_file.flush()
    if csv_path is not None:
        csv_file.close()
