import numpy as np
import tensorflow as tf


def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train / 255.0, img_test / 255.0
    img_train = img_train[..., tf.newaxis]
    img_test = img_test[..., tf.newaxis]
    return img_train, label_train, img_test, label_test


def get_mnist_dataset(
        count_train,
        batch_size=64,
        buffer_size=60000,
        op=lambda args: args[0] + args[1],
        imbalance=1.
):
    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()
    neg_imbalance = imbalance < 0
    if neg_imbalance:
        imbalance = -imbalance

    unique, counts = np.unique(label_train, return_counts=True)
    a_max, a_min = np.argmax(counts), np.argmin(counts)
    if count_train != "all":
        assert count_train <= counts[a_min]
    else:
        count_train = counts[a_min]

    imbalanced_class = np.random.choice(unique)

    for j, c in enumerate(unique):
        mask = label_train == j
        n_remove = int(counts[j] - count_train)

        if neg_imbalance and c != imbalanced_class:
            n_remove += int((1 - imbalance) * count_train)
        elif not neg_imbalance and c == imbalanced_class:
            n_remove += int((1 - imbalance) * count_train)

        if n_remove == 0:
            continue
        toberemoved = np.random.choice(np.where(mask)[0], n_remove, replace=False)
        label_train = np.delete(label_train, toberemoved)
        img_train = np.delete(img_train, toberemoved, axis=0)

    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train)) \
        .shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test)).batch(batch_size)

    return ds_train, ds_test, np.unique(label_train, return_counts=True)
