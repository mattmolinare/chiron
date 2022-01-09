import json
import tkinter as tk

import tensorflow as tf
import yaml


def set_visible_gpus(*indices):
    """Set visibility of GPU devices.

    Parameters
    ----------
    *indices
        GPU device indices.

    References
    ----------
    https://www.tensorflow.org/api_docs/python/tf/config/set_visible_devices

    """
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.set_visible_devices(
            [physical_devices[index] for index in indices], "GPU"
        )
    except RuntimeError:
        pass


def get_distribution_strategy():
    """Get distribution strategy for multiple GPU capability.

    References
    ----------
    https://www.tensorflow.org/guide/distributed_training
    """
    logical_devices = tf.config.list_logical_devices("GPU")
    if len(logical_devices) > 1:
        return tf.distribute.MirroredStrategy()
    else:
        return tf.distribute.get_strategy()


def convert_to_tensor(value, dtype=None):
    """Convert to tensor.

    Unlike :func:`tf.convert_to_tensor`, allows casting if input is already a
    tensor.
    """
    if isinstance(value, tf.Tensor):
        return value if dtype is None else tf.cast(value, dtype)
    else:
        return tf.convert_to_tensor(value, dtype=dtype)


class HashTable(tf.lookup.StaticHashTable):
    """Hash table with dictionary constructor."""

    def __init__(self, *args, **kwargs):
        self.assert_in_table = kwargs.pop("assert_in_table", False)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, data, default_value, **kwargs):
        """Instantiate hash table from dictionary."""
        keys = list(data.keys())
        values = list(data.values())
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
        return cls(initializer, default_value, **kwargs)

    def __getitem__(self, keys):
        values = super().__getitem__(convert_to_tensor(keys))
        if self.assert_in_table:
            assert_op = tf.Assert(
                tf.reduce_all(values != self.default_value), [keys]
            )
            with tf.control_dependencies([assert_op]):
                values = tf.identity(values)
        return values


def save_yaml(filename, data):
    """Save data to YAML file."""
    with open(filename, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)


def load_yaml(filename):
    """Load data from YAML file."""
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def save_model(filename, model):
    """Save model configuration to JSON file."""
    with open(filename, "w") as file:
        json.dump(json.loads(model.to_json()), file, indent=2)


def load_model(filename):
    """Load model configuration from JSON file."""
    with open(filename, "r") as file:
        return tf.keras.models.model_from_json(file.read())


def get_weights_path():
    return "epoch-{epoch:04d}"


def standardize(x, axis=None):
    """Standardize data along given axis."""
    if axis is None:
        axis = tf.range(tf.rank(x))
    x_mean, x_var = tf.nn.moments(x, axis, keepdims=True)
    return tf.math.divide_no_nan(x - x_mean, tf.sqrt(tf.maximum(x_var, 0.0)))


def min_max_scale(x, axis=None):
    """Scale data to range 0 to 1 along given axis."""
    x = x - tf.reduce_min(x, axis=axis, keepdims=True)
    return tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))
