import tensorflow as tf
import yaml

__all__ = [
    "convert_to_tensor",
    "HashTable",
    "set_visible_gpus",
    "load_yaml",
    "save_yaml",
]


def set_visible_gpus(*indices):
    """Set visibility of GPU devices.

    Parameters
    ----------
    *indices
        GPU device indices.

    """
    physical_devices = tf.config.list_physical_devices(device_type="GPU")
    visible_devices = [physical_devices[index] for index in indices]
    try:
        tf.config.set_visible_devices(visible_devices, device_type="GPU")
    except RuntimeError:
        pass


def convert_to_tensor(value, dtype=None):
    """Convert to tensor.

    Unlike :func:`tf.convert_to_tensor`, allows casting if input is already a
    tensor.
    """
    if isinstance(value, tf.Tensor):
        return value if dtype is None else tf.cast(value, dtype)
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


def load_yaml(filename):
    """Load data from YAML file."""
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def save_yaml(filename, data):
    """Save data to YAML file."""
    with open(filename, "w") as file:
        yaml.safe_dump(data, stream=file, default_flow_style=False)
