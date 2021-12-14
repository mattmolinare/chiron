import abc

import tensorflow as tf

from . import utils

__all__ = [
    "LabelMapper",
    "OneHotEncoder",
    "IdentityWhitener",
    "StandardWhitener",
    "PerBatchStandardWhitener",
    "PerImageStandardWhitener",
    "Resizer",
    "Repeater",
    "ShapeSetter",
]


class TargetTransformer(abc.ABC):
    """Target transformer.

    Parameters
    ----------
    elementwise : bool, optional
        If `True`, transform each element in the target data structure. Default
        is `False`.

    """

    def __init__(self, elementwise=False):
        self.elementwise = elementwise

    @abc.abstractmethod
    def _transform_op(self, targets):
        pass

    @tf.function
    def __call__(self, *args):
        """Transform target vectors.

        Parameters are unpacked as `*extra_args, targets = args`. If not empty,
        `extra_args` are passed through.
        """
        *extra_args, targets = args
        if self.elementwise:
            targets = tf.nest.map_structure(self._transform_op, targets)
        else:
            targets = self._transform_op(targets)
        return (*extra_args, targets) if extra_args else targets


class LabelMapper(TargetTransformer):
    """Map labels to class indices."""

    def __init__(self, label_map, default_class_index=None, elementwise=False):
        super().__init__(elementwise=elementwise)
        if default_class_index is None:
            default_class_index = -1
            assert_in_table = True
        else:
            assert_in_table = False
        self.label_map = utils.HashTable.from_dict(
            label_map, default_class_index, assert_in_table=assert_in_table
        )

    def _transform_op(self, targets):
        return self.label_map.lookup(targets)


class OneHotEncoder(TargetTransformer):
    """One-hot encode class indices."""

    def __init__(self, num_classes, elementwise=False):
        super().__init__(elementwise=elementwise)
        self.num_classes = num_classes

    def _transform_op(self, target):
        return tf.one_hot(target, self.num_classes)


class ImageTransformer(abc.ABC):
    """Image transformer."""

    @abc.abstractmethod
    def _transform_op(self, images):
        pass

    def __call__(self, images, *extra_args):
        """Transform image data.

        Additional parameters `extra_args` are passed through.
        """
        images = self._transform_op(images)
        return (images, *extra_args) if extra_args else images


class BaseWhitener(ImageTransformer):
    """Base image whitener."""

    def _transform_op(self, images):
        return self._whitening_op(images)

    @abc.abstractmethod
    def _whitening_op(self, images):
        pass


class IdentityWhitener(BaseWhitener):
    """Identity whitener."""

    @staticmethod
    def _whitening_op(images):
        return images


def standardize(x, axes=None):
    """Standardize data along given axes."""
    if axes is None:
        axes = tf.range(tf.rank(x))
    x_mean, x_var = tf.nn.moments(x, axes, keepdims=True)
    x_var = tf.maximum(x_var, 0.0)  # Clamp to zero
    return tf.math.divide_no_nan(x - x_mean, tf.sqrt(x_var))


class StandardWhitener(BaseWhitener):
    """Standard whitener."""

    def __init__(self, axes=None):
        self.axes = axes

    def _whitening_op(self, images):
        return standardize(images, axes=self.axes)


class PerBatchStandardWhitener(StandardWhitener):
    """Per-batch standard whitener."""

    def __init__(self):
        super().__init__(axes=[0, 1, 2])


class PerImageStandardWhitener(StandardWhitener):
    """Per-image standard whitener."""

    def __init__(self):
        super().__init__(axes=[1, 2])


class Resizer(ImageTransformer):
    """Image resizer."""

    def __init__(self, size):
        self.size = size

    def _transform_op(self, images):
        return tf.image.resize(images, self.size)


class Repeater(ImageTransformer):
    """Image repeater."""

    def __init__(self, repeats, axis=-1):
        self.repeats = repeats
        self.axis = axis

    def _transform_op(self, images):
        return tf.repeat(images, self.repeats, axis=self.axis)


class ShapeSetter(ImageTransformer):
    """Image shape setter."""

    def __init__(self, shape):
        self.shape = shape

    def _transform_op(self, images):
        return tf.ensure_shape(images, self.shape)
