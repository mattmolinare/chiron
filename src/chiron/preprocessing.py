import abc

import tensorflow as tf

from . import utils

__all__ = [
    "IdentityWhitener",
    "LabelEncoder",
    "PerBatchStandardWhitener",
    "PerImageStandardWhitener",
    "Resizer",
    "ShapeSetter",
    "StandardWhitener",
]


class LabelEncoder:
    """Label encoder."""

    def __init__(self, label_map, num_classes, default_class_index=None):
        if default_class_index is None:
            default_class_index = -1
            assert_in_table = True
        else:
            assert_in_table = False
        self.label_map = utils.HashTable.from_dict(
            label_map, default_class_index, assert_in_table=assert_in_table
        )
        self.num_classes = num_classes

    @tf.autograph.experimental.do_not_convert
    def __call__(self, image, labels):
        """One-hot encode labels."""
        class_indices = self.label_map[labels]
        labels = tf.one_hot(class_indices, self.num_classes)
        return image, labels


class ImageTransformer(abc.ABC):
    """Image transformer."""

    @abc.abstractmethod
    def _transform_op(self, images):
        pass

    def __call__(self, images, *args):
        """Transform image data.

        Additional parameters `args` are passed through.
        """
        images = self._transform_op(images)
        return (images, *args) if args else images


class ShapeSetter(ImageTransformer):
    """Image shape setter."""

    def __init__(self, shape):
        self.shape = shape

    def _transform_op(self, images):
        images.set_shape(self.shape)
        return images


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
