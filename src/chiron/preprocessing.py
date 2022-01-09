import abc

import tensorflow as tf
from numpy import vectorize

from . import utils


class TargetTransformer(abc.ABC):
    """Target transformer.

    Parameters
    ----------
    apply_elementwise : bool, optional
        If `True`, apply transformation to each element in `targets` data
        structure. Default is `False`.
    """

    def __init__(self, apply_elementwise=False):
        self.apply_elementwise = apply_elementwise

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
        if self.apply_elementwise:
            targets = tf.nest.map_structure(self._transform_op, targets)
        else:
            targets = self._transform_op(targets)
        return (*extra_args, targets) if extra_args else targets


class LabelMapper(TargetTransformer):
    """Label mapper.

    Map class labels to class indices.
    """

    def __init__(
        self, label_map, default_class_index=None, apply_elementwise=False
    ):
        super().__init__(apply_elementwise=apply_elementwise)
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
    """One-hot encoder.

    One-hot encode class indices.
    """

    def __init__(self, num_classes, apply_elementwise=False):
        super().__init__(apply_elementwise=apply_elementwise)
        self.num_classes = num_classes

    def _transform_op(self, target):
        return tf.one_hot(target, self.num_classes)


class ImageTransformer(abc.ABC):
    """Image transformer.

    Parameters
    ----------
    vectorize : bool, optional
        If `True`, apply transformation to each element unstacked on axis 0 of
        `images` tensor. Default is `False`.
    """

    def __init__(self, vectorize=False):
        self.vectorize = vectorize

    @abc.abstractmethod
    def _transform_op(self, images):
        pass

    @tf.function
    def __call__(self, images, *extra_args):
        """Transform image data.

        Additional parameters `extra_args` are passed through.
        """
        if self.vectorize:
            images = tf.map_fn(self._transform_op, images)
        else:
            images = self._transform_op(images)
        return (images, *extra_args) if extra_args else images


class Whitener(ImageTransformer):
    """Image whitener."""

    def __init__(self, axis=None):
        super().__init__(vectorize=False)
        self.axis = axis

    def _transform_op(self, images):
        return self._whitening_op(images, self.axis)

    @abc.abstractmethod
    def _whitening_op(self, images, axis):
        pass


class PerBatchWhitener(Whitener):
    """Per-batch whitener."""

    def __init__(self):
        super().__init__(axis=[-4, -3, -2])


class PerImageWhitener(Whitener):
    """Per-image whitener."""

    def __init__(self):
        super().__init__(axis=[-3, -2])


class Standardizer(Whitener):
    """Standardizer.

    Scale image to zero mean and unit variance.
    """

    @staticmethod
    def _whitening_op(images, axis):
        return utils.standardize(images, axis=axis)


class PerBatchStandardizer(PerBatchWhitener, Standardizer):
    """Per-batch standardizer."""


class PerImageStandardizer(PerImageWhitener, Standardizer):
    """Per-image standardizer."""


class MinMaxScaler(Whitener):
    """Min-max scaler.

    Scale image to range 0 to 1.
    """

    @staticmethod
    def _whitening_op(images, axis):
        return utils.min_max_scale(images, axis=axis)


class PerBatchMinMaxScaler(PerBatchWhitener, MinMaxScaler):
    """Per-batch min-max scaler."""


class PerImageMinMaxScaler(PerImageWhitener, MinMaxScaler):
    """Per-image min-max scaler."""


class Resizer(ImageTransformer):
    """Image resizer."""

    def __init__(self, size):
        super().__init__(vectorize=False)
        self.size = size

    def _transform_op(self, images):
        return tf.image.resize(images, self.size)


class GrayscaleToRgb(ImageTransformer):
    """Grayscale to RGB converter."""

    def __init__(self):
        super().__init__(vectorize=False)

    @staticmethod
    def _transform_op(images):
        return tf.image.grayscale_to_rgb(images)


class ShapeSetter(ImageTransformer):
    """Image shape setter."""

    def __init__(self, shape):
        super().__init__(vectorize=False)
        self.shape = shape

    def _transform_op(self, images):
        return tf.ensure_shape(images, self.shape)


class ConvertImageDtype(ImageTransformer):
    """Convert image data type."""

    def __init__(self, dtype, saturate=False):
        super().__init__(vectorize=False)
        self.dtype = dtype
        self.saturate = saturate

    def _transform_op(self, images):
        return tf.image.convert_image_dtype(
            images, self.dtype, saturate=self.saturate
        )


class RandomJpegQuality(ImageTransformer):
    """Randomly change jpeg encoding quality."""

    def __init__(self, min_quality, max_quality, vectorize=False):
        super().__init__(vectorize=vectorize)
        self.min_quality = min_quality
        self.max_quality = max_quality

    @tf.function
    def _transform_op(self, images):
        return tf.image.random_jpeg_quality(
            images, self.min_quality, self.max_quality
        )


class RandomBrightness(ImageTransformer):
    """Adjust image brightness by random factor."""

    def __init__(self, max_delta):
        super().__init__(vectorize=False)
        self.max_delta = max_delta

    def _transform_op(self, images):
        return tf.image.random_brightness(images, self.max_delta)


class RandomContrast(ImageTransformer):
    """Adjust image contrast by random factor."""

    def __init__(self, lower, upper):
        super().__init__(vectorize=False)
        self.lower = lower
        self.upper = upper

    def _transform_op(self, images):
        return tf.image.random_contrast(images, self.lower, self.upper)


class RandomRot90(ImageTransformer):
    """Rotate image by 90 degrees some random number of times."""

    def __init__(self):
        super().__init__(vectorize=False)

    def _transform_op(self, images):
        return tf.image.rot90(
            images, tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        )


class RandomFlipLeftRight(ImageTransformer):
    """Randomly flip image horizontally."""

    def __init__(self):
        super().__init__(vectorize=False)

    @staticmethod
    def _transform_op(images):
        return tf.image.random_flip_left_right(images)


class RandomFlipUpDown(ImageTransformer):
    """Randomly flip image vertically."""

    def __init__(self):
        super().__init__(vectorize=False)

    @staticmethod
    def _transform_op(images):
        return tf.image.random_flip_up_down(images)
