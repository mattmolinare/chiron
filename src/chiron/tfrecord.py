import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tfrecord(filename, generator):
    """
    Save image data to a TFRecord file.

    Parameters
    ----------
    filename : str
        TFRecord file.

    generator : generator
        Image data generator.

    """
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in generator:
            feature = {
                "image": _bytes_feature([image.tobytes()]),
                "image_shape": _int64_feature(list(image.shape)),
                "label": _bytes_feature([label.encode()]),
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


@tf.autograph.experimental.do_not_convert
def _parser(serialized):
    """Parse image data."""
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_shape": tf.io.FixedLenFeature([3], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    image = tf.io.decode_raw(parsed["image"], tf.float32)
    image = tf.reshape(image, parsed["image_shape"])
    return image, parsed["label"]


def load_tfrecord(filenames):
    """Load TFRecord file and parse data.

    Parameters
    ----------
    filenames : str or list of str
        TFRecord files.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset.

    """
    return tf.data.TFRecordDataset(filenames).map(_parser)
