import tensorflow as tf
import tensorflow_hub as hub

__all__ = ["get_model"]


def get_model(num_classes, dropout_rate=0, regularizer=None):
    return tf.keras.Sequential(
        [
            hub.KerasLayer(
                "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
                trainable=True,
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizer),
        ]
    )
