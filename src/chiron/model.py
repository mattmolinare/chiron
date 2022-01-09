import tensorflow as tf
import tensorflow_hub as hub


@tf.keras.utils.register_keras_serializable()
class KerasLayer(hub.KerasLayer):
    pass


def get_model(num_classes, regularizer=None):
    return tf.keras.Sequential(
        [
            KerasLayer(
                "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
                trainable=True,
            ),
            tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizer),
        ]
    )
