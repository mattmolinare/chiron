import argparse
import os

import tensorflow as tf
from tensorflow.python.training.tracking.base import (
    no_automatic_dependency_tracking,
)

import chiron


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "train_file", help="Path to training dataset TFRecord file"
    )
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument(
        "--val_file",
        default=None,
        help="Path to validation dataset TFRecord file",
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        type=int,
        default=None,
        help="GPU device indices",
    )
    return parser.parse_args()


def parse_config_file(filename):
    config = chiron.load_yaml(filename)
    class_labels = []
    label_map = {}
    for class_index, class_config in enumerate(config["classes"]):
        class_labels.append(class_config["label"])
        for orig_label in class_config["orig_labels"]:
            label_map[orig_label] = class_index
    config.update(
        {
            "num_classes": class_index + 1,
            "class_labels": class_labels,
            "label_map": label_map,
        }
    )
    return config


def prepare_dataset(
    dataset, label_mapper, one_hot_encoder, image_size, batch_size, whitener
):
    return (
        dataset.map(label_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .map(one_hot_encoder, num_parallel_calls=tf.data.AUTOTUNE)
        .map(chiron.Resizer(image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .map(chiron.Repeater(3, axis=2), num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            chiron.ShapeSetter(image_size + [3]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .map(whitener, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )


def main():

    # Parse command line arguments.
    args = parse_args()

    # Make checkpoint/log directories.
    output_dir = os.path.abspath(args.output_dir)
    log_dir = os.path.join(output_dir, "log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    weights_dir = os.path.join(output_dir, "weights")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    # Set GPU device visibility.
    devices = args.devices
    if devices is not None:
        chiron.set_visible_gpus(*devices)

    # Parse configuration file.
    config = parse_config_file(args.config_file)

    # Save configuration file.
    chiron.save_yaml(os.path.join(output_dir, "config.yaml"), config)

    # Read training dataset from TFRecord file.
    train_dataset = chiron.load_tfrecord(args.train_file)

    # Define regularizer.
    regularizer = tf.keras.regularizers.l2(config["reg_scale"])

    # Get input shape.
    image_size = tf.TensorShape(config["image_size"])
    input_shape = [None] + image_size + [3]

    # Get distribution strategy.
    strategy = chiron.get_distribution_strategy()

    with strategy.scope():

        # Define model.
        model = chiron.get_model(
            config["num_classes"],
            dropout_rate=config["dropout_rate"],
            regularizer=regularizer,
        )

        # Save model configuration.
        chiron.save_model(os.path.join(output_dir, "model.json"), model)

        # Define optimizer.
        optimizer_kwargs = {}
        if config["grad_clip"] is not None:
            optimizer_kwargs["clipvalue"] = config["grad_clip"]
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config["base_lr"], **optimizer_kwargs
        )

        # Define loss.
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Define metrics.
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            # tf.keras.metrics.AUC(curve="ROC", multi_label=True),
            # tf.keras.metrics.AUC(curve="PR", multi_label=True),
        ]

        # Compile model.
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            run_eagerly=config["run_eagerly"],
        )

        # Build model.
        model.build(input_shape)

    # Define label mapper.
    label_mapper = chiron.LabelMapper(config["label_map"])

    # Define one-hot encoder.
    one_hot_encoder = chiron.OneHotEncoder(config["num_classes"])

    # Get global batch size.
    global_batch_size = (
        config["batch_size_per_device"] * strategy.num_replicas_in_sync
    )

    # Define image whitener.
    whitener = chiron.PerBatchStandardWhitener()

    # Shuffle training data.
    shuffle_buffer_size = config["shuffle_buffer_size"]
    if shuffle_buffer_size is not None:
        train_dataset = train_dataset.shuffle(
            shuffle_buffer_size, seed=config.get("shuffle_seed")
        )

    # Prepare training data.
    train_dataset = prepare_dataset(
        train_dataset,
        label_mapper,
        one_hot_encoder,
        image_size,
        global_batch_size,
        whitener,
    )

    if args.val_file is not None:

        # Read validation dataset from TFRecord file.
        val_dataset = chiron.load_tfrecord(args.val_file)

        # Prepare validation data.
        val_dataset = prepare_dataset(
            val_dataset,
            label_mapper,
            one_hot_encoder,
            image_size,
            global_batch_size,
            whitener,
        )

    else:
        val_dataset = None

    # Define callbacks.
    callbacks = [
        chiron.Triangular2CLR(
            config["base_lr"], config["max_lr"], config["clr_step_size"]
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=config["early_stopping_patience"],
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(weights_dir, "epoch-{epoch:04}"),
            save_weights_only=True,
            save_freq="epoch",
            period=config["save_interval"],
        ),
        chiron.TensorBoardNamedLogs(exclude_names=["lr"], log_dir=log_dir),
        chiron.TensorBoardLearningRate(log_dir=log_dir, update_freq=1),
        chiron.SilentTerminateOnNaN(),
    ]

    try:
        # Train model.
        model.fit(
            x=train_dataset,
            epochs=config["num_epochs"],
            callbacks=callbacks,
            validation_data=val_dataset,
        )
    except KeyboardInterrupt:
        # Allow interruption.
        pass


if __name__ == "__main__":
    main()
