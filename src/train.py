import argparse
import os

import tensorflow as tf

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


def prepare_dataset(dataset, label_mapper, image_size, batch_size, whitener):
    return (
        dataset.map(label_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        .map(chiron.Resizer(image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .map(chiron.Repeater(3, axis=2), num_parallel_calls=tf.data.AUTOTUNE)
        .map(
            chiron.ShapeSetter(image_size + [3]),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .map(whitener, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def main():

    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)

    log_dir = os.path.join(output_dir, "log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    weights_dir = os.path.join(output_dir, "weights")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    devices = args.devices
    if devices is not None:
        chiron.set_visible_gpus(*devices)

    config = parse_config_file(args.config_file)

    chiron.save_yaml(os.path.join(output_dir, "config.yaml"), config)

    reg_scale = config.get("reg_scale")
    regularizer = (
        None if reg_scale is None else tf.keras.regularizers.l2(reg_scale)
    )

    image_size = tf.TensorShape(config["image_size"])
    input_shape = [None] + image_size + [3]

    strategy = chiron.get_distribution_strategy()

    with strategy.scope():

        model = chiron.get_model(
            config["num_classes"], regularizer=regularizer
        )

        chiron.save_model(os.path.join(output_dir, "model.json"), model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
            run_eagerly=config["run_eagerly"],
        )

        model.build(input_shape)

    train_dataset = chiron.load_tfrecord(args.train_file)

    shuffle_buffer_size = config["shuffle_buffer_size"]
    if shuffle_buffer_size is not None:
        train_dataset = train_dataset.shuffle(
            shuffle_buffer_size, seed=config.get("shuffle_seed")
        )

    label_mapper = chiron.LabelMapper(config["label_map"])

    global_batch_size = (
        config["batch_size_per_device"] * strategy.num_replicas_in_sync
    )

    whitener = chiron.PerImageStandardWhitener()

    train_dataset = prepare_dataset(
        train_dataset, label_mapper, image_size, global_batch_size, whitener
    )

    if args.val_file is not None:
        val_dataset = prepare_dataset(
            chiron.load_tfrecord(args.val_file),
            label_mapper,
            image_size,
            global_batch_size,
            whitener,
        )
    else:
        val_dataset = None

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            patience=config["early_stopping_patience"], verbose=1
        ),
    )

    weights_file = os.path.join(weights_dir, chiron.get_weights_path())

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=config["reduce_lr_patience"], verbose=1
        ),
        early_stopping,
        tf.keras.callbacks.ModelCheckpoint(
            weights_file, save_weights_only=True, period=config["save_period"]
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    try:
        model.fit(
            x=train_dataset,
            epochs=config["num_epochs"],
            callbacks=callbacks,
            validation_data=val_dataset,
        )
    except KeyboardInterrupt:
        pass

    model.save_weights(weights_file.format(epoch=early_stopping.stopped_epoch))


if __name__ == "__main__":
    main()
