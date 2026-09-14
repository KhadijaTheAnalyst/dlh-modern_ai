#!/usr/bin/env python3
"""
Task 4: Knowledge Transfer — Taming the 101

Builds, trains, and saves an image classifier for the Caltech101
dataset (101 object classes + 1 background class = 102 classes)
using transfer learning with a pretrained CNN backbone.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 102

EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 15
UNFREEZE_LAYERS = 40


def build_data_augmentation():
    """Builds a small data augmentation pipeline."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(0.15, seed=42),
        keras.layers.RandomZoom(0.15, seed=42),
        keras.layers.RandomContrast(0.1, seed=42),
    ], name="data_augmentation")


def build_feature_extractor():
    """Loads a frozen, pretrained EfficientNetB0 backbone."""
    base_model = keras.applications.EfficientNetB0(
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        pooling="avg",
    )
    base_model.trainable = False
    return base_model


def prepare_datasets():
    """
    Loads the Caltech101 dataset via tensorflow_datasets, and builds
    preprocessed, batched, augmented tf.data pipelines for training
    and validation.
    """
    (train_raw, val_raw), info = tfds.load(
        "caltech101",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    augmentation = build_data_augmentation()
    preprocess_input = keras.applications.efficientnet.preprocess_input

    def prepare(x, y, training):
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32)
        if training:
            x = augmentation(x)
        x = preprocess_input(x)
        return x, y

    train_ds = (
        train_raw
        .map(lambda x, y: prepare(x, y, training=True),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw
        .map(lambda x, y: prepare(x, y, training=False),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


def build_model(base_model):
    """Attaches a classification head to the frozen base model."""
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def unfreeze_top_layers(base_model, n_layers):
    """Unfreezes the last n_layers of base_model, freezes the rest."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    # Keep BatchNorm layers frozen even among the "unfrozen" ones,
    # to avoid destabilizing pretrained statistics on a small dataset.
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


def train_transfer_model():
    """
    Builds, trains, and saves an image classifier for Caltech101
    using two-phase transfer learning (frozen head training,
    followed by fine-tuning of the top backbone layers).

    Saves the trained model to 'caltech101_model.h5'.
    """
    train_ds, val_ds = prepare_datasets()

    base_model = build_feature_extractor()
    model = build_model(base_model)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            "caltech101_model.h5", monitor="val_accuracy",
            save_best_only=True
        ),
    ]

    # --- Phase 1: train the classification head only ---
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_HEAD, callbacks=callbacks
    )

    # --- Phase 2: unfreeze top backbone layers and fine-tune ---
    unfreeze_top_layers(base_model, UNFREEZE_LAYERS)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_FINETUNE, callbacks=callbacks
    )

    model.save("caltech101_model.h5")
    return model


if __name__ == "__main__":
    train_transfer_model()
