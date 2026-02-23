import numpy as np
import glob
import tensorflow as tf
import random

TIME_STAMP = 24
FEATURES = 16
FUTURE_HORIZON = 6
TARGETS = 4

BATCH_SIZE = 64

x_files = sorted(glob.glob("X_train_part_*.npy"))
y_files = sorted(glob.glob("y_train_part_*.npy"))


def chunk_loader(x_files, y_files):
    file_pairs = list(zip(x_files, y_files))
    # shuffle files so we do not learn 1 city at time .
    random.shuffle(file_pairs)

    for xf, yf in file_pairs:
        X = np.load(xf)
        Y = np.load(yf)
        yield X.astype(np.float32), Y.astype(np.float32)


def gen():
    for X, Y in chunk_loader(x_files, y_files):
        for i in range(len(X)):
            yield X[i], Y[i]


# =============   DATASET =============

dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(TIME_STAMP, FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(FUTURE_HORIZON, TARGETS), dtype=tf.float32)
    )
)


def split_targets(x, y):
    y_reg = tf.cast(y[:, :3], tf.float32)
    y_wmo = tf.cast(y[:, 3], tf.int32)  # classification cast to int
    return x, {"regression": y_reg, "classification": y_wmo}


dataset = (
    dataset.map(
        split_targets, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)


val_dataset = dataset.take(100)
train_dataset = dataset.skip(100)


#  MODEL

inputs = tf.keras.Input(shape=(TIME_STAMP, FEATURES))
# drop out avoid overfitting
x = tf.keras.layers.LSTM(128, dropout=0.2)(inputs)

x = tf.keras.layers.RepeatVector(FUTURE_HORIZON)(x)
# decoder
x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)

# regresion :
regression_output = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(3),
    name="regression"
)(x)


# classification wmo :

classification_output = tf.keras.layers.TimeDistributed(
    # five possible weather classes
    tf.keras.layers.Dense(7, activation="softmax"),
    name="classification"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=[regression_output, classification_output]
)


# compile the model  - how to use and work

model.compile(
    optimizer="adam",
    loss={
        "regression": "mse",
        "classification": "sparse_categorical_crossentropy"
    },
    metrics={  # for reporting performance
        "regression": "mae",
        "classification": "accuracy"
    }
)

model.summary()


#   TRAIN

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "weather_lstm_6h_checkpoint.keras",
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=1, verbose=1
)


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[checkpoint, early_stop, reduce_lr]
)


model.save("weather_lstm_6h_prediction.keras")
print("model saved")
