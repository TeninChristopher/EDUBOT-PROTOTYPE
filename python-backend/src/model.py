# src/model.py
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

def build_crnn(img_height=64, img_width=1024, rnn_units=256, vocab_size=80):
    """
    Build a CRNN that outputs softmax over (vocab_size + 1) classes where 0 is CTC-blank.
    """
    inp = layers.Input(shape=(img_height, img_width, 1), name="image")
    x = inp

    # CNN extractor
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)  # H/2 W/2
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)  # H/4 W/4

    # permute / reshape to sequence: time dimension = width//4
    # after pooling shapes: [B, H', W', C]
    x = layers.Permute((2,1,3))(x)     # [B, W', H', C]
    x = layers.Reshape((img_width // 4, -1))(x)  # [B, T, F]

    # RNN
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)

    # final softmax (vocab+1 for blank)
    out = layers.Dense(vocab_size + 1, activation="softmax", name="y_pred")(x)

    model = models.Model(inputs=inp, outputs=out, name="crnn")
    return model

def ctc_loss(y_true, y_pred, input_length, label_length):
    # y_true shape: [B, L], y_pred: [B, T, C]
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
