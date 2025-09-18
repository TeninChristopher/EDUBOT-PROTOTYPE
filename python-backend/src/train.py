# src/train.py
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from src import utils
from src.model import build_crnn, ctc_loss
from tensorflow import keras
from keras import layers

# ---------- CONFIG ----------
IMG_H = utils.IMG_HEIGHT
IMG_W = utils.IMG_MAX_WIDTH
BATCH_SIZE = 8
EPOCHS = 30
MAX_LABEL_LEN = 256

# Character set (tweak to your dataset)
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()-/ "


def build_vocabulary(charset):
    # model indices: 1..V map to characters; 0 reserved for CTC blank
    char_list = list(charset)
    char_to_idx = {c: i+1 for i, c in enumerate(char_list)}  # 1-based
    idx_to_char = {i+1: c for i, c in enumerate(char_list)}
    return char_to_idx, idx_to_char


def text_to_labels(text, char_to_idx):
    return [char_to_idx[c] for c in text if c in char_to_idx]


def load_dataset(data_dir, char_to_idx):
    images_dir = os.path.join(data_dir, "images")
    labels_txt = os.path.join(data_dir, "labels.txt")
    fnames = []
    texts = []
    with open(labels_txt, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            fname, text = ln.strip().split("\t", 1)
            path = os.path.join(images_dir, fname)
            fnames.append(path)
            texts.append(text)
    return fnames, texts


def tf_data_generator(fnames, texts, char_to_idx, batch_size=BATCH_SIZE):
    def gen():
        for p, txt in zip(fnames, texts):
            yield p.encode('utf-8'), txt.encode('utf-8')
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.string, tf.string))
    def map_fn(p_bytes, t_bytes):
        p = tf.strings.substr(p_bytes, 0, tf.strings.length(p_bytes))
        p = tf.py_function(lambda x: utils.load_gray(x.decode('utf-8')), inp=[p_bytes], Tout=tf.uint8)
        p.set_shape([None, None])
        img = tf.py_function(lambda x: utils.resize_and_pad(x), inp=[p], Tout=tf.float32)
        img.set_shape([IMG_H, IMG_W, 1])
        text = tf.strings.unicode_decode(t_bytes, input_encoding='UTF-8')
        return {"image": img, "label": text}
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size,
                         padded_shapes={"image": (IMG_H, IMG_W, 1), "label": (None,)},
                         padding_values={"image": 1.0, "label": 0})
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train(data_dir, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    char_to_idx, idx_to_char = build_vocabulary(CHARSET)
    fnames, texts = load_dataset(data_dir, char_to_idx)

    # Convert text to integer sequences using char_to_idx for CTC training
    labels_seq = [text_to_labels(t, char_to_idx) for t in texts]
    max_label = max(len(s) for s in labels_seq)
    print("Samples:", len(fnames), "Max label len:", max_label)

    model = build_crnn(img_height=IMG_H, img_width=IMG_W, rnn_units=256, vocab_size=len(char_to_idx))
    model.summary()

    # optimizer and custom training loop with CTC
    optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(images, labels, label_lens):
        batch_size = tf.shape(images)[0]
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)  # [B, T, C]
            input_len = tf.fill([batch_size, 1], tf.shape(y_pred)[1])
            loss = tf.reduce_mean(ctc_loss(labels, y_pred, input_len, tf.expand_dims(label_lens, 1)))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Build generator datasets (we will use a simpler py loader for labels lengths)
    # For reliability, use a small Python loader
    for epoch in range(1, EPOCHS+1):
        total_loss = []
        for i in range(0, len(fnames), BATCH_SIZE):
            batch_files = fnames[i:i+BATCH_SIZE]
            batch_texts = texts[i:i+BATCH_SIZE]
            imgs = []
            label_seqs = []
            label_lens = []
            for p, t in zip(batch_files, batch_texts):
                img = utils.load_gray(p)
                img = utils.resize_and_pad(img)
                imgs.append(img)
                lab = text_to_labels(t, char_to_idx)
                label_seqs.append(lab)
                label_lens.append(len(lab))
            imgs = np.stack(imgs, axis=0).astype(np.float32)
            # pad labels to max in batch
            maxl = max(label_lens) if label_lens else 1
            padded = np.zeros((len(label_seqs), maxl), dtype=np.int32)
            for r, seq in enumerate(label_seqs):
                padded[r, :len(seq)] = seq
            loss = train_step(imgs, padded, np.array(label_lens, dtype=np.int32))
            total_loss.append(float(loss))
        print(f"Epoch {epoch}/{EPOCHS} - loss: {np.mean(total_loss):.4f}")

    # Save weights and charset mapping
    weights_path = os.path.join(out_dir, "crnn_weights.h5")
    model.save_weights(weights_path)
    meta = {"charset": CHARSET, "img_h": IMG_H, "img_w": IMG_W}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("Saved weights to", weights_path)
