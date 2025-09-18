# src/predict.py
import os
import sys
import json
import argparse
import cv2
import numpy as np
import tensorflow as tf
from src import utils
from src.model import build_crnn

def load_meta(models_dir="models"):
    with open(os.path.join(models_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    charset = meta["charset"]
    idx_to_char = {i+1: c for i, c in enumerate(list(charset))}
    return meta, idx_to_char

def greedy_decode_batch(preds, idx_to_char):
    from src.utils import ctc_greedy_decode
    return ctc_greedy_decode(preds, idx_to_char)

def predict_page(page_path, models_dir="models", debug=False):
    meta, idx_to_char = load_meta(models_dir)
    IMG_H = meta["img_h"]
    IMG_W = meta["img_w"]
    vocab_size = len(meta["charset"])

    model = build_crnn(img_height=IMG_H, img_width=IMG_W, rnn_units=256, vocab_size=vocab_size)
    model.load_weights(os.path.join(models_dir, "crnn_weights.h5"))

    gray = utils.load_gray(page_path)
    boxes = utils.segment_page_lines(gray, debug_out="output/seg_debug.png" if debug else None)
    paragraphs = utils.reconstruct_paragraphs(boxes)

    results = []
    tmp_dir = "output/tmp_lines"
    os.makedirs(tmp_dir, exist_ok=True)

    for para in paragraphs:
        para_lines = []
        for i, box in enumerate(para):
            y, x, h, w = box
            line_img = gray[y:y+h, x:x+w]
            tmp_path = os.path.join(tmp_dir, f"line_{y}_{x}.png")
            cv2.imwrite(tmp_path, line_img)
            proc = utils.resize_and_pad(line_img, target_h=IMG_H, max_w=IMG_W)
            proc = np.expand_dims(proc, 0).astype(np.float32)
            pred = model.predict(proc)  # [1, T, C]
            text = greedy_decode_batch(pred, idx_to_char)[0].strip()
            para_lines.append(text)
        # join lines with single space per original request to preserve formatting line-by-line
        results.append("\n".join(para_lines))
    # join paragraphs with a blank line
    final_text = "\n\n".join(results)
    return final_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to page image")
    parser.add_argument("--models", default="models", help="Models directory")
    args = parser.parse_args()
    out = predict_page(args.image, models_dir=args.models, debug=True)
    print(out)
