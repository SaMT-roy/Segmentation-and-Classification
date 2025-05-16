import cv2
import numpy as np

# Imagenet means/std you used
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_img_array(img: np.ndarray, apply_clahe=True) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256))            # W,H
    img = img.astype(np.float32) / 255.0
    img = (img - IMNET_MEAN) / IMNET_STD
    img = np.transpose(img, (2, 0, 1))           # CHW
    return np.expand_dims(img, 0)                # [1,3,256,128]

def predict_from_crop(sess, crop_img: np.ndarray) -> str:
    inp = preprocess_img_array(crop_img)
    logits = sess.run(None, {"images": inp.astype(np.float32)})[0]  # [1,2]
    pred = int(logits.argmax(1)[0])
    return "employee" if pred == 1 else "customer"