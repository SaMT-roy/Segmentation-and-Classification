import onnxruntime as ort
import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- load ONNX model ---
session = ort.InferenceSession(
    "latte_art_classifier.onnx",
    providers=["CPUExecutionProvider"]  # or CUDAExecutionProvider
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
class_names = ['bad latte art', 'good latte art']

def preprocess(img_bgr):
    """
    img_bgr: uint8 BGR image from cv2.imread
    returns: (1, 3, 512, 512) float32 NCHW
    """

    # BGR → RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize (bilinear, same as torchvision default)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # uint8 → float32 [0,1]
    img = img.astype(np.float32) / 255.0

    # Normalize
    img = (img - MEAN) / STD

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dim
    img = np.expand_dims(img, axis=0)

    return img


def predict_image_onnx(img_path):
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, "Image not found"

    input_tensor = preprocess(img_bgr)

    logits = session.run(
        [output_name],
        {input_name: input_tensor}
    )[0]

    # softmax
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    cls = int(np.argmax(probs, axis=1)[0])
    conf = float(probs[0, cls])

    return class_names[cls], conf, probs[0]



# pth = os.path.join(folder, p)
# cls, conf, probs = predict_image_onnx(pth)
# print(cls, conf, probs)

# img = cv2.imread(pth)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
