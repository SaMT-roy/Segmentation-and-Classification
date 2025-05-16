import cv2
import numpy as np

# YOLO-specific preprocessing function
def preprocess_image(img, input_size):
    img_height, img_width = img.shape[:2]
    r = min(input_size[0]/img_width, input_size[1]/img_height)
    new_width, new_height = int(img_width * r), int(img_height * r)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
    return blob, r, (img_width, img_height)


# Process YOLOv11 output with NMS
def process_output(outputs, conf_threshold=0.2, nms_threshold=0.45, img_shape=None, ratio=1.0):
    output = outputs[0][0]
    boxes, confidences = [], []
 
    for idx in range(output.shape[1]):
        confidence = output[4, idx]
        if confidence >= conf_threshold:
            x, y, w, h = output[:4, idx]
            x1 = int((x - w / 2) / ratio)
            y1 = int((y - h / 2) / ratio)
            width, height = int(w / ratio), int(h / ratio)
 
            boxes.append([x1, y1, width, height])
            confidences.append(float(confidence))
 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = []
 
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append({
                'box': [x, y, x + w, y + h],
                'confidence': confidences[i]
            })
 
    return detections