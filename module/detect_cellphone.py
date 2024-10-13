import cv2
import os
from ultralytics import YOLO

model_path = os.path.join('module', 'pretrained', 'yolov8n.pt')
model = YOLO(model_path)

def detectCellphone(img):
    results = model(img)
    detected = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = boxes.conf.cpu().numpy()  # Confidence scores
            classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs

            for bbox, conf, cls_id in zip(xyxy, confs, classes):
                class_name = model.names[cls_id]
                if class_name == 'cell phone':
                    detected.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': bbox.tolist()
                    })
    return detected
