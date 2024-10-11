import cv2
import os
from ultralytics import YOLO

model_path = os.path.join('module', 'pretrained', 'yolov8l.pt')
model = YOLO(model_path)

def detectObjects(img):
    """
    Detect cigarettes and handphones in the provided image using YOLOv8.

    Args:
        img (numpy.ndarray): The input image in which to detect objects.

    Returns:
        list: A list of dictionaries containing detection results with keys:
              'class' (str): The class name ('cigarette' or 'handphone'),
              'confidence' (float): The confidence score,
              'bbox' (list): The bounding box coordinates [x1, y1, x2, y2].
    """
    results = model(img)

    detected_objects = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = boxes.conf.cpu().numpy()  # Confidence scores
            classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs

            for bbox, conf, cls_id in zip(xyxy, confs, classes):
                class_name = model.names[cls_id]
                if class_name in ['cigarette', 'cell phone']:
                    detected_objects.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': bbox.tolist()
                    })
    return detected_objects
