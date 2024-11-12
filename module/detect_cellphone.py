import cv2
import os
import threading
from ultralytics import YOLO

model_path = os.path.join('module', 'pretrained', 'yolov8s.pt')
model = YOLO(model_path)

def detectCellphone(img) -> list:
    try:
        results = model(img)
        detected = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() 
                classes = boxes.cls.cpu().numpy().astype(int)

                for bbox, conf, cls_id in zip(xyxy, confs, classes):
                    class_name = model.names[cls_id]
                    if class_name == 'cell phone':
                        detected.append({
                            'class': 'cellphone',
                            'confidence': float(conf),
                            'bbox': bbox.tolist()
                        })
        return detected
    
    except Exception as e:
        print(f'Error during detectCellphone : {e}')
        return []
