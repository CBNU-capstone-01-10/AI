import os
from ultralytics import YOLO

model_path = os.path.join('module', 'pretrained', 'yolov8n.pt')
model = YOLO(model_path)

def detectNearestPerson(img):
    results = model(img)
    person_bboxes = []

    for result in results:
        boxes = result.boxes 
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = boxes.conf.cpu().numpy()  # Confidence scores
            classes = boxes.cls.cpu().numpy().astype(int)  # Class IDs

            for bbox, conf, cls_id in zip(xyxy, confs, classes):
                class_name = model.names[cls_id]
                if class_name == 'person':
                    detection = {
                        'confidence': float(conf),
                        'bbox': bbox.tolist()
                    }
                    person_bboxes.append(detection)

    if not person_bboxes:
        return {'person_detected': False, 'bbox': None}

    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    person_bboxes.sort(key=lambda x: bbox_area(x['bbox']), reverse=True)
    nearest_person = person_bboxes[0]

    return {'person_detected': True, 'bbox': nearest_person['bbox']}
