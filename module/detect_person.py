import os
from ultralytics import YOLO

model_path = os.path.join('module', 'pretrained', 'yolov8l.pt')
model = YOLO(model_path)

def detectNearestPerson(img):
    """
    Detects the nearest person in the provided image using YOLOv8.

    Args:
        img (numpy.ndarray): The input image in which to detect the nearest person.

    Returns:
        dict: A dictionary containing detection result with keys:
              'person_detected': bool indicating if a person was detected,
              'bbox': The bounding box coordinates [x1, y1, x2, y2] of the nearest person.
    """
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

    # Identify the nearest (largest) person
    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    person_bboxes.sort(key=lambda x: bbox_area(x['bbox']), reverse=True)
    nearest_person = person_bboxes[0]

    return {'person_detected': True, 'bbox': nearest_person['bbox']}
