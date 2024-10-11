from flask import Flask, request, jsonify
import cv2
import numpy as np
from dataclasses import dataclass
from module.eye import detectFacesAndEyes, checkDrowsiness
from module.detect_person import detectNearestPerson
from module.detect_object import detectObjects

app = Flask(__name__)

"""class"""
@dataclass
class Config():
    EAR_THRESHOLD: float = 0.25
    CONSECUTIVE_FRAMES: int = 10

class Counter:
    def __init__(self):
        self.value = 0

config = Config()
counter = Counter()

"""route"""
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    person_detection = detectNearestPerson(img)

    if not person_detection['person_detected']:
        return jsonify({'person_detected': False}), 200

    x1, y1, x2, y2 = map(int, person_detection['bbox'])
    
    height, width = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    person_img = img[y1:y2, x1:x2].copy()

    _, eye_data = detectFacesAndEyes(person_img)

    object_data = detectObjects(person_img)

    response_data = {
        'person_detected': True,
        'face_detected': False,
        'drowsy': False,
        'objects_detected': object_data,
    }

    if len(eye_data) > 0:
        response_data['face_detected'] = True

        for left_ear, right_ear, left_eye, right_eye in eye_data:
            avg_ear = (left_ear + right_ear) / 2

            counter.value, drowsy = checkDrowsiness(
                avg_ear, counter.value, config.EAR_THRESHOLD, config.CONSECUTIVE_FRAMES
            )

            response_data['avg_ear'] = avg_ear
            response_data['counter'] = counter.value
            response_data['left_ear'] = left_ear
            response_data['right_ear'] = right_ear

            if drowsy:
                response_data['drowsy'] = True

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
