from flask import Flask, request, jsonify
import cv2
import numpy as np
from dataclasses import dataclass
from module.eye import detectFacesAndEyes, checkDrowsiness
from module.detect_person import detectNearestPerson
from module.detect_cellphone import detectCellphone
from module.detect_cigarette import detectCigarette

app = Flask(__name__)

"""class"""
@dataclass
class Config():
    EAR_THRESHOLD: float = 0.2
    CONSECUTIVE_DROWSY_FRAMES: int = 2
    CONSECUTIVE_OBJECT_FRAMES: int = 1
    OBJECT_CONFIDENCE: float = 0.6

class Counter:
    def __init__(self):
        self.drowsy_value = 0
        self.cigarette_value = 0
        self.cellphone_value = 0

    def increment_drowsy(self):
        self.drowsy_value += 1

    def increment_cigarette(self):
        self.cigarette_value += 1

    def increment_cellphone(self):
        self.cellphone_value += 1
    
    def reset_drowsy(self):
        self.drowsy_value = 0
    
    def reset_cigarette(self):
        self.cigarette_value = 0
    
    def reset_cellphone(self):
        self.cellphone_value = 0

CONFIG = Config()
COUNTER = Counter()

"""route"""
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    ear_threshold = request.form.get('ear_threshold', type=float)
    consecutive_drowsy_frames = request.form.get('consecutive_drowsy_frames', type=int)
    consecutive_object_frames = request.form.get('consecutive_object_frames', type=int)
    object_confidence = request.form.get('object_confidence', type=float)
    
    if ear_threshold is not None:
        CONFIG.EAR_THRESHOLD = ear_threshold
    
    if consecutive_drowsy_frames is not None:
        CONFIG.CONSECUTIVE_DROWSY_FRAMES = consecutive_drowsy_frames
        
    if consecutive_object_frames is not None:
        CONFIG.CONSECUTIVE_OBJECT_FRAMES = consecutive_object_frames

    if object_confidence is not None:
        CONFIG.OBJECT_CONFIDENCE = object_confidence

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    ### Person
    
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
    
    ## Ready Data
    _, eye_data = detectFacesAndEyes(person_img)
    cellphone_data = detectCellphone(img)
    cigarette_data = detectCigarette(img)
    object_data = cellphone_data + cigarette_data
    
    cellphone_detected = False
    cigarette_detected = False
    drowsy_detected = False

    response_data = {
        'safe_driving': True,
        'label': [],
        'detail': {
            'person_detected': True,
            'face_detected': False,
            'drowsy': False,
            'objects_detected': object_data,
        }
    }
    
    ### Obj
    
    for obj in object_data:
        if (obj['class'] == 'cellphone') and (obj['confidence'] > CONFIG.OBJECT_CONFIDENCE):
            cellphone_detected = True
        
        if (obj['class'] == 'cigarette') and (obj['confidence'] > CONFIG.OBJECT_CONFIDENCE):
            cigarette_detected = True
    
    if cellphone_detected:
        COUNTER.increment_cellphone()
    else:
        COUNTER.reset_cellphone()
    
    if cigarette_detected:
        COUNTER.increment_cigarette()
    else:
        COUNTER.reset_cigarette()
    
    if COUNTER.cellphone_value > CONFIG.CONSECUTIVE_OBJECT_FRAMES:
        response_data['label'].append('cellphone')
        response_data['safe_driving'] = False
        
    if COUNTER.cigarette_value > CONFIG.CONSECUTIVE_OBJECT_FRAMES:
        response_data['label'].append('cigarette')
        response_data['safe_driving'] = False
        
    response_data['detail']['cellphone_count'] = COUNTER.cellphone_value
    response_data['detail']['cigarette_count'] = COUNTER.cigarette_value
        
    ### Eye
    if eye_data:
        response_data['detail']['face_detected'] = True

        avg_ear, left_ear, right_ear, left_eye, right_eye = eye_data

        drowsy_detected = checkDrowsiness(
            avg_ear, COUNTER.drowsy_value, CONFIG.EAR_THRESHOLD
        )
        
        if drowsy_detected:
            COUNTER.increment_drowsy()
            
        if COUNTER.drowsy_value > CONFIG.CONSECUTIVE_DROWSY_FRAMES:
            drowsy = True
            response_data['detail']['drowsy'] = drowsy
            response_data['safe_driving'] = False

        response_data['detail']['drowsy_count'] = COUNTER.drowsy_value
        response_data['detail']['avg_ear'] = avg_ear
        response_data['detail']['left_ear'] = left_ear
        response_data['detail']['right_ear'] = right_ear
        
    if drowsy_detected == False:
        COUNTER.reset_drowsy()

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
