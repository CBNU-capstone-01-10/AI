import cv2
import os
import dlib
from scipy.spatial import distance
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join('module', 'pretrained', 'shape_predictor_68_face_landmarks.dat'))

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

def detectFacesAndEyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) == 0:
        return img, None
    
    face = faces[0]
    
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    left_ear = eyeAspectRatio(left_eye)
    right_ear = eyeAspectRatio(right_eye)

    avg_ear = (left_ear + right_ear) / 2

    eye_data = (avg_ear, left_ear, right_ear, left_eye, right_eye)

    return img, eye_data

def eyeAspectRatio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    EAR = (A + B) / (2.0 * C)
    return EAR

def checkDrowsiness(ear, ear_threshold):
    if ear < ear_threshold:
        return True

    return False
