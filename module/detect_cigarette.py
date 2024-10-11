import cv2
import os
from ultralytics import YOLO

def get_model_path(test):
    if test:
        model_path = os.path.abspath(os.path.join('.', 'pretrained', 'yolov8n_cigarette.pt'))
    else:
        model_path = os.path.abspath(os.path.join('module', 'pretrained', 'yolov8n_cigarette.pt'))
    return model_path

def detect_cigarette(source, test=False):
    """
    Perform inference using the trained YOLOv8 model.

    Args:
        source (str): Path to the input image or video file, or a webcam index (0, 1, etc.).
        test (bool): If True, will display the image or video with detected results.

    Returns:
        results: YOLOv8 inference results.
    """
    model_path = get_model_path(test)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(model_path)
    
    results = model.predict(source, conf=0.7)

    if test:
        for result in results:
            img = result.plot()
            cv2.imshow('Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return results

if __name__ == '__main__':
    source_path = '../test/images/test_cigarette2.jpg'
    
    try:
        results = detect_cigarette(source_path, test=True)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
