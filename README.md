Python 3.8
CUDA 12.4


```
docker pull hongbeomsun/driver-detection:1.1.0

docker run -p 9999:9999 --name driver-detection-container hongbeomsun/driver-detection:1.1.0
```

## API
### POST /detect

Detects a person, objects (like cigarette and cellphone), and checks for driver drowsiness in the provided image.

URL: /detect
Method: POST
Request Format: multipart/form-data
Request Parameters:
    image: The image file to analyze (JPEG, PNG, etc.).

Response:
    person_detected (boolean): Whether a person was detected in the image.
    face_detected (boolean): Whether a face was detected in the image.
    drowsy (boolean): Whether the detected person is drowsy.
    objects_detected (list of objects): Detected objects in the image (e.g., cigarette, cellphone). Each object includes:
    class: Type of object (e.g., cigarette).
    confidence: Confidence score of the object detection.
    bbox: Bounding box coordinates [x1, y1, x2, y2].

Response Example:
    {
        "person_detected": true,
        "face_detected": false,
        "drowsy": false,
        "objects_detected": [
            {
            "class": "cigarette",
            "confidence": 0.847,
            "bbox": [1256.06, 726.22, 1547.11, 931.78]
            }
        ]
    }
