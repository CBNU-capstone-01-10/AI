# Driver Detection API

This API detects a person, objects (like cigarettes and cellphones), and checks for driver drowsiness in the provided image. It uses YOLO models for object detection and can be run in a Docker container.

## System Requirements

- **Python**: 3.8
- **CUDA**: 12.4

## Quickstart

Pull the Docker image and run the container:

```bash
docker pull hongbeomsun/driver-detection:1.2.0

docker run -p 9999:9999 --name driver-detection-container hongbeomsun/driver-detection:1.2.0
```

This will start the driver detection service on `http://localhost:9999`.

## API Documentation

### **POST** `/detect`

Detects a person, objects (like cigarettes and cellphones), and checks for driver drowsiness in the provided image.

#### Request:

```http
POST /detect
Host: localhost:9999
Content-Type: multipart/form-data
```

- **URL**: `/detect`
- **Method**: `POST`
- **Request Format**: `multipart/form-data`
- **Request Parameters**:
  - `image`: The image file to analyze (supported formats: JPEG, PNG).
  - `ear_threshold` (optional, float): EAR (Eye Aspect Ratio) threshold for drowsiness detection. Default: 0.2.
  - `consecutive_frames` (optional, int): Minimum consecutive frames for detecting objects (e.g., cigarettes, cellphones). Default: 3.
  - `object_confidence` (optional, float): Confidence threshold for object detection. Default: 0.5.

#### Response:

```json
{
  "detail": {
    "drowsy": false,
    "face_detected": true,
    "drowsy_count": 0,
    "avg_ear": 0.3,
    "left_ear": 0.32,
    "right_ear": 0.28,
    "counter": 0,
    "cellphone_count": 0,
    "cigarette_count": 4,
    "objects_detected": [
      {
        "bbox": [
          87.56160736083984,
          80.62358093261719,
          227.22669982910156,
          173.17254638671875
        ],
        "class": "cigarette",
        "confidence": 0.7384440898895264
      }
    ],
    "person_detected": true
  },
  "label": [
    "cigarette"
  ],
  "safe_driving": false
}
```

#### Response Fields:

- `person_detected` (boolean): Indicates if a person was detected in the image.
- `face_detected` (boolean): Indicates if a face was detected in the image.
- `drowsy` (boolean): Indicates if the detected person appears drowsy.
- `drowsy_count` (int): The number of consecutive frames drowsiness has been detected.
- `avg_ear` (float): Average Eye Aspect Ratio (EAR) calculated from both eyes.
- `left_ear`, `right_ear` (float): The EAR values for the left and right eyes, respectively.
- `counter` (int): The number of consecutive frames that drowsiness has been detected.
- `cellphone_count`, `cigarette_count` (int): The number of consecutive frames a cellphone or cigarette has been detected.
- `objects_detected` (array): A list of detected objects, each containing:
  - `class`: The type of detected object (e.g., "cellphone", "cigarette").
  - `confidence`: The confidence score of the detection (between 0 and 1).
  - `bbox`: The bounding box coordinates for the detected object, formatted as `[x1, y1, x2, y2]`.
- `label` (array): List of object classes detected (e.g., "cigarette", "cellphone"). Objects are added to this list only after being detected in enough consecutive frames.
- `safe_driving` (boolean): Indicates whether it is safe to drive, based on the detection of objects like cellphones, cigarettes, or drowsiness.

## Configuration Options

- `EAR_THRESHOLD`: Adjust the threshold for detecting drowsiness based on the Eye Aspect Ratio (default: `0.2`).
- `CONSECUTIVE_FRAMES`: The minimum number of consecutive frames required to detect an object (like a cigarette or cellphone) before it is flagged (default: `3`).
- `OBJECT_CONFIDENCE`: Confidence threshold for object detection (default: `0.5`).
