# Driver Detection API

This API is designed to detect a person, identify objects such as cigarettes and cellphones, and monitor driver drowsiness within an uploaded image. Using a YOLO-based model for object detection and custom configurations, it is optimized for Docker deployment.

## System Requirements

- **Python**: 3.8
- **CUDA**: 12.4 (for GPU-accelerated processing)

## Quickstart

To quickly start the API, pull the Docker image and launch the container:

```bash
docker pull hongbeomsun/driver-detection:1.4.0

docker run -p 9999:9999 --name driver-detection-container hongbeomsun/driver-detection:1.4.0
```

The API service will start and be accessible at `http://localhost:9999`.

## API Documentation

### **POST** `/detect`

This endpoint accepts an image and analyzes it to identify a person, detect objects such as cellphones or cigarettes, and check for driver drowsiness.

#### Request Format

- **URL**: `/detect`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (required): The image file to be analyzed (supports JPEG, PNG).
  - `ear_threshold` (optional): Custom threshold for Eye Aspect Ratio (EAR) in drowsiness detection (default: 0.2).
  - `consecutive_drowsy_frames` (optional): Number of consecutive frames needed for a confirmed drowsiness detection (default: 2).
  - `consecutive_object_frames` (optional): Minimum frames needed to flag objects like cellphones or cigarettes (default: 1).
  - `object_confidence` (optional): Minimum confidence score for object detection (default: 0.5).

#### Example Request

```http
POST /detect
Host: localhost:9999
Content-Type: multipart/form-data
```

#### Example Response

```json
{
  "safe_driving": false,
  "label": ["cigarette"],
  "detail": {
    "person_detected": true,
    "face_detected": true,
    "drowsy": false,
    "drowsy_count": 0,
    "avg_ear": 0.3,
    "left_ear": 0.32,
    "right_ear": 0.28,
    "cellphone_count": 0,
    "cigarette_count": 4,
    "objects_detected": [
      {
        "class": "cigarette",
        "confidence": 0.738,
        "bbox": [87.56, 80.62, 227.22, 173.17]
      }
    ]
  }
}
```

#### Response Fields

- `safe_driving`: Indicates if it's safe to drive, based on drowsiness or object detection.
- `label`: List of detected objects (e.g., "cigarette", "cellphone") that could impair driving safety.
- `detail`: Contains specific detection details:
  - `person_detected`: Boolean indicating if a person is detected.
  - `face_detected`: Boolean indicating if a face is detected.
  - `drowsy`: Boolean indicating if drowsiness is detected.
  - `drowsy_count`: Count of consecutive frames detecting drowsiness.
  - `avg_ear`: Average Eye Aspect Ratio (EAR) across both eyes.
  - `left_ear`, `right_ear`: EAR values for the left and right eyes.
  - `cellphone_count`, `cigarette_count`: Counts of frames where cellphones or cigarettes were detected consecutively.
  - `objects_detected`: List of detected objects, each containing:
    - `class`: Object type (e.g., "cellphone", "cigarette").
    - `confidence`: Confidence score of the detection.
    - `bbox`: Bounding box for the detected object, given as `[x1, y1, x2, y2]`.

## Configuration

You can adjust the detection parameters to suit specific needs:

- `EAR_THRESHOLD`: Threshold for drowsiness detection based on Eye Aspect Ratio (default: 0.2).
- `CONSECUTIVE_DROWSY_FRAMES`: Minimum frames required for a confirmed drowsiness detection (default: 2).
- `CONSECUTIVE_OBJECT_FRAMES`: Minimum frames for detecting objects like cellphones and cigarettes (default: 1).
- `OBJECT_CONFIDENCE`: Confidence threshold for object detection (default: 0.5).

