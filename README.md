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

#### Response:

```json
{
  "detail": {
    "drowsy": false,
    "face_detected": false,
    "objects_detected": [
      {
        "bbox": [
          87.56160736083984,
          80.62358093261719,
          227.22669982910156,
          173.17254638671875
        ],
        "class": "cell phone",
        "confidence": 0.7384440898895264
      }
    ],
    "person_detected": true
  },
  "label": [
    "cell phone"
  ],
  "safe_driving": false
}
```

- `person_detected` (boolean): Indicates if a person was detected in the image.
- `face_detected` (boolean): Indicates if a face was detected in the image.
- `drowsy` (boolean): Indicates if the detected person appears drowsy.
- `objects_detected` (array): A list of detected objects, each containing:
  - `class`: The type of detected object (e.g., "cigarette").
  - `confidence`: The confidence score of the detection (between 0 and 1).
  - `bbox`: The bounding box coordinates for the detected object, formatted as `[x1, y1, x2, y2]`.
