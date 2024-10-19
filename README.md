# Driver Detection API

This API detects a person, objects (like cigarettes and cellphones), and checks for driver drowsiness in the provided image. It uses YOLO models for object detection and can be run in a Docker container.

## System Requirements

- **Python**: 3.8
- **CUDA**: 12.4

## Quickstart

Pull the Docker image and run the container:

```bash
docker pull hongbeomsun/driver-detection:1.1.0

docker run -p 9999:9999 --name driver-detection-container hongbeomsun/driver-detection:1.1.0
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
```

- `person_detected` (boolean): Indicates if a person was detected in the image.
- `face_detected` (boolean): Indicates if a face was detected in the image.
- `drowsy` (boolean): Indicates if the detected person appears drowsy.
- `objects_detected` (array): A list of detected objects, each containing:
  - `class`: The type of detected object (e.g., "cigarette").
  - `confidence`: The confidence score of the detection (between 0 and 1).
  - `bbox`: The bounding box coordinates for the detected object, formatted as `[x1, y1, x2, y2]`.
