# Real-Time Face Recognition System
This project is a computer vision system that detects and recognizes faces in a real-time video stream (from a webcam).

## Features

- **Face detection** in real time using Haar cascades
- **Person recognition** – the system compares detected faces against a database of known people
- If the person is in the database – their **name** is displayed
- If the person is not in the database – the message **"Who is it?"** is shown
- Displays the **match percentage** between the detected face and the entry in the database
- Works with a live video stream

## Requirements

### Python Version
The project has been tested and works on **Python 3.7**.
Using other Python versions may cause compatibility issues with dependencies (especially TensorFlow and OpenCV).

### Dependencies
| Library | Version | Purpose |
|---|---|---|
| cvzone | 1.3.3 | Utilities for computer vision tasks |
| opencv-python | 4.5.2.54 | Image and video stream processing |
| tensorflow | 2.5.0 | Neural network models for recognition |

> **Important:** These library versions are specifically chosen for Python 3.7.
> Installing newer versions may break the project.
