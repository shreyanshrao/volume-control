Volume Control with Hand Gestures
This project allows you to control your system volume using finger gestures captured through your webcam. By detecting the distance between your thumb and index finger, the volume can be increased or decreased dynamically.

Features
Real-time hand tracking using MediaPipe.

Adjusts system volume by changing the distance between the thumb and index finger.

Smooth volume transitions for a better user experience.

Works on Windows, macOS, and Linux (with platform-specific audio libraries).

Tech Stack:

Python

OpenCV – For video capture and image processing.

MediaPipe – For hand landmark detection.

PyCaw / pyautogui / alsaaudio – For controlling system volume (OS-specific).

NumPy

How It Works
The webcam feed is processed frame by frame.

MediaPipe detects hand landmarks (21 points on the hand).

The distance between the tip of the thumb and tip of the index finger is calculated.

This distance is mapped to a volume range (e.g., 0%–100%).

System volume is updated in real-time based on this distance.
