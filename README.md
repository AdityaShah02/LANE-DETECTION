# Lane Detection using Image Processing

## Overview
This project is aimed at detecting lanes on the road using image processing techniques. It utilizes computer vision algorithms to identify lane markings in images or video streams.

![Lane Detection Demo](solidWhiteRight.gif)

## Dependencies
- Python (>=3.6)
- OpenCV
- NumPy

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/lane-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install opencv-python numpy
    ```

## Usage
1. Navigate to the project directory:
    ```bash
    cd lane-detection
    ```
2. Run the lane detection script:
    ```bash
    python lane_detection.py --input path/to/input/video.mp4
    ```
    Replace `path/to/input/video.mp4` with the path to your input video file.

## Algorithm
The lane detection algorithm consists of the following steps:
1. **Preprocessing**: Convert the image to grayscale and apply Gaussian blur to reduce noise.
2. **Edge Detection**: Use the Canny edge detection algorithm to detect edges in the image.
3. **Region of Interest**: Define a region of interest where lanes are expected to appear.
4. **Hough Transform**: Apply Hough line transformation to detect lines in the region of interest.
5. **Lane Identification**: Based on slope and position, identify left and right lane lines.
6. **Lane Drawing**: Draw the detected lanes on the original image.

## Demo
![Lane Detection Demo](demo.gif)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by [Lane Detection Using OpenCV](https://github.com/udacity/CarND-LaneLines-P1) by Udacity.
