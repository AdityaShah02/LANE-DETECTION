# Lane Detection using Image Processing

## Overview
This Python project utilizes OpenCV to perform lane detection in road scenes captured in video format. Lane detection is a crucial aspect of autonomous driving systems, aiding in navigation and lane-keeping tasks. The code processes each frame of the input video, identifies lane markings, and highlights them with a green color for visualization.

## Sample Video
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
   - OpenCV
    ```bash
    pip install opencv-python
    ```
   - Numpy
    ```bash
    pip install numpy
    ```
    
## Functions

### `BGR to RGB Image`
- Converts an image from BGR to RGB format.
- Uses OpenCV's `cvtColor` function.
- Returns the converted RGB image.
```python
def bgr_rgb(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb
```

### `Region Of Interest`
- Defines a region of interest (ROI) mask for lane detection.
- Specifies a trapezoidal region at the bottom of the image.
- Calculates the edges of the trapezoidal region using polynomial fitting.
- Generates a binary mask where pixels inside the ROI are set to True.
- Returns the binary mask.
```python
def region_of_interest(image_height, image_width,xsize,ysize):
    left_bottom = [100, image_height - 1]
    right_bottom = [950, image_height - 1]
    apex = [480, 290]
    
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))
    
    return region_thresholds
```
### `Color Threshold`
- Calculates color thresholds for image processing.
- Sets thresholds for red, green, and blue channels.
- Generates a binary mask based on whether each pixel's RGB values meet the defined thresholds.
- Returns the binary mask.
```python
def c_thresh(image):
    # Define color selection criteria
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    
    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                       (image[:, :, 1] < rgb_threshold[1]) | \
                       (image[:, :, 2] < rgb_threshold[2])
    return color_thresholds
```
### `Processing the Image`
- Main image processing function.
- Converts the input image from BGR to RGB format.
- Calculates dimensions of the image and creates copies for line drawing and color selection.
- Computes region of interest mask and color thresholds.
- Sets pixels outside ROI or failing to meet color thresholds to black.
- Marks pixels within ROI and meeting color criteria with a green color.
- Blends the line image with the original image to create the final output.
- Returns the processed image.
```python
def process_image(image):
    # Convert to RGB
    image_rgb=bgr_rgb(image)
        
    ysize = image.shape[0]
    xsize = image.shape[1]
    line_image = np.copy(image)
    color_select=np.copy(image)

    region_thresholds = region_of_interest(539,680,xsize,ysize)

    color_thresholds=c_thresh(image)

    # Mask color and region selection
    color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
    # Color pixels green where both color and region selections met
    # line_image[~color_thresholds & region_thresholds] = [79,250,161]
    line_image[~color_thresholds & region_thresholds] = [0,255,91]

    # Add the line to the original image
    combo_image = cv2.addWeighted(image, 0.001, line_image, 1, 1)

    return combo_image
```
### `Main Implementation`

The code reads a video file (`solidWhiteRight.mp4`) using OpenCV's VideoCapture. Each frame of the video is processed using `process_image`. The processed image is displayed using `cv2.imshow`. The loop continues until the video is finished or the user presses 'q' to quit. Finally, it releases the video capture and closes all OpenCV windows.
```python
# Open video file
cap = cv2.VideoCapture('solidWhiteRight.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        print("Completed !")
        break

    # Process the image
    result = process_image(image)

    # Display the processed image
    cv2.imshow('image', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Output Video
![Lane Detection Demo](output.gif)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```css
This README now provides a clear overview of the lane detection project, including a description of the algorithm, installation instructions, function definitions, sample video, and a demo visualization.
```
