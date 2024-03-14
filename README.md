# CellVision ImageAnalyzer
**CellVision ImageAnalyzer** allows to detect and locate cells in microscopy images.
The service can be used easily via API REST.

# How to use
> This section will be documented as soon as the API details are set in stone.

# Architecture

## Cell detections
Cells detections work in using the CellDetector class, that is capable of detecting an `array` of cells from a single image.
The loading of images is left to be personalized for each client, altough we provide a default implementation using **OpenCV** in the [[utils]].

## API
> The API section is not yet described (in code and in this document) yet.
> Will be documented as soon as it is available.

## Utils
The utils folder provides space to leave non-exclusive functions that can work as part of the main program.
It contains:
- `io_utils.py`
    - `load_image`: Loads an image using **OpenCV** from a provided string path.

# Testing
To easily run the tests we recommend using your IDE testing tools.

The tests can also be checked to understand the program behaviour.
