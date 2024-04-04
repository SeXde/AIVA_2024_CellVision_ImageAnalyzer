# CellVision ImageAnalyzer
**CellVision ImageAnalyzer** allows to detect and locate cells in microscopy images.
The service can be used easily via API REST.

# How to use
## Using without the API
### Dependencies
- Python 3.11
- Install PyTorch with GPU if desired.
- Library versions are described in `requirements.txt`.
- You will need a trained model to run the program. You can request the trained model from us or train your own.

To install the libraries automatically::
> Install PyTorch with GPU support first if desired.
> [PyTorch installation docs](https://pytorch.org/get-started/locally/)

The rest of the libraries can be automatically installed using:
```bash
pip install -r requirements.txt
```

### How to download the project?
To download the project we recommend using:
```bash
git clone https://github.com/SeXde/AIVA_2024_CellVision_ImageAnalyzer.git
```
For all the ways to download a project from GitHub we recommend checking the [official documentation](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github).

### How to run
To run the program simply create a file like the one in the `./examples/args_example.py` directory and run it using Python with the appropriate image path as the first argument.

#### Additional examples
To use the cell detection capabilities of **CellVision ImageAnalyzer**, you can:
- Check our examples in the `./example` directory.
- Check our tests in the `./test` directory.


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
- `model_utils.py`
  - `create_fcos_model`: Creates a new FCOS model. 
  - `model_outputs_to_bbox`: Converts the model outputs into OpenCV-like bounding boxes.

# Testing
To easily run the tests we recommend using your IDE testing tools.

The tests can also be checked as examples to understand the program behaviour.

# Branches
This project uses GitHub flow, where:
- **Master:** contains the last stable release.
- **Develop:** contains the latest changes.
- **feature/*:** contains feature branches with individual features in each one.
