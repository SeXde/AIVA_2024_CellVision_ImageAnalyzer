# CellVision ImageAnalyzer
**CellVision ImageAnalyzer** allows to detect and locate cells in microscopy images.
The service can be used easily via API REST.

# How to download the project?
To download the project we recommend using:
```bash
git clone https://github.com/SeXde/AIVA_2024_CellVision_ImageAnalyzer.git
```
For all the ways to download a project from GitHub we recommend checking the [official documentation](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github).

# How to use
## Using without the API
### Dependencies
- Python 3.11
- Install PyTorch with GPU support if desired.
- Library versions are described in `requirements.txt`.
- You will need a trained model to run the program. You can request the trained model from us or train your own.

To install the libraries automatically:
> Install PyTorch with GPU support first if desired.
> [PyTorch installation docs](https://pytorch.org/get-started/locally/)

The rest of the libraries can be installed manually or using:
```bash
pip install -r requirements.txt
```

### How to run
To run the program simply create a file like the one in the `./examples/args_example.py` directory and run it using Python with the appropriate image path as the first argument.

If you just want to test if the application is ready and working, you can run the tests instead.

#### Additional examples
To use the cell detection capabilities of **CellVision ImageAnalyzer**, you can:
- Check our examples in the `./example` directory.
- Check our tests in the `./test` directory.


## Using with the API
There are two ways to use the API, directly from the source and from Docker.
We recommend the Docker version as it also includes the trained model.

### Using Docker
Using Docker is the easiest and most straightforward approach to execute the API.
> To run the Docker commands you need to have Docker and DockerHub installed and running!

Just download the Docker image from DockerHub using:
```bash
docker pull lsedev/cellvision:latest
```

And then run it using:
```bash
docker run -it --rm -p 80:80 lsedev/cellvision:latest
```

### Using the source code
If you already have the source files downloaded, just run the following command to start the local server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 80
```
You can choose a different port other than 80 if that one is already occupied.

## Use the Swagger UI
Both the Docker and the source versions include the [OpenAPI/Swagger](https://swagger.io/specification/) UI for easy testing.
To open it up just go to http://127.0.0.1:80/docs on your browser (after starting the local server).

# Architecture

## Cell detections
Cells detections work in using the CellDetector class, that is capable of detecting an `array` of cells from a single image.
The loading of images is left to be personalized for each client, altough we provide a default implementation using **OpenCV** in the [[utils]].

## API
The application uses FastAPI to generate its routes.
There are two available POST endpoints, both routes require an authorization token that must be provided by us.
> If you are just testing the code, a demo token is available inside the `src/token_validator.py` file.

### Single image
`POST /cell-detections/single`\
Generates the detections for a single image.
#### Body
An image must be sent in binary form as a `multipart/form-data` body.\
#### Returns
- Status 200 with an image in case of success.
- Status 401, if the token is invalid.
- Status 422, if there was a validation error.
- Status 500, if there was a problem loading the image.

### Multiple images
`POST /cell-detections/multiple`\
We also provide an endpoint that allows to generate detections for multiple images in a zip file.
#### Body
The zip must be sent in binary form as a `multipart/form-data` body and MUST only contain images.
#### Returns
- Status 200 with a zip file containing all the images with the detections in case of success.
- Status 401, if the token is invalid.
- Status 422, if there was a validation error.
- Status 500, if there was a problem loading the image.


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
You can also run:
```bash
python -m unittest discover tests
```

The tests can also be checked as examples to understand the program behaviour.

# Branches
This project uses GitHub flow, where:
- **master:** contains the last stable release.
- **develop:** contains the latest changes.
- **feature/*:** contains feature branches with individual features in each one.
