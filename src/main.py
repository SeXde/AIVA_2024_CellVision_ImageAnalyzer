from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Depends, Header
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.responses import FileResponse
import tempfile

import utils.model_utils as model_utils
import zipfile
import cell_detector
import token_validator

app = FastAPI()
detector = cell_detector.CellDetector()
validator = token_validator.TokenValidator()
common_path = '/cell-detections'


# Decorator to verify token
def verify_token(token: str = Header(...)):
    if not validator.validate_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")


def process_image(image_pil):
    image_np = np.array(image_pil)

    # Process the image using the detect_cells function
    bboxes = detector.detect_cells(image_np)
    detected_image_np = model_utils.print_bboxes(image_np, bboxes, print_image=False)

    # Convert the image back to bytes
    detected_image_pil = Image.fromarray(detected_image_np)
    detected_image_data = BytesIO()
    detected_image_pil.save(detected_image_data, format="PNG")
    detected_image_bytes = detected_image_data.getvalue()
    return detected_image_bytes


@app.post(fr'{common_path}/single')
async def detect_cells_single_image(image: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Detect cells in a single image.

    Args:
        image (UploadFile): Image to be processed.
        token (str): Authentication token.

    Returns:
        Response: Image with cells detected.
    """
    try:
        # Read the image
        image_data = await image.read()

        # Convert the image to a np.ndarray
        image_pil = Image.open(BytesIO(image_data))
        detected_image_bytes = process_image(image_pil)

        # Return the processed image as an attachment in the HTTP response
        return Response(content=detected_image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image `{str(e)}`")


@app.post(fr'{common_path}/multiple')
async def detect_cells_multiple_images(images_zip: UploadFile = File(...), token: str = Depends(verify_token)):
    """
    Detect cells in a single image.

    Args:
        images_zip (UploadFile): Image to be processed.
        token (str): Authentication token.

    Returns:
        Response: Image with cells detected.
    """
    try:
        # Read the zip file
        images_zip_data = await images_zip.read()

        # Create a .zip file in memory to store the processed images
        processed_zip_data = BytesIO()
        with zipfile.ZipFile(processed_zip_data, 'w') as processed_zip:
            with zipfile.ZipFile(BytesIO(images_zip_data), 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    with zip_ref.open(filename) as image_file:
                        # Convert the image to a np.ndarray
                        image_pil = Image.open(image_file)
                        detected_image_bytes = process_image(image_pil)

                        # Write the processed image to the .zip file
                        processed_zip.writestr(filename, detected_image_bytes)

        # Save the .zip file to the temporary file system
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(processed_zip_data.getvalue())
            tmp_file_path = tmp_file.name

        # Return the .zip file from the temporary file system as an HTTP response
        return FileResponse(tmp_file_path, media_type='application/zip', filename='processed_images.zip')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images `{str(e)}`")
