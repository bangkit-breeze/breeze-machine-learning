from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from model.model import predict_image_segmentation

app = FastAPI()

model_name = "BREEZE Food Segmentation"
model_version = "v1.0.0"

@app.get("/")
async def index():
    """Landing Page"""
    return "Welcome to BREEZE Food Segmentation"

@app.post("/predict")
async def predict(image: UploadFile):
    """Predicting Image"""
    # check form data
    if not image:
        raise HTTPException(status_code=422, detail="Image field cannot be blank.")

    # check image type
    if "image" not in image.content_type:
        raise HTTPException(status_code=400, detail="File must be an image")

    img = Image.open(image.file)
    ingredient = predict_image_segmentation(img)

    return {
        "name": model_name,
        "version": model_version,
        "filename": image.filename,
        "class": ingredient
    }