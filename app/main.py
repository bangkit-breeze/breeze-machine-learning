from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from model.model_classification import predict_image_clf
from model.model_segmentation import predict_image_sgmnt

app = FastAPI()

model_name = "BREEZE Food Recognition"
model_version = "v1.0.0"

@app.get("/")
async def index():
    """Landing Page"""
    return "Welcome to BREEZE Food Recognition"

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
    predicted_class_clf, confidence_clf = predict_image_clf(img)
    ingredient = predict_image_sgmnt(img)

    return {
        "name": model_name,
        "version": model_version,
        "filename": image.filename,
        "classifier": {
            "class": predicted_class_clf,
            "confidence": str(confidence_clf)
        },
        "segmentation": {
            "class": ingredient
        },
        
    }