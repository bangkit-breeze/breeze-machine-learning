from fastapi import FastAPI, HTTPException, UploadFile
from model.carbon_emission import getCarbonEmission
from model.model_classification import predict_image_clf
from model.model_segmentation import predict_image_sgmnt
from PIL import Image

THRESHOLD = 0.6

app = FastAPI()

model_name = "BREEZE Food Recognition"
model_version = "v1.0.1"

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
    ingredients, total_emission = getCarbonEmission(predicted_class_clf)
    ingredient_segmentation = predict_image_sgmnt(img)

    # Check Threshold
    low_confidence = True if (confidence_clf < THRESHOLD) else False
    disclaimer_text = "The confidence is under 40%, change your angle!"

    return {
        "name": model_name,
        "version": model_version,
        "filename": image.filename,
        "classifier": {
            "class": predicted_class_clf,
            "confidence": str(confidence_clf)
        },
        "segmentation": {
            "class": ingredient_segmentation,
        },
        "ingredients": ingredients,
        "total_emissions": total_emission,
        "disclaimer": disclaimer_text if low_confidence else ""
    }