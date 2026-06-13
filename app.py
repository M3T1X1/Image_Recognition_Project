from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Car Body Classifier API", description="API do rozpoznawania karoserii samochodów")

session = ort.InferenceSession("resnet18.onnx", providers=['CPUExecutionProvider'])

CATEGORIES = ['convertible', 'coupe', 'hatchback', 'other', 'sedan', 'suv', 'wagon']


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Zmiana rozmiaru

    img_data = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std

    img_data = np.transpose(img_data, (2, 0, 1))

    img_data = np.expand_dims(img_data, axis=0)

    return img_data


@app.post("/predict")
async def predict_car_body(file: UploadFile = File(...)):
    image_bytes = await file.read()

    input_tensor = preprocess_image(image_bytes)

    outputs = session.run(["output"], {"input": input_tensor})[0]

    predicted_idx = np.argmax(outputs[0])
    predicted_class = CATEGORIES[predicted_idx]

    return {
        "filename": file.filename,
        "predicted_body_type": predicted_class
    }