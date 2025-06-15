from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
from fer import FER

app = FastAPI()
detector = FER()

@app.get("/")
def read_root():
    return {"message": "Emotion Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = detector.detect_emotions(image)

    if not result:
        return JSONResponse(content={"error": "No face detected"}, status_code=400)

    top_emotion = result[0]["emotions"]
    predicted_emotion = max(top_emotion, key=top_emotion.get)

    return {"emotion": predicted_emotion, "confidence": top_emotion[predicted_emotion]}
