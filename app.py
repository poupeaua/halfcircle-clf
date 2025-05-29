"""
API for model inference
"""

from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from PIL import Image
import torch
import io

from src.data.transforms import INFERENCE_TRANSFORMS
from src.modeling.model import HCCLF

MODEL_PATH = "models/keep/model_20250315_234944_47.pt"

app = FastAPI()

model = HCCLF()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval().to("cpu")

@app.post("/predict/")
def predict(file: bytes = File(...)):
    try:
        image = Image.open(io.BytesIO(file))
        tensor: torch.Tensor = INFERENCE_TRANSFORMS(image).to("cpu")
        tensor = tensor.unsqueeze(0)
        
        with torch.no_grad():
            prediction: torch.Tensor = model(tensor)
        
        return JSONResponse(content={"prediction": prediction.numpy().tolist()[0][0]})
    
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=400)
    
@app.get("/test/")
def test():
    return JSONResponse(content={"message": "API is working!"})