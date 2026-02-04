from fastapi import FastAPI,HTTPException 
from pydantic import BaseModel 
from src.summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(
    title="Text Summarization API",
    description="Summarize long articles using a Transformer model",
    version="1.0.0"
)

_predictor = None

def get_predictor():
    global _predictor 
    if _predictor is None: 
        _predictor = PredictionPipeline()
    return _predictor

class PredictRequest(BaseModel):
    text: str 

class PredictResponse(BaseModel):
    summary: str

@app.get("/")
def root():
    return {"message":"Text Summarization API running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict",response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        predictor = get_predictor()
        summary = predictor.predict(req.text)
        return {"summary":summary}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))