from fastapi import FastAPI,HTTPException 
from pydantic import BaseModel 
from src.summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(
    title="Text Summarization API",
    description="Summarize long articles using a Transformer model",
    version="1.0.0"
)

predictor = PredictionPipeline()

class PredictRequest(BaseModel):
    text: str 

class PredictResponse(BaseModel):
    summary: str

@app.get("/")
def root():
    return {"message":"Text Summarization API running"}

@app.post("/predict",response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        summary = predictor.predict(req.text)
        return {"summary":summary}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))