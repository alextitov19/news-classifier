# app/main.py
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()

class Article(BaseModel):
    headline: str
    short_description: str

# Load the trained model
model = joblib.load('app/models/classifier.joblib')

@app.post("/predict/")
async def predict(article: Article):
    try:
        text = f"{article.headline} {article.short_description}"
        prediction = model.predict([text])
        return {"category": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))