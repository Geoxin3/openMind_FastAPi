from fastapi import FastAPI
from model_class import TextInput
from model import detect_emotions

app = FastAPI()

@app.post("/analyze_emotion/")
async def analyze_emotion(input_data: TextInput):
    emotional_summary = detect_emotions(input_data.text)
    
    return {"emotion_summary": emotional_summary}
