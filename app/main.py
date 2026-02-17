from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import torch
import os
import time

app = FastAPI()

MODEL_ID = os.getenv("MODEL_ID", "snunlp/KR-FinBert-SC")

REQUEST_COUNT = Counter(
    "request_count", "Total request count", ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency", ["endpoint"]
)

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

class RequestBody(BaseModel):
    text: str

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: RequestBody):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return {"probabilities": probs.tolist()}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
