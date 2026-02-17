FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN pytest -q

EXPOSE 8000

ENV MODEL_ID=snunlp/KR-FinBert-SC

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
