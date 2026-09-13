# The RAG's Dockerfile

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHUNK_SIZE=500 \
    CHUNK_OVERLAP=100 \
    K=4 \
    TEMPERATURE=0

CMD ["python","./app.py"]