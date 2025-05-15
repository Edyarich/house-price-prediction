FROM python:3.10-slim

WORKDIR /app
COPY inference.py model.pkl requirements.txt ./
COPY code/ ./code

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000
CMD ["python", "inference.py"]