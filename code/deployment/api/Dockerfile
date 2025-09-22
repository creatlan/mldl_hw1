FROM python:3.9-slim

WORKDIR /app

# Копируем requirements.txt из корня проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем дополнительные зависимости для API
RUN pip install fastapi uvicorn

# Копируем код API
COPY code/deployment/api/api.py .

# Копируем сохраненные модели
COPY code/saved_models/ ./saved_models/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]