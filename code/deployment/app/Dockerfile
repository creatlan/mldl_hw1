FROM python:3.9-slim

WORKDIR /app

# Копируем requirements.txt из корня проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем дополнительные зависимости для приложения
RUN pip install streamlit requests

# Копируем код приложения
COPY code/deployment/app/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]