FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# Default to service name of backend when running in compose
ENV BACKEND_URL=http://aph_if_backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "app/bot.py", "--server.address=0.0.0.0", "--server.port", "8501"]

