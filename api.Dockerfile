FROM python:3.10-slim

WORKDIR /app

COPY setup.py pyproject.toml requirements.txt ./

# Install runtime deps (uvicorn MUST be here)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
