FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg is required for embedded subtitle extraction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY poptrivia ./poptrivia

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8765

CMD ["uvicorn", "poptrivia.main:app", "--host", "0.0.0.0", "--port", "8765"]
