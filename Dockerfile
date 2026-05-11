FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# System deps:
# - ffmpeg: embedded subtitle extraction
# - sqlite3: ops convenience for inspecting /config/poptrivia.db
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        sqlite3 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY poptrivia ./poptrivia

RUN pip install --upgrade pip \
    && pip install .

# Install only Chromium for Playwright (skip Firefox/WebKit to save ~600 MB).
# --with-deps brings in the system libraries Chromium needs (libnss, libasound,
# fonts, etc.) which apt-get installs on top of the slim base.
RUN python -m playwright install --with-deps chromium

EXPOSE 8765

CMD ["uvicorn", "poptrivia.main:app", "--host", "0.0.0.0", "--port", "8765"]
