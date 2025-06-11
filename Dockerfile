# Stage 1: Build dependencies and wheels
FROM python:3.9-slim-buster AS builder

# Install build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --no-deps -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.9-slim-buster

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy built wheels and install
COPY --from=builder /install/*.whl /wheels/
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Copy application code
COPY . .

# Expose Flask port\ nEXPOSE 5000

# Run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", "--threads", "2", "--worker-class", "gthread"]