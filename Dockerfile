# Use a lightweight Python base image
FROM python:3.9-slim

# Ensure Python output is not buffered, so we see logs in real time
ENV PYTHONUNBUFFERED=1

# Set /app as working directory
WORKDIR /app

# 1) Copy only requirements, install dependencies (cached if requirements.txt is unchanged)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy the rest of the application source
COPY . .

# (Optional) If your ML pipeline needs build tools, uncomment:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# Expose the port Flask runs on (documentation)
EXPOSE 5000

# Run the Flask app when container starts
# for developement usecase
# CMD ["python", "app.py"]

# for production usecase
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]