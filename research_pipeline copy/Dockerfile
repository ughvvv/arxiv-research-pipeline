FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies and cron
RUN apt-get update && apt-get install -y cron && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directory for embeddings
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Copy application code and configuration
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create cron job
COPY crontab.txt /etc/cron.d/research-cron
RUN chmod 0644 /etc/cron.d/research-cron && \
    crontab /etc/cron.d/research-cron

# Create log file
RUN touch /var/log/research.log && \
    chmod 0666 /var/log/research.log

# Run cron in foreground
CMD ["cron", "-f"]
