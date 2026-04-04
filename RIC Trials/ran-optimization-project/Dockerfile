# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies for the emulator
COPY emulator/requirements.txt /app/emulator/
RUN pip install --no-cache-dir -r emulator/requirements.txt

# Copy and install dependencies for the automation script
COPY app/requirements.txt /app/app/
RUN pip install --no-cache-dir -r app/requirements.txt

# Copy application files
COPY emulator /app/emulator
COPY app /app/app

# Install Supervisor to manage multiple processes
RUN apt-get update && apt-get install -y supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose any required ports
EXPOSE 5000 5001

# Start Supervisor
CMD ["supervisord", "-n"]
