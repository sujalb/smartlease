FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file
COPY --chown=user requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["chainlit", "run", "chainlit.py", "--host", "0.0.0.0", "--port", "7860"]