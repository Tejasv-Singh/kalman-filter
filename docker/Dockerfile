FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit dashboard
CMD ["streamlit", "run", "dashboards/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative: run Jupyter notebook server
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

