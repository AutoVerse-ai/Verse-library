# 1. Match your local environment exactly
FROM python:3.11.9-slim

WORKDIR /app

# Ensure logs appear in real-time and python looks in /app for modules
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Tell VTK and Matplotlib to run without a physical screen
    QT_QPA_PLATFORM=offscreen \
    MPLBACKEND=Agg

# 2. Install system dependencies
# Added libgl1, libglib2.0, and xvfb to solve the 'libGL.so.1' and VTK errors
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Torch for CUDA 12.1 (The Linux version)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Use your cleaned requirements (from the PowerShell script below)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# 5. Install my forked auto_LiRPA directly from GitHub
RUN pip install --no-cache-dir git+https://github.com/AlexYFM/auto_LiRPA.git@711153a6e9253bd996f80d467c081314469f4e34

# 6. Copy the rest of your project
COPY . /app

# Default command
CMD ["python", "demo/aprod/orbital_docking_demo_ver_gpa_parser_wrapper.py"]