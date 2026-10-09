FROM python:3.11.9-slim

ARG AUTO_LIRPA_REF=711153a6e9253bd996f80d467c081314469f4e34

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y git graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Verse-library

COPY requirements.txt setup.py ./

# Keep the CUDA-enabled PyTorch versions used by the working environment.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.3.1+cu121 \
        torchaudio==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        "git+https://github.com/AlexYFM/auto_LiRPA.git@${AUTO_LIRPA_REF}"

COPY . .
RUN python -m pip install --no-deps -e .

CMD ["python"]