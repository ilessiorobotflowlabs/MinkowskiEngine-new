ARG PYTORCH="2.5.1"
ARG CUDA="12.1.0"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# Modify to match your GPU compute capability
# Ada Lovelace (L4, RTX 4090): 8.9
# Ampere (A100, A10): 8.0
# Hopper (H100): 9.0
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update && \
    apt-get install -y git ninja-build cmake build-essential libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV MAX_JOBS=8

RUN pip install uv

RUN git clone --recursive "https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new" && \
    cd MinkowskiEngine-new && \
    uv pip install --system -e .
