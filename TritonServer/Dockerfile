FROM nvcr.io/nvidia/tritonserver:22.10-py3

WORKDIR /workspace

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

COPY requirements.txt /workspace
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash", "-c"]