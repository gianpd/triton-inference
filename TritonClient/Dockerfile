FROM nvcr.io/nvidia/tritonserver:22.08-py3-sdk

WORKDIR /workspace

COPY requirements.txt /workspace

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./code /workspace/

ENTRYPOINT ["python", "client.py"]