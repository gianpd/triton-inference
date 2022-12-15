FROM nvcr.io/nvidia/tritonserver:22.08-py3-sdk

WORKDIR /workspace

COPY requirements.txt /workspace

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV OBJ_CODE='OBJ_NJ'
ENV AP_CODE='AIJ'
ENV TH_SCORE_DC=0.5
ENV TH_SCORE_OD=0.8
ENV MODEL='rock_nj_model'
ENV MODEL_VERSION='v0.1'
ENV DOWNSAMPLING=3
ENV APPLIANCE_UID='b49691e20fb8'
ENV HARDWARE_VERSION='ROCK_ETH_001'
ENV SOFTWARE_VERSION='AIJ_od_predc_dc_0.1'

COPY ./code/ .
ENTRYPOINT ["python", "client.py"]