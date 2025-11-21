FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8 LANG=C.UTF-8
ARG TZ=Asia/Shanghai
ENV TZ=$TZ
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libsnappy-dev zstd tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt --trusted-host mirrors.aliyun.com --index-url http://mirrors.aliyun.com/pypi/

COPY . /app
RUN useradd -m -u 10001 -s /bin/bash appuser && chown -R appuser:appuser /app
USER appuser
VOLUME ["/app/output", "/app/analysis_results", "/app/logs"]
ENV STREAM_ALL=0 THIN_PROJECTION=0 BATCH_PROCESS_EXECUTOR=process BATCH_PROCESS_WORKERS=12 SEGMENT_PARALLEL=1 SEGMENT_EXECUTOR=thread SEGMENT_PARALLEL_WORKERS=8 SKIP_ANALYSIS=0 RUN_CSV_EXPORT=1
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ENTRYPOINT ["python"]
CMD ["iPPG_info.py"]