FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# uv 引导安装源；本机使用官方 PyPI，避免镜像站下载返回 403。
# 项目依赖源由 pyproject.toml 与 uv.lock 统一配置。
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_NO_CACHE_DIR=1

# 系统库：curl(健康检查) / libmagic(python-magic) / poppler-utils(pdf2image) / libgl、libglib(opencv、unstructured-inference)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libmagic1 \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（走上述 PyPI 源）。
RUN pip install --no-cache-dir uv

ENV UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    # 将依赖直接装入系统 Python（/usr/local），而不是默认的项目 .venv
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# 先用锁文件安装依赖以利用 Docker 层缓存（pyproject/uv.lock 未变化时跳过重装）
# --no-install-project：运行时不依赖 pip 安装本包，源码直接 COPY 进 /app 即可
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
