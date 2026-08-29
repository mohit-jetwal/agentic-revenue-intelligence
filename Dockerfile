# syntax=docker/dockerfile:1
#
# Two stages. The builder resolves and installs dependencies; the runtime copies
# the finished virtualenv and the source. Splitting them keeps uv, the build
# toolchain and the package caches out of the shipped image - they are needed to
# produce the environment and never to run it.
#
# The dependency layer is built before the source is copied, so editing a Python
# file rebuilds in seconds rather than re-resolving ~40 packages including
# lightgbm, xgboost and mlflow.

# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Pinned to the uv that wrote uv.lock. The lock is `revision = 3`, which older
# uv releases cannot read - an unpinned or older tag makes `--frozen` fail on a
# lockfile that is perfectly valid.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Dependencies first, from the lockfile alone. `--no-install-project` stops uv
# installing the application here: the source is not present yet, and that is
# deliberate - it is what makes this layer cacheable across source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui --no-install-project

COPY app ./app
COPY ml ./ml
COPY data ./data
COPY prompts ./prompts
COPY evaluation ./evaluation
COPY README.md ./

# `--extra ui` because Streamlit is an optional extra, not a main dependency:
# the API does not need it, and the compose UI service runs from this same image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui

# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# libgomp is required by lightgbm and xgboost at import time. Without it the
# image builds cleanly and then fails on the first forecast, which is a worse
# failure than not building at all.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The container writes only to /app/data/local, which is owned below.
RUN useradd --create-home --uid 10001 ari

WORKDIR /app

COPY --from=builder --chown=ari:ari /build/.venv /app/.venv
COPY --from=builder --chown=ari:ari /build/app /app/app
COPY --from=builder --chown=ari:ari /build/ml /app/ml
COPY --from=builder --chown=ari:ari /build/data /app/data
COPY --from=builder --chown=ari:ari /build/prompts /app/prompts
COPY --from=builder --chown=ari:ari /build/evaluation /app/evaluation

# The generated dataset, DuckDB file and app-state database live here. Mounted
# as a volume in compose so a rebuilt image does not discard a generated dataset
# that takes minutes to produce.
RUN mkdir -p /app/data/local /app/mlruns /app/mlartifacts \
    && chown -R ari:ari /app/data /app/mlruns /app/mlartifacts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP__ENVIRONMENT=local

USER ari
EXPOSE 8000

# Hits the real health endpoint rather than checking that a port is open. A
# process that is listening but cannot reach its data is not healthy, and a TCP
# probe would report it as such.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
