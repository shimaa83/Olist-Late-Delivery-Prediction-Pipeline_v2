# ==========================================
# Stage 1: build - نستخدم uv نفسه (نفس أداة المشروع) لتثبيت الـ core deps بس
# ==========================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# ✅ بننسخ ملفات الـ dependency resolution بس الأول (لتحسين الـ layer caching)
# لو عندك uv.lock في الريبو، اتأكد إنه متزامن (uv lock) قبل الـ build
COPY pyproject.toml uv.lock* ./
RUN touch README.md
# ✅ --no-dev بيستبعد مجموعة dev، ومجموعة train (غير افتراضية) بتتستبعد تلقائيًا
# لأننا مش بنطلبها بـ --group train. يعني بيتنصب الـ core dependencies بس.
RUN uv sync --no-dev --no-install-project

# دلوقتي بننسخ كود المشروع ونثبته
COPY . .
RUN uv sync --no-dev


# ==========================================
# Stage 2: runtime - صورة نهائية صغيرة، بدون notebooks وبدون uv
# ==========================================
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# مستخدم غير root
RUN useradd --create-home --shell /bin/bash appuser

# ✅ بننسخ الـ virtualenv الجاهز بس (فيه core deps بس، بدون mlflow/dvc/xgboost...)
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# ✅ بننسخ كود الخدمة فقط - مفيش notebooks, مفيش data, مفيش سكريبتات تدريب
COPY app/ ./app/
COPY src/task3/config.py ./src/task3/config.py

# نضمن وجود ملفات __init__.py حتى لو مش موجودة في السورس (عشان الـ import يشتغل)
RUN mkdir -p ./src/task3 && \
    touch ./src/__init__.py ./src/task3/__init__.py

# مجلدات الـ artifacts والـ models بتتوصل عن طريق volumes وقت التشغيل
# (مش بتتنسخ جوه الصورة - عشان الصورة تفضل خفيفة ومنفصلة عن البيانات)
RUN mkdir -p /app/artifacts /app/models /app/logs && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health').read(); sys.exit(0)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
