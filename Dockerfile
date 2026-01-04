FROM python:3.12
RUN apt update && apt-get -y install libportaudio2 build-essential curl postgresql-client
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ADD pyproject.toml uv.lock README.md /audit/
WORKDIR /audit
RUN uv sync --frozen
ADD audit.py /audit/audit.py
