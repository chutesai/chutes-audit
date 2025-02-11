FROM python:3.12
RUN apt update && apt-get -y install libportaudio2
RUN curl -sSL https://install.python-poetry.org | python3 -
ADD pyproject.toml /audit/
ADD poetry.lock /audit/
ENV PATH=/root/.local/bin:$PATH
WORKDIR /audit
RUN poetry install --no-root
ADD audit.py /audit/audit.py
