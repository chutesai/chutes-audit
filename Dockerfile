FROM python:3.11

WORKDIR /audit

# Install system dependencies
RUN apt-get update && apt-get install -y libportaudio2 && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Ensure Poetry does NOT use virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-root --no-interaction --no-ansi && poetry show

# Copy the application code
COPY . .

# Explicitly set PYTHONPATH
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages"

# Set the command to run the audit script
CMD ["python", "audit.py"]