# vinrei — Local AI coding assistant
# Packages the Python assistant only. The model is served by Ollama
# which must be running on the host (or in a sidecar container).
#
# Build:
#   docker build -t vinrei .
#
# Run (CLI):
#   docker run --rm -it --network host vinrei "explain this bug"
#
# Run (TUI):
#   docker run --rm -it --network host vinrei vinrei-tui
#
# The --network host flag is required so the container can reach
# Ollama at http://localhost:11434 on the host machine.

FROM python:3.11-slim

# system deps for sentence-transformers and psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy project files
COPY ai/pyproject.toml ai/poetry.lock* ./
COPY ai/vinrei ./vinrei

# install dependencies
RUN pip install --no-cache-dir \
    numpy \
    sentence-transformers \
    psutil \
    "textual==0.47.1"

# install the package itself
RUN pip install --no-cache-dir -e .

# default command — CLI chat
ENTRYPOINT ["vinrei"]
CMD ["--help"]
