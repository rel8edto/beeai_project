FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONUNBUFFERED=1 \
    PRODUCTION_MODE=true \
    PORT=8080 \
    WATSONX_URL=https://us-south.ml.cloud.ibm.com \
    WATSONX_APIKEY=wbjku_0Hh4qHFy7etXNqUT5vHs_p_yj2YPGmyUkHv5KE \
    WATSONX_PROJECT_ID=5b398c1a-4363-4faf-8c47-11e4ec87d24b \
    OCTAGON_API_KEY=sk_HaMi_e7zbjBar0Rcw_E4Q18M8S0q6SNhGxFwghG0OeVeWx0KSIZ_dSUAEVPsYQp7u60R9xNo_djQXo1gSbldGw


# 1️⃣ copy only the lock + metadata first
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY . .

# 2️⃣ create venv and install deps
RUN rm -rf .venv && uv venv && uv sync --no-cache --locked --link-mode copy \
 && uv pip install -e .

COPY start.sh /app/start.sh

#### Install apt-get and CURL command
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*


RUN chmod +x /app/start.sh

EXPOSE 8080

CMD ["/app/start.sh"]
