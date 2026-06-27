# llm-fusion — Fusion Proxy container image.
#
# Build:  docker build -t llm-fusion .
# Run:    docker run --rm -p 8080:8080 \
#           -e OLLAMA_API_KEY=ollama-... \
#           llm-fusion
#
# IMPORTANT — binding: the in-image fusion.yaml sets `server.bind: 127.0.0.1`,
# which only serves *inside* the container. To reach the proxy from the host you
# MUST make it listen on 0.0.0.0. This image defaults FUSION_BIND=0.0.0.0 (see
# below) so a plain `-p 8080:8080` works; override per run with
# `-e FUSION_BIND=...` or mount a config whose `server.bind` you control.
FROM node:24-slim

WORKDIR /app

# Install dependencies first for layer caching (only re-runs when lockfile changes).
COPY package.json package-lock.json ./
RUN npm ci

# Application source + configs (tsx runs the TypeScript directly; no build step).
COPY tsconfig.json ./
COPY src ./src
COPY fusion.yaml fusion.example.yaml ./

ENV NODE_ENV=production
# Listen on all interfaces inside the container so the published port reaches the
# host. The committed fusion.yaml still binds 127.0.0.1 for non-Docker local use;
# this env override wins without editing the file. Override with -e FUSION_BIND=...
ENV FUSION_BIND=0.0.0.0

EXPOSE 8080

# OLLAMA_API_KEY must be supplied at run time (never baked into the image).
CMD ["npx", "tsx", "src/index.ts"]
