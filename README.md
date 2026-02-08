# LLMOps: Multi-Adapter LLM Serving Platform

A comprehensive MLOps solution for deploying, serving, and monitoring fine-tuned LLMs.

## Project Overview

This project demonstrates an end-to-end LLM deployment with a focus on:

- Serving fine-tuned Llama 3.2 1B models with vLLM
- Backend API with LangChain for structured LLM interactions
- Frontend interface with Gradio
- Comprehensive monitoring with Prometheus and Grafana
- Log aggregation with Loki

## Components

- **vLLM API**: Serves the Llama 3.2 1B base model and custom LoRA adapters
- **Backend**: FastAPI service that handles prompting, model selection, and response formatting
- **Frontend**: Gradio web interface for easy interaction with the models
- **Monitoring**: Prometheus, Grafana, and Loki for observability

## Use Cases

1. **Sentiment Analysis**: Analyzes text sentiment using a fine-tuned model
2. **Medical QA**: Answers medical multiple-choice questions with domain-specific tuning

## Getting Started

1. Set up the network:

   ```bash
   docker network create aio-network
   ```

2. Start the monitoring stack:

   ```bash
   cd monitor
   docker compose up -d
   ```

3. Launch the vLLM API server:

   ```bash
   cd vllm_api
   docker compose up -d
   ```

4. Start the backend API:

   ```bash
   cd backend
   docker compose up -d
   ```

5. Launch the frontend application:

   ```bash
   cd frontend
   docker compose up -d
   ```

## Accessing Services (Local)

- **vLLM API**: `http://localhost:8000`
- **Backend API**: `http://localhost:8001`
- **Gradio UI**: `http://localhost:7861`
- **Open WebUI**: `http://localhost:8080`
- **Grafana**: `http://localhost:3000`
- **Prometheus**: `http://localhost:9090`

## Deployment (HTTPS with one DuckDNS subdomain)

DuckDNS allows only one subdomain. Use a single domain with path-based routing:

- **Backend API**: `https://lora-llm-serving.duckdns.org/api/v1`
- **Gradio UI**: `https://lora-llm-serving.duckdns.org/app/`
- **Open WebUI**: `https://lora-llm-serving.duckdns.org/webui/`

Steps:

1. Provision a G2 VM on Google Compute Engine (enable GPUs, pick region/zone with capacity, and choose a suitable NVIDIA GPU).
2. Ensure the host has a public IP and ports 80 and 443 are allowed by your firewall.
3. Create a DuckDNS subdomain and map it to your server IP (example: `lora-llm-serving.duckdns.org`).
4. Update the DuckDNS record whenever your public IP changes (use the DuckDNS client or a cron job).
5. Configure Caddy with path-based routing:

    ```caddy
    lora-llm-serving.duckdns.org {
       handle_path /api/* {
          reverse_proxy localhost:8001
       }

       handle_path /app/* {
          reverse_proxy localhost:7861
       }

       handle_path /webui/* {
          reverse_proxy localhost:8080
       }
    }
    ```

6. Update frontend env vars for the backend URL and Gradio root path:

    ```bash
    BACKEND_API_URL=https://lora-llm-serving.duckdns.org/api/v1
    GRADIO_ROOT_PATH=/app
    ```

Note: Open WebUI may not support a subpath. If `/webui` fails, use a DNS provider that supports subdomains or keep Open WebUI on its own HTTPS port.

## Benchmark

```bash
export OPENAI_API_KEY=<your vllm api key>
make bench_serving
make benchmark_prefix_caching
```
