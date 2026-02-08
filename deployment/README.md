# vLLM Multi-LoRA API Server

A Docker-based vLLM serving solution with LoRA (Low-Rank Adaptation) support for efficient fine-tuned model serving.

## Overview

This setup provides:
- vLLM OpenAI-compatible API server
- BitsAndBytes quantization for memory efficiency
- Dynamic LoRA adapter loading
- NVIDIA GPU support
- Automated adapter management

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (tested on GCP G2 instance with L4 GPU)
- **OS**: Ubuntu 22.04 or later
- **Disk Space**: At least 5GB free space for model cache
- **Memory**: Recommended 8GB+ RAM

### Software Requirements
- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA Container Toolkit
- NVIDIA Drivers (version 535+)
- CUDA 12.x compatible drivers

## Installation

### 1. Install NVIDIA Drivers (GCP G2 Instance)

```bash
# Install GPU drivers
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# Verify installation
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA L4                      Off |   00000000:00:03.0 Off |                    0 |
+-----------------------------------------+------------------------+----------------------+
```

### 2. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
```

### 3. Install NVIDIA Container Toolkit

```bash
# Configure package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
dpkg -l | grep nvidia-container-toolkit
```

### 4. Test GPU Access in Docker

```bash
docker run --rm --runtime=nvidia nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If successful, you should see GPU information displayed.

## Configuration

### Project Structure

```
vllm_api/
├── docker-compose.yaml      # Docker Compose configuration
├── entrypoint.sh           # Container startup script
├── adapters.sh             # LoRA adapter loading script
├── download_adapters.py    # Adapter download utility
├── .env                    # Environment variables (create this)
└── README.md               # This file
```

### Environment Variables (.env)

Create a `.env` file in the project directory:

```bash
# API Authentication
VLLM_API_KEY=your-secret-api-key-here

# Optional: Hugging Face token for private models
HF_TOKEN=your-huggingface-token
```

### Docker Compose Configuration

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.8.0
    container_name: vllm
    runtime: nvidia
    entrypoint: ["/bin/bash", "/app/entrypoint.sh"]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - huggingface_cache:/root/.cache/huggingface
      - vllm_cache:/root/.cache/vllm
      - ./entrypoint.sh:/app/entrypoint.sh
      - ./adapters.sh:/app/adapters.sh
      - ./download_adapters.py:/app/download_adapters.py
      - ../benchmarks:/app/benchmarks
    networks:
      - aio-network
    logging:
      driver: loki
      options:
        loki-url: "http://localhost:3100/loki/api/v1/push"

volumes:
  huggingface_cache:
  vllm_cache:

networks:
  aio-network:
    external: true
```

### vLLM Server Configuration (entrypoint.sh)

```bash
#!/bin/bash
set -e

# Find the Python executable
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON_CMD" ]; then
    echo "Python executable not found! Please install Python."
    exit 1
fi

echo "Using Python executable: $PYTHON_CMD"

# Start vllm in the background
$PYTHON_CMD -m vllm.entrypoints.openai.api_server \
  --model thuanan/Llama-3.2-1B-Instruct-Chat-sft \
  --compilation-config '{"cache_dir":"../cache"}' \
  --port 8000 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --enable-prefix-caching \
  --swap-space 4 \
  --gpu-memory-utilization 0.7 \
  --disable-log-requests \
  --enable-sleep-mode \
  --max-model-len 8192 \
  --enable-lora &

# Store the PID of the background process
VLLM_PID=$!

# Function to check if the API is ready
wait_for_api() {
  echo "Waiting for vLLM API to be ready..."
  while ! curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/v1/models > /dev/null; do
    echo "API not ready yet, waiting..."
    sleep 10
  done
  echo "vLLM API is ready!"
}

# Wait for the API to be ready
wait_for_api

# Download adapters inside the container to ensure they exist
echo "Ensuring adapters are downloaded..."
$PYTHON_CMD /app/download_adapters.py

# Run the script to load lora adapters
echo "Loading LoRA adapters..."
export VLLM_API_KEY=$VLLM_API_KEY
bash /app/adapters.sh || echo "Failed to load adapters"

# Wait for the vllm process
wait $VLLM_PID
```

**Important:** Make the script executable:
```bash
chmod +x entrypoint.sh
```

## Usage

### Starting the Server

```bash
# Start the service
docker compose up -d

# View logs
docker compose logs -f vllm

# Check status
docker compose ps
```

### Stopping the Server

```bash
# Stop the service
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### API Endpoints

Once running, the server exposes OpenAI-compatible endpoints at `http://localhost:8000`:

- **List models**: `GET /v1/models`
- **Chat completion**: `POST /v1/chat/completions`
- **Completions**: `POST /v1/completions`

### Example API Request

```bash
# List available models
curl -H "Authorization: Bearer $VLLM_API_KEY" \
  http://localhost:8000/v1/models

# Generate completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "thuanan/Llama-3.2-1B-Instruct-Chat-sft",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

### Using with LoRA Adapters

```bash
# Generate completion with LoRA adapter
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
    "model": "your-lora-adapter-name",
    "messages": [
      {"role": "user", "content": "Your prompt here"}
    ],
    "max_tokens": 100
  }'
```

## Troubleshooting

### Issue: "No CUDA GPUs are available"

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
```

**Solutions:**

1. **Verify GPU is visible on host:**
   ```bash
   nvidia-smi
   ```

2. **Check Docker has GPU access:**
   ```bash
   docker run --rm --runtime=nvidia nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Ensure nvidia-container-toolkit is installed:**
   ```bash
   dpkg -l | grep nvidia-container-toolkit
   ```

4. **Add `runtime: nvidia` to docker-compose.yaml** (already configured above)

### Issue: "Unsupported display driver / CUDA driver combination"

**Symptoms:**
```
Error 803: system has unsupported display driver / cuda driver combination
```

**Solution:**
Use a compatible vLLM image version. We're using `vllm/vllm-openai:v0.8.0` which works with CUDA 12.x.

### Issue: "BitsAndBytes quantization and QLoRA adapter only support 'bitsandbytes' load format"

**Symptoms:**
```
ValueError: BitsAndBytes quantization and QLoRA adapter only support 'bitsandbytes' load format, but got auto
```

**Solution:**
Add `--load-format bitsandbytes` to the vLLM startup command (already configured in entrypoint.sh above).

### Issue: "No space left on device"

**Symptoms:**
```
No space left on device (os error 28)
Expected file size: 2471.65 MB
Available space: 1057.07 MB
```

**Solutions:**

1. **Check available disk space:**
   ```bash
   df -h
   ```

2. **Clean up Docker resources:**
   ```bash
   docker system prune -a --volumes
   docker image prune -a
   docker volume prune
   ```

3. **Use host directories with more space:**
   
   Update docker-compose.yaml:
   ```yaml
   volumes:
     - /path/to/large/disk/huggingface:/root/.cache/huggingface
     - /path/to/large/disk/vllm:/root/.cache/vllm
   ```

4. **Increase disk size on GCP instance:**
   - Go to GCP Console → Compute Engine → Disks
   - Select your disk and click "Edit"
   - Increase size (recommend 50GB+)
   - Resize filesystem:
     ```bash
     sudo growpart /dev/sda 1
     sudo resize2fs /dev/sda1
     ```

### Issue: Docker Compose command not found

**Symptoms:**
```
Command 'docker-compose' not found
```

**Solution:**
Use `docker compose` (with space, Docker Compose V2) instead of `docker-compose` (hyphen, V1):

```bash
# Correct (V2)
docker compose up -d

# Old syntax (V1)
docker-compose up -d
```

## Performance Tuning

### GPU Memory Utilization

Adjust `--gpu-memory-utilization` in entrypoint.sh (default: 0.7):
- **Lower (0.5-0.6)**: More conservative, leaves room for other processes
- **Higher (0.8-0.9)**: More aggressive, better throughput but less headroom

### Context Length

Adjust `--max-model-len` based on your needs:
- Default: 8192 tokens
- Increase for longer conversations (uses more memory)
- Decrease for better throughput with shorter requests

### Quantization

Current configuration uses BitsAndBytes quantization:
- **Pros**: ~50% memory reduction
- **Cons**: Slightly slower inference

To disable quantization, remove:
```bash
--quantization bitsandbytes \
--load-format bitsandbytes \
```

## Monitoring

### View Logs

```bash
# Follow logs in real-time
docker compose logs -f vllm

# View last 100 lines
docker compose logs --tail=100 vllm

# View logs since specific time
docker compose logs --since 10m vllm
```

### Check GPU Usage

```bash
# On host
nvidia-smi

# Inside container
docker exec -it vllm nvidia-smi

# Watch GPU usage continuously
watch -n 1 nvidia-smi
```

### API Health Check

```bash
# Check if API is responding
curl -H "Authorization: Bearer $VLLM_API_KEY" \
  http://localhost:8000/v1/models

# Check server health
curl http://localhost:8000/health
```

## Network Configuration

This setup assumes an external Docker network named `aio-network`. If it doesn't exist:

```bash
# Create the network
docker network create aio-network

# Or remove the network requirement from docker-compose.yaml
```

## Security Considerations

1. **API Key**: Always set a strong `VLLM_API_KEY` in `.env`
2. **Firewall**: Restrict port 8000 to trusted IPs only
3. **HTTPS**: Use a reverse proxy (nginx, Caddy) for HTTPS in production
4. **Updates**: Regularly update vLLM image for security patches

## Support

For issues or questions:
- vLLM Documentation: https://docs.vllm.ai/
- GitHub Issues: https://github.com/vllm-project/vllm/issues
- Discord: https://discord.gg/vllm

## License

This configuration is provided as-is for educational and development purposes.