#!/bin/bash

# Resolve latest snapshot paths dynamically
MED_QA_LORA_DIR=$(ls -d /root/.cache/huggingface/hub/models--thuanan--med-mcqa-llama-3.2-1B-Instruct-4bit-lora/snapshots/* 2>/dev/null | head -n 1)
VSF_LORA_DIR=$(ls -d /root/.cache/huggingface/hub/models--thuanan--Llama-3.2-1B-Instruct-lora-vsf/snapshots/* 2>/dev/null | head -n 1)

if [ -z "$MED_QA_LORA_DIR" ]; then
    echo "MedQA LoRA snapshot not found. Download it first."
    exit 1
fi

if [ -z "$VSF_LORA_DIR" ]; then
    echo "VSF LoRA snapshot not found. Download it first."
    exit 1
fi

# Curl command to load lora adaptors
curl -X POST "http://localhost:8000/v1/load_lora_adapter" \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "lora_name": "medqa-lora",
        "lora_path": "'"$MED_QA_LORA_DIR"'"
    }'

curl -X POST "http://localhost:8000/v1/load_lora_adapter" \
    -H "Authorization: Bearer $VLLM_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "lora_name": "vsf-lora",
        "lora_path": "'"$VSF_LORA_DIR"'"
    }'