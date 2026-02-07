from huggingface_hub import snapshot_download
import os

print("Downloading/Verifying LoRA adapters...")
try:
    vsf_lora_path = snapshot_download(repo_id="thuanan/Llama-3.2-1B-Instruct-lora-vsf")
    print(f"VSF LoRA path: {vsf_lora_path}")
    
    med_qa_lora_path = snapshot_download(repo_id="thuanan/med-mcqa-llama-3.2-1B-Instruct-4bit-lora")
    print(f"MedQA LoRA path: {med_qa_lora_path}")
    print("Adapters downloaded successfully.")
except Exception as e:
    print(f"Error downloading adapters: {e}")
    exit(1)
