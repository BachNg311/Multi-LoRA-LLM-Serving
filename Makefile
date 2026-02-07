.PHONY: bench_serving bench_serving_docker benchmark_prefix_caching

# Load OPENAI_API_KEY and other variables if present
-include backend/.env

export OPENAI_API_KEY

PYTHON ?= python3
BASE_URL ?= http://10.255.255.254:8000 # Replace with the actual IP address and port of your vLLM server
MODEL ?= thuanan/Llama-3.2-1B-Instruct-Chat-sft
GPU_MEMORY_UTILIZATION ?= 0.5
MAX_MODEL_LEN ?= 8192

bench_serving:
	$(PYTHON) benchmarks/benchmark_serving.py \
		--backend openai-chat \
		--base-url $(BASE_URL) \
		--model $(MODEL) \
		--tokenizer $(MODEL) \
		--endpoint /v1/chat/completions \
		--dataset-name sharegpt \
		--dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
		--request-rate 10.0 \
		--max-concurrency 5 \
		--result-dir ./benchmarks/results --save-result --save-detailed

bench_serving_docker:
	docker exec -e OPENAI_API_KEY="$(OPENAI_API_KEY)" vllm python3 /app/benchmarks/benchmark_serving.py \
		--backend openai-chat \
		--base-url http://localhost:8000 \
		--model $(MODEL) \
		--tokenizer $(MODEL) \
		--endpoint /v1/chat/completions \
		--dataset-name sharegpt \
		--dataset-path /app/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
		--request-rate 10.0 \
		--max-concurrency 5 \
		--result-dir /app/benchmarks/results --save-result --save-detailed

benchmark_prefix_caching:
	docker exec -e OPENAI_API_KEY="$(OPENAI_API_KEY)" vllm python3 /app/benchmarks/benchmark_prefix_caching.py \
		--model $(MODEL) \
		--tokenizer $(MODEL) \
		--dataset-path /app/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
		--enable-prefix-caching \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--num-prompts 20 \
		--repeat-count 5 \
		--input-length-range 128:256