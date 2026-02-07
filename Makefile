.PHONY: bench_serving benchmark_prefix_caching

PYTHON ?= python3
BASE_URL ?= http://127.0.0.1:8000
MODEL ?= thuanan/Llama-3.2-1B-Instruct-Chat-sft

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

benchmark_prefix_caching:
	docker exec vllm python3 /app/benchmarks/benchmark_prefix_caching.py \
		--model $(MODEL) \
		--tokenizer $(MODEL) \
		--dataset-path /app/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
		--enable-prefix-caching \
		--num-prompts 20 \
		--repeat-count 5 \
		--input-length-range 128:256