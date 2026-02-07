import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith import utils as ls_utils

# Load environment variables from the backend/.env file
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

client = Client()


# Define dataset: these are your test cases
dataset_name = "Medical QA Dataset"
created = False
try:
	dataset = client.create_dataset(dataset_name)
	created = True
except ls_utils.LangSmithConflictError:
	dataset = client.read_dataset(dataset_name=dataset_name)

if created:
	client.create_examples(
		dataset_id=dataset.id,
		examples=[
			{
				"inputs": {
					"question": "Which vitamin deficiency causes scurvy?",
					"choices": [
						"Vitamin A",
						"Vitamin B1",
						"Vitamin C",
						"Vitamin D",
					],
				},
				"outputs": {"answer": "C"},
			},
			{
				"inputs": {
					"question": "What is the normal adult resting heart rate range?",
					"choices": [
						"30-40 bpm",
						"60-100 bpm",
						"120-160 bpm",
						"160-200 bpm",
					],
				},
				"outputs": {"answer": "B"},
			},
			{
				"inputs": {
					"question": "Which organism causes tuberculosis?",
					"choices": [
						"Staphylococcus aureus",
						"Mycobacterium tuberculosis",
						"Escherichia coli",
						"Streptococcus pneumoniae",
					],
				},
				"outputs": {"answer": "B"},
			},
			{
				"inputs": {
					"question": "Insulin is produced by which cells of the pancreas?",
					"choices": [
						"Alpha cells",
						"Beta cells",
						"Delta cells",
						"PP cells",
					],
				},
				"outputs": {"answer": "B"},
			},
			{
				"inputs": {
					"question": "First-line treatment for anaphylaxis?",
					"choices": [
						"Oral antihistamine",
						"IV fluids only",
						"Intramuscular epinephrine",
						"Corticosteroids only",
					],
				},
				"outputs": {"answer": "C"},
			},
		],
	)


def _extract_choice(text: str, choices: list[str]) -> str | None:
	"""Extract the answer choice letter (A-D) from the model output."""
	if not text:
		return None

	# Prefer exact single-letter answers first.
	exact = re.match(r"^\s*([A-D])(?:[.)])?\s*$", text, re.IGNORECASE)
	if exact:
		return exact.group(1).upper()

	# Otherwise find the first standalone A-D token.
	token = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
	if token:
		return token.group(1).upper()

	# Fallback: match the text of a choice if present.
	lowered = text.lower()
	choice_map = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
	for letter, option in choice_map.items():
		if option.lower() in lowered:
			return letter
	return None


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
	"""Check if the predicted option matches the reference answer letter."""
	pred = _extract_choice(outputs.get("response", ""), inputs.get("choices", []))
	ref = reference_outputs.get("answer", "").strip().upper()
	return bool(pred and ref and pred == ref)


def concision(outputs: dict, reference_outputs: dict) -> bool:
	"""Check if the model outputs a concise single-letter answer (A-D)."""
	response = outputs.get("response", "")
	return bool(re.match(r"^\s*[A-D](?:[.)])?\s*$", response, re.IGNORECASE))


def my_app(question: str, choices: list[str]) -> str:
	import requests

	url = "http://localhost:8001/v1/medical-qa"
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY is not set in backend/.env")
	headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
	payload = {"question": question, "choices": choices}
	response = requests.post(url, headers=headers, json=payload)
	return response.json()["content"]


def ls_target(inputs: dict) -> dict:
	return {"response": my_app(inputs["question"], inputs["choices"])}


experiment_results = client.evaluate(
	ls_target,
	data=dataset_name,
	evaluators=[concision, correctness],
	experiment_prefix="Llama-3.2-1B-Instruct-MedQA",
)
