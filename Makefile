PYTHONPATH=src
export PYTHONPATH

.PHONY: test counts corpus dataset train figures api demo

test:
	pytest

counts:
	python3 -m fos_triage.collect counts

corpus:
	python3 -m fos_triage.collect corpus

dataset:
	python3 -m fos_triage.build_dataset

train:
	python3 -m fos_triage.train

api:
	uvicorn fos_triage.api:app --reload

demo:
	streamlit run src/fos_triage/streamlit_app.py
