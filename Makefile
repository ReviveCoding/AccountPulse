PYTHON := /home/bjw-0/.local/share/accountpulse/project-venv/bin/python
PIPELINE := $(PYTHON) -m accountpulse.pipeline run --config configs/full.yaml

.PHONY: bootstrap acquire qualify eda develop deep causal freeze locked systems mlops report verify all

bootstrap:
	UV_PROJECT_ENVIRONMENT=/home/bjw-0/.local/share/accountpulse/project-venv uv sync --all-extras --locked

acquire:
	$(PIPELINE) --stage acquire

qualify:
	$(PYTHON) -m accountpulse.gpu --out artifacts/bootstrap/post_sync_gpu_smoke.json

eda:
	$(PIPELINE) --stage eda

develop:
	$(PIPELINE) --stage tracka_qualify --stage tracka_features --stage tracka_develop

deep:
	$(PIPELINE) --stage temporal --stage graph --stage text --stage fusion

causal:
	$(PIPELINE) --stage causal

freeze:
	$(PIPELINE) --stage freeze

locked:
	$(PIPELINE) --stage locked

systems:
	$(PIPELINE) --stage qualify

mlops:
	$(PIPELINE) --stage mlops

report:
	$(PIPELINE) --stage report

verify:
	$(PIPELINE) --stage verify

all:
	$(PIPELINE)
