# Penrose — research referee. Common tasks.
# The core (eval, calib-*, worked-example, connections, test) runs with no API key and no external data.
# Ingesting a paper needs a model key; the cz-* / rdagent targets need a data download (see the notebook).
# Sandboxed auto-module implementation optionally requires Docker.

PY ?= python3
export PYTHONPATH := src:.

.PHONY: help eval worked-example calib-placebo calib-injection calib-nulls calib-sensitivity \
        calib-breadth calib-synth calib-hypothesis connections test \
        run review dream synthesize stage0 dash cz-referee cz-decay rdagent-referee reset clean

help:
	@echo "Penrose targets (the core runs keyless, no external data):"
	@echo "  make eval             — planted strategies with known verdicts (proves it discriminates)"
	@echo "  make worked-example   — same return series, two search processes, two verdicts"
	@echo "  make calib-placebo    — negative control: no-edge signals must NOT certify"
	@echo "  make calib-injection  — positive control: detection curve for a known injected edge"
	@echo "  make calib-nulls      — 5-null falsification battery: no null may certify"
	@echo "  make calib-sensitivity— detection-threshold vs sample size and cost"
	@echo "  make calib-breadth    — native-breadth recalibration (IC floor vs N)"
	@echo "  make calib-synth      — honesty controls for concepts / synthesis / firewall"
	@echo "  make calib-hypothesis — planted-principle recovery + grounding firewall"
	@echo "  make connections      — advisory connection-discovery over the corpus"
	@echo "  make test             — run the test suite"
	@echo "  make dash             — launch the read-only dashboard (localhost)"
	@echo ""
	@echo "  Needs a data download (see the notebook):"
	@echo "    make cz-referee  /  make cz-decay  /  make rdagent-referee"
	@echo ""
	@echo "  Ingest your own claim (needs a model key):  make run ARGS='--paper inbox/<file>'"
	@echo "  Commit a proposal (human gate, you run this yourself):"
	@echo "    $(PY) -m penrose.pipeline.p9_review approve <i> --approver <you>"

# --- keyless: discrimination + calibration -------------------------------------
eval:
	$(PY) scripts/eval_suite.py
worked-example:
	$(PY) scripts/worked_example_process_conditional.py
calib-placebo:
	$(PY) scripts/calibration_placebo.py 100 0
calib-persistence:
	$(PY) scripts/calibration_persistence.py 100 0
calib-injection:
	$(PY) scripts/calibration_injection.py 15 0
calib-nulls:
	$(PY) scripts/calibration_nulls.py 60
calib-sensitivity:
	$(PY) scripts/calibration_sensitivity.py 12
calib-breadth:
	$(PY) scripts/calibration_breadth.py 12
calib-synth:
	$(PY) scripts/calibration_synthesizer.py
calib-hypothesis:
	$(PY) scripts/calibration_hypothesis_creation.py
connections:
	$(PY) scripts/brain_connections.py
test:
	$(PY) -m pytest tests/ -q

# --- pipeline (model key as noted) ---------------------------------------------
run:
	$(PY) -m penrose.pipeline.run $(ARGS)
review:
	$(PY) -m penrose.pipeline.p9_review list
dream:
	$(PY) -m penrose.cli dream $(ARGS)
synthesize:
	$(PY) -m penrose.cli synthesize $(ARGS)
stage0:
	$(PY) scripts/stage0_funding_carry.py
dash:
	$(PY) dashboard/live_server.py

# --- external data required (see the notebook) ---------------------------------
cz-referee:
	$(PY) scripts/cz_referee.py
cz-decay:
	$(PY) scripts/cz_decay.py
rdagent-referee:
	$(PY) scripts/rdagent_referee.py

# --- housekeeping --------------------------------------------------------------
reset:
	$(PY) scripts/reset.py
clean:
	rm -f backtest_ledger.tsv .holdout_burned decisions.jsonl review_queue.jsonl runs.jsonl
	rm -rf reports/*.md archives/papers/* dashboard/live.json
	@echo "cleaned run artifacts"
