# Reproducibility image for Penrose. Builds the package and runs the keyless gate.
# No API key, no external services, and (at runtime) no network are required.
#
#   docker build -t penrose .
#   docker run --rm --network none penrose      # the gate needs no network
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Build/install the package + test extra (pip needs network here for dependencies only).
RUN pip install --no-cache-dir -e '.[test]'

# Default: the keyless reproduction gate. Everything below runs with no key, no network, no data setup.
CMD ["bash", "-lc", "set -e; \
  python -c 'import penrose; print(\"import: ok\")'; \
  penrose --help >/dev/null && echo 'cli: ok'; \
  python scripts/eval_suite.py; \
  python scripts/calibration_synthesizer.py; \
  python scripts/worked_example_process_conditional.py; \
  if [ -f scripts/calibration_hypothesis_creation.py ]; then python scripts/calibration_hypothesis_creation.py; fi; \
  python -m pytest -q; \
  echo 'CLEAN-ROOM GATE PASSED'"]
