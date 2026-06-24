import json
import os
import subprocess
import sys


def _role_models(env):
    code = (
        "import json;"
        "from penrose import config;"
        "print(json.dumps({"
        "'default': config.DEFAULT_LLM_MODEL,"
        "'implementer': config.LLM_ROLES['module_implementer']['model'],"
        "'refuter': config.LLM_ROLES['fidelity_refuter']['model']"
        "}))"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        text=True,
    )
    return json.loads(out)


def test_fidelity_refuter_model_is_independently_configurable():
    env = dict(os.environ)
    env["PENROSE_LLM_DEFAULT_MODEL"] = "bulk-model"
    env["PENROSE_LLM_VERIFIER_MODEL"] = "verifier-model"

    models = _role_models(env)

    assert models["default"] == "bulk-model"
    assert models["implementer"] == "bulk-model"
    assert models["refuter"] == "verifier-model"


def test_fidelity_refuter_model_defaults_unchanged_when_unset():
    env = dict(os.environ)
    env["PENROSE_LLM_DEFAULT_MODEL"] = "bulk-model"
    env.pop("PENROSE_LLM_VERIFIER_MODEL", None)

    models = _role_models(env)

    assert models["default"] == "bulk-model"
    assert models["implementer"] == "bulk-model"
    assert models["refuter"] == "bulk-model"
