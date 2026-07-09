"""Deterministic execution for formulaic signal->return claims."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .predictive_regression import _observations_per_year
from .provided_series import _finite_datetime_series, _series_data


class FormulaicSignalError(ValueError):
    """Raised for invalid formula syntax or semantics."""


_ALLOWED_FUNCS = frozenset({
    "returns",
    "price",
    "funding",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "sign",
    "lag",
    "delta",
    "zscore",
})

_FUNC_ARITY = {
    "returns": 2,
    "price": 1,
    "funding": 1,
    "rolling_sum": 2,
    "rolling_mean": 2,
    "rolling_std": 2,
    "sign": 1,
    "lag": 2,
    "delta": 2,
    "zscore": 2,
}


def _as_positive_int(value: Any, func: str) -> int:
    if isinstance(value, bool):
        raise FormulaicSignalError(f"{func} window must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise FormulaicSignalError(f"{func} window must be a positive integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise FormulaicSignalError(f"{func} window must be a positive integer") from exc
    if out <= 0:
        raise FormulaicSignalError(f"{func} window must be > 0")
    return out


# FS2-1: cap AST depth so a pathologically nested claim string (e.g. "sig"+"+sig"*1000) is rejected with a
# clean formulaic_signal_invalid reason instead of blowing the Python stack into a coarse engine_error. A
# legitimate signal formula is a handful of levels deep; 100 is far above any real DSL expression.
_MAX_FORMULA_DEPTH = 100


def _validate_node(node: ast.AST, names: set[str], depth: int = 0) -> None:
    if depth > _MAX_FORMULA_DEPTH:
        raise FormulaicSignalError("formula too deeply nested")
    if isinstance(node, ast.Expression):
        _validate_node(node.body, names, depth + 1)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise FormulaicSignalError("unsupported binary operator")
        _validate_node(node.left, names, depth + 1)
        _validate_node(node.right, names, depth + 1)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise FormulaicSignalError("unsupported unary operator")
        _validate_node(node.operand, names, depth + 1)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaicSignalError("only direct function calls are allowed")
        func = node.func.id
        if func not in _ALLOWED_FUNCS:
            raise FormulaicSignalError(f"unsupported function {func}")
        if node.keywords:
            raise FormulaicSignalError("keyword arguments are not allowed")
        for arg in node.args:
            _validate_node(arg, names, depth + 1)
        return
    if isinstance(node, ast.Name):
        names.add(node.id)
        return
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FormulaicSignalError("constants must be numeric")
        if not math.isfinite(float(value)):
            raise FormulaicSignalError("constants must be finite")
        return
    raise FormulaicSignalError(f"unsupported syntax {type(node).__name__}")


def parse_formula(signal: str) -> tuple[ast.Expression, set[str]]:
    text = str(signal or "").strip()
    if not text:
        raise FormulaicSignalError("missing signal formula")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaicSignalError("formula syntax error") from exc
    except RecursionError as exc:  # S72-2: a ~3000-deep BinOp chain exhausts the parser stack BEFORE the
        raise FormulaicSignalError("formula too deeply nested") from exc  # _validate_node depth cap runs
    names: set[str] = set()
    _validate_node(tree, names)
    return tree, names


def referenced_names(signal: str) -> set[str]:
    """Return catalog/bundle series names referenced by a valid formula."""
    _, names = parse_formula(signal)
    return set(names)


def validate_signal(signal: str) -> None:
    """Validate formulaic-signal syntax and static operator arity."""
    tree, _ = parse_formula(signal)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = node.func.id
            expected = _FUNC_ARITY.get(func)
            if expected is None:
                raise FormulaicSignalError(f"unsupported function {func}")
            if len(node.args) != expected:
                raise FormulaicSignalError(f"{func} expects {expected} arguments")


def _utc_indexed(series: pd.Series) -> pd.Series:
    out = series.sort_index(kind="mergesort")
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.Series(out.to_numpy(dtype="float64"), index=idx, name=out.name)


def _load_series(bundle, name: str) -> pd.Series | None:
    raw = _series_data(bundle, name)
    clean = _finite_datetime_series(raw) if raw is not None else None
    if clean is None:
        return None
    return _utc_indexed(clean).rename(name)


def _compute_node(node: ast.AST, env: dict[str, pd.Series]):
    if isinstance(node, ast.Expression):
        return _compute_node(node.body, env)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise FormulaicSignalError(f"unknown series {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_compute_node(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = _compute_node(node.left, env)
        right = _compute_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id
        args = [_compute_node(arg, env) for arg in node.args]
        if func in {"price", "funding", "sign"}:
            if len(args) != 1:
                raise FormulaicSignalError(f"{func} expects 1 argument")
        elif func in {
            "returns",
            "rolling_sum",
            "rolling_mean",
            "rolling_std",
            "lag",
            "delta",
            "zscore",
        }:
            if len(args) != 2:
                raise FormulaicSignalError(f"{func} expects 2 arguments")
        else:
            raise FormulaicSignalError(f"unsupported function {func}")

        if func == "price":
            return args[0]
        if func == "funding":
            return args[0]
        if func == "sign":
            return np.sign(args[0])

        n = _as_positive_int(args[1], func)
        s = args[0]
        if not isinstance(s, pd.Series):
            raise FormulaicSignalError(f"{func} first argument must be a series")
        if func == "returns":
            return np.log(s / s.shift(n))
        if func == "rolling_sum":
            return s.rolling(n).sum()
        if func == "rolling_mean":
            return s.rolling(n).mean()
        if func == "rolling_std":
            return s.rolling(n).std()
        if func == "lag":
            return s.shift(n)
        if func == "delta":
            return s - s.shift(n)
        if func == "zscore":
            return (s - s.rolling(n).mean()) / s.rolling(n).std()
    raise FormulaicSignalError(f"unsupported syntax {type(node).__name__}")


def _position_map(signal: pd.Series, name: str) -> pd.Series:
    mode = str(name or "sign").strip().lower()
    if mode == "sign":
        return pd.Series(np.sign(signal), index=signal.index, name="formulaic_signal_position")
    if mode == "zscore_clip":
        return signal.clip(lower=-1.0, upper=1.0).rename("formulaic_signal_position")
    raise FormulaicSignalError(f"unsupported position_map {name}")


def _required_inputs(spec: dict, formula_names: set[str]) -> set[str]:
    out = {str(x or "").strip() for x in formula_names if str(x or "").strip()}
    trade = str(spec.get("trade_series") or "").strip()
    if trade:
        out.add(trade)
    funding = str(spec.get("funding_pnl_series") or "").strip()
    if funding:
        out.add(funding)
    return out


@dataclass
class FormulaicSignalModule:
    """Trusted module object for one declared formulaic signal spec."""

    spec: dict
    claim_id: str
    strategy_class: str

    __auto_generated__ = False
    __file__ = __file__

    def __post_init__(self) -> None:
        self.__module_id__ = str(self.spec.get("module_id") or f"formulaic_signal_{self.claim_id}")
        self.__strategy_class__ = self.strategy_class or "formulaic_signal"
        self.__strategy_class_aliases__ = [self.__strategy_class__]
        self.__description__ = "Deterministic formulaic signal executor."

    def run(self, bundle, claim, cost_frac):  # noqa: ARG002 - contract-compatible signature
        try:
            signal_text = str(self.spec.get("signal") or "").strip()
            trade_name = str(self.spec.get("trade_series") or "").strip()
            if not trade_name:
                return {"ok": False, "reason": "formulaic_signal_invalid: missing trade_series"}
            tree, formula_names = parse_formula(signal_text)
            required = _required_inputs(self.spec, formula_names)
            env: dict[str, pd.Series] = {}
            unavailable: list[str] = []
            for name in sorted(required):
                loaded = _load_series(bundle, name)
                if loaded is None:
                    unavailable.append(name)
                elif name in formula_names:
                    env[name] = loaded
            if unavailable:
                return {"ok": False, "reason": "data_unavailable: " + ", ".join(unavailable)}
            trade = _load_series(bundle, trade_name)
            if trade is None:
                return {"ok": False, "reason": f"data_unavailable: {trade_name}"}
            signal_raw = _compute_node(tree, env)
            if not isinstance(signal_raw, pd.Series):
                return {"ok": False, "reason": "formulaic_signal_invalid: signal must evaluate to a series"}
            signal = pd.to_numeric(signal_raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
            target_position = _position_map(signal, str(self.spec.get("position_map") or "sign"))

            ret = trade.pct_change().rename("formulaic_signal_trade_return")
            payoff = ret.copy()
            funding_name = str(self.spec.get("funding_pnl_series") or "").strip()
            if funding_name:
                funding_series = _load_series(bundle, funding_name)
                if funding_series is None:
                    return {"ok": False, "reason": f"data_unavailable: {funding_name}"}
                payoff = (payoff - funding_series.reindex(payoff.index)).rename("formulaic_signal_payoff")

            delayed_position = target_position.shift(1).rename("formulaic_signal_position_lag1")
            turnover = delayed_position.fillna(0.0).diff().abs().fillna(delayed_position.fillna(0.0).abs())
            cost = turnover * float(cost_frac or 0.0)
            net = (delayed_position * payoff - cost).replace([np.inf, -np.inf], np.nan).dropna()
            if net.empty:
                return {"ok": False, "reason": "data_unavailable: no_finite_formulaic_signal_trades"}
            positions = delayed_position.reindex(net.index).fillna(0.0)
            payoff = payoff.reindex(net.index)
            bars_per_year = max(1.0, _observations_per_year(pd.DatetimeIndex(net.index), len(net)))
            return {
                "ok": True,
                "net": net.rename("formulaic_signal_net"),
                "positions": positions,
                "bars_per_year": float(bars_per_year),
                "n_trades": int(len(net)),
                "payoff": payoff,
                "position_signed": positions,
                "prices": trade.reindex(net.index),
                "formulaic_signal": {
                    "signal": signal_text,
                    "trade_series": trade_name,
                    "formula_names": sorted(formula_names),
                    "inputs": sorted(required),
                    "position_map": str(self.spec.get("position_map") or "sign"),
                    "funding_pnl_series": funding_name or None,
                    "cost_frac": float(cost_frac or 0.0),
                    "structural_execution_lag": 1,
                    "n_position_changes": int((turnover.reindex(net.index).fillna(0.0) > 0.0).sum()),
                },
            }
        except FormulaicSignalError as exc:
            return {"ok": False, "reason": f"formulaic_signal_invalid: {exc}"}
        except (TypeError, ValueError) as exc:
            return {"ok": False, "reason": f"formulaic_signal_invalid: {exc}"}
        except RecursionError:  # FS2-1 belt-and-suspenders: never a raw stack trace / engine_error
            return {"ok": False, "reason": "formulaic_signal_invalid: formula too deeply nested"}


def build_module(spec: dict, claim) -> FormulaicSignalModule:
    strategy_class = (
        str(spec.get("strategy_class") or "")
        or str(getattr(claim, "applicable_strategy_class", "") or "")
        or "formulaic_signal"
    )
    return FormulaicSignalModule(
        spec=dict(spec or {}),
        claim_id=str(getattr(claim, "claim_id", "") or "unknown"),
        strategy_class=strategy_class,
    )
