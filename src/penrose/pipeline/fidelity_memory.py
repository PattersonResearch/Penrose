"""Claim-type routing and fidelity-rejection memory for generated modules.

This module is intentionally small and deterministic. The classifier is a
fail-open heuristic: when it cannot find a clear shape cue, it returns today's
implicit default, ``trading_strategy``.
"""
from __future__ import annotations

import fcntl
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .. import config

CLAIM_TYPES = frozenset({
    "descriptive_statistical",
    "event_market_strategy",
    "trading_strategy",
    "structural_proposition",
    "provided_series_statistic",
    "predictive_regression",
    "factor_spanning",
    "cross_sectional_sort",
    "event_study",
    "forecast_skill",
})
DEFAULT_CLAIM_TYPE = "trading_strategy"

# 6g: "test the statistic of a provided/pre-computed series" is a FIRST-CLASS claim type,
# distinct from (and checked before) descriptive_statistical/trading_strategy. It fires on
# claims that declare a single pooled/cohort statistic over series the claim itself names
# (e.g. a pre-registered cohort-mean test) -- the exact shape that made spec-gen either
# invent gates the claim never stated (over-specification) or fall back to an empty
# trading-strategy stub (under-specification), because a provided-series-statistics claim
# was otherwise misrouted as trading_strategy.
_PROVIDED_SERIES_STAT_STRONG_PATTERNS = [
    r"\bone\s+(?:declared\s+)?deflation cohort\b",
    r"\bsingle\s+(?:declared\s+)?deflation cohort\b",
]
_PROVIDED_SERIES_STAT_PROVENANCE_PATTERNS = [
    r"\bdeclared series\b",
    r"\bprovided series\b",
    r"\bpre-?computed series\b",
]
_PROVIDED_SERIES_STAT_WEAK_PATTERNS = [
    r"\bpooled\s+(?:one[- ]sample\s+)?statistic\b",
    r"\bcohort[- ]level mean\b",
    r"\bcohort[- ]mean\b",
    r"\bone[- ]sample\b",
    r"\bpooled mean\b",
    r"\bsingle (?:pooled )?statistic\b",
    r"\bone (?:pooled )?statistic\b",
]
_PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS = [
    r"\balready\s+(?:pre-?)?encodes?\b.{0,40}\b(?:p&l|pnl|net|return|returns|profit|edge)\b",
    r"\bseries\b.{0,40}\balready\s+encodes?\b",
]
_PROVIDED_SERIES_DECLARATION_PATTERNS = [
    *_PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS,
    r"\b(?:provided|pre-?computed|declared|pre-?encoded)\s+(?:net\s+)?(?:p&l|pnl|return|returns|series)\b",
    r"\bpool these\s+\d*\s*(?:[\w&.-]+\s+)*series\b",
]
_TRADING_CONSTRUCTION_PATTERNS = [
    r"\bgo\s+long\b|\bgo\s+short\b|\blong[- ]short\b|\blong[- ]only\b|\bshort[- ]only\b",
    r"\bentry\b|\bexit\b",
    r"\bposition\b|\bpositions\b",
    r"\bsignal\b",
    r"\bmomentum\b",
    r"\brebalance\b|\brebalanced\b|\brebalancing\b",
]

_EVENT_MARKET_MARKET_PATTERNS = [
    r"\bevent[- ]market\b",
    r"\bprediction[- ]market\b",
    r"\bkalshi\b",
    r"\bpolymarket\b",
]
_EVENT_MARKET_BRACKET_PATTERNS = [
    r"\bbracket(?:s|ed)?\b",
    r"\bstrike(?:s)?\b",
    r"\bbinary\s+market\b",
    r"\bsettled?\s+(?:inside|within)\b",
]
_EVENT_MARKET_PRICING_PATTERNS = [
    r"\bdeclared\s+(?:bracket\s+)?pricing\s+model\b",
    r"\bnormal[_ -]bracket\b",
    r"\bpricing\s+model\b.{0,80}\b(?:mu|forecast|spot)\b.{0,40}\bsigma\b",
    r"\bphi\s*\(",
    r"\bprobability\s+model\b.{0,80}\bbracket\b",
]

_PREDICTIVE_DIRECTION_PATTERNS = [
    r"\b(?:predicts?|forecasts?|forecasting)\b",
    r"\bleads?\b.{0,60}\b(?:target|return|returns|volatility|rv|realized|outcome)\b",
    r"\b(?:target|return|returns|volatility|rv|realized|outcome)\b.{0,60}\b(?:on|against)\b.{0,30}\b(?:predictor|signal)\b",
]
_PREDICTIVE_RELATIONSHIP_PATTERNS = [
    r"\bregress(?:ion|ed|es|)\b",
    r"\bols\b|\bleast\s+squares\b|\blinear\s+regression\b",
    r"\bcoefficient\b|\bbeta\b",
    r"\bt\s*[-=]\s*-?\d+(?:\.\d+)?\b|\bt[- ]stat(?:istic)?\b",
    r"\br\s*(?:\^2|2)\b|\br-squared\b",
    r"\bmsfe\b|\brmse\b",
]
_FORECAST_SKILL_MODEL_PATTERNS = [
    r"\b(?:forecast|forecasts|forecasting|predicts?|prediction)\b",
    r"\bmodel\s+(?:forecast|prediction)\b",
]
_FORECAST_SKILL_ACCURACY_PATTERNS = [
    r"\bmsfe\b|\bmspe\b|\brmse\b",
    r"\bmean\s+squared\s+(?:forecast\s+)?error\b",
    r"\bout[- ]of[- ]sample\s+r\s*(?:\^2|2)\b|\boos\s+r\s*(?:\^2|2)\b",
    r"\bdiebold[- ]mariano\b|\bclark[- ]west\b",
]
_FORECAST_SKILL_BENCHMARK_PATTERNS = [
    r"\bbeats?\b.{0,80}\b(?:random[- ]walk|benchmark|naive|historical\s+mean)\b",
    r"\boutperform(?:s|ed)?\b.{0,80}\b(?:random[- ]walk|benchmark|naive|historical\s+mean)\b",
    r"\b(?:random[- ]walk|benchmark|naive|historical\s+mean)\b.{0,80}\b(?:forecast|model|msfe|rmse|mspe)\b",
    r"\brelative\s+(?:to|against)\b.{0,80}\b(?:random[- ]walk|benchmark|naive|historical\s+mean)\b",
    r"\b(?:msfe|mspe|rmse)\s+ratio\b",
    r"\b(?:msfe|mspe|rmse)\s*[=:]\s*0?\.\d+\b",
    r"\bout[- ]of[- ]sample\s+r\s*(?:\^2|2)\b|\boos\s+r\s*(?:\^2|2)\b",
]
_FORECAST_SKILL_OOS_PATTERNS = [
    r"\bout[- ]of[- ]sample\b|\boos\b",
    r"\bholdout\b",
]
_PREDICTIVE_HORIZON_PATTERNS = [
    r"\bnext[- ](?:day|week|period|month|quarter|year)\b",
    r"\bforward\b",
    r"\bfuture\b",
    # digit horizons: "5-day ahead", "5 day ahead", "3-week-ahead" (hyphen OR space between number and unit)
    r"\b\d+[-\s]*(?:d|day|days|week|weeks|period|periods|month|months|quarter|quarters|year|years)[- ]ahead\b",
    # spelled-out horizons: "five-day ahead", "one-month-ahead"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)[- ](?:day|week|month|quarter|year)s?[- ]?ahead\b",
    r"\bh[- ]?ahead\b",
    # "5-day horizon", "monthly horizon", "next 5 days", "horizon of 5 days"
    r"\b\d+[- ](?:day|week|month|quarter|year)s?\s+horizon\b",
    r"\b(?:daily|weekly|monthly|quarterly|annual|yearly)\s+horizon\b",
    r"\bnext\s+\d+\s+(?:days?|weeks?|months?|quarters?|years?)\b",
    r"\bhorizon\s+of\s+\d+\s+(?:days?|weeks?|months?|quarters?|years?)\b",
]
_TRADABLE_TARGET_PATTERNS = [
    r"\breturns?\b",
    r"\bexcess\s+returns?\b",
    r"\bstrategy\s+returns?\b",
    r"\bpnl\b|\bp&l\b",
    r"\bprofit(?:s|ability)?\b",
    r"\bsharpe\b",
    r"\btradeable\b|\btradable\b",
]
_PREDICTIVE_TRADING_BLOCKER_PATTERNS = [
    r"\bentry\b|\bexit\b",
    r"\bposition(?:s| sizing)?\b",
    r"\bpnl\b|\bp&l\b",
    r"\bsharpe\b",
    r"\btrade\b|\btrading\b",
    r"\bmomentum\b",
    r"\blong\b|\bshort\b",
    r"\bstrategy\b",
    r"\breturns?\b.{0,40}\b(?:tradable|tradeable|strategy|pnl|p&l|sharpe)\b",
]
_NON_TRADABLE_TARGET_PATTERNS = [
    r"\bvolatility\b|\brealized\s+vol(?:atility)?\b|\brv\b",
    r"\bvariance\b",
    r"\bvolume\b",
    r"\bflow\b|\bflows\b",
    r"\binflation\b",
    r"\boutcome\b",
    r"\btarget\b",
]
# PR3-1: the tell that a claim is a STRATEGY (route to trading_strategy) is a strategy PERFORMANCE metric
# or a trading MECHANISM — not merely the word "returns" (a return-predictability *regression* like
# "dividend yield predicts stock returns, t=3" is a genuine regression, tested as one). We route a
# structurally-clear predictive claim to predictive_regression UNLESS it is strategy-framed. This replaces
# the old closed non-tradable-target noun list (which wrongly rejected GDP-growth/unemployment/spread
# regressions) with a negative strategy-framing check.
_STRATEGY_FRAMING_PATTERNS = [
    r"\bsharpe\b",
    r"\bpnl\b|\bp&l\b",
    r"\bprofit(?:s|ability)?\b",
    r"\bcagr\b|\bannualized\s+return\b",
    r"\bdrawdown\b",
    r"\bwin[- ]?rate\b|\bhit[- ]?rate\b",
    r"\binformation\s+ratio\b",
    r"\bturnover\b",
    r"\bstrateg(?:y|ies)\b",
    r"\bposition(?:s| sizing)?\b",
    r"\bentry\b|\bexit\b",
    r"\bexcess\s+returns?\b|\bstrategy\s+returns?\b",
    r"\btradeable\b|\btradable\b",
]

_EVENT_STUDY_TIMING_PATTERNS = [
    r"\baround\b.{0,60}\b(?:event|announcement|earnings|fomc|listing|addition|halving)\b",
    r"\b(?:following|after|before)\b.{0,60}\b(?:event|announcement|earnings|fomc|listing|addition|halving)\b",
    r"\b(?:event|announcement|earnings|fomc|listing|addition|halving)\b.{0,60}\b(?:window|date|day|days)\b",
]
_EVENT_STUDY_ABNORMAL_PATTERNS = [
    r"\babnormal\s+returns?\b",
    r"\bcumulative\s+abnormal\s+returns?\b",
    r"\bcar\b",
]
_EVENT_STUDY_WINDOW_PATTERNS = [
    r"\bevent\s+window\b",
    r"\[\s*-?\d+\s*,\s*\+?\d+\s*\]\s*(?:day|days)?\s*window\b",
    r"\bwindow\s*\[\s*-?\d+\s*,\s*\+?\d+\s*\]",
]

_FACTOR_SPANNING_ALPHA_PATTERNS = [
    r"\balpha\b",
    r"\bintercept\b",
    r"\babnormal\s+returns?\b",
]
_FACTOR_SPANNING_CONTROL_PATTERNS = [
    r"\bafter\s+controlling\s+for\b",
    r"\bcontrolling\s+for\b",
    r"\bcontrol(?:s|led)?\s+for\b",
    r"\bnet\s+of\b.{0,60}\bfactors?\b",
    r"\bspanning\b|\bspanned\b",
    r"\bregress(?:ion|ed|es|)?\b.{0,80}\bon\b.{0,80}\bfactors?\b",
]
_FACTOR_SPANNING_BENCHMARK_PATTERNS = [
    r"\bcapm\b",
    r"\bff\s*3\b|\bfama[- ]french\s+3\b|\bthree[- ]factors?\b",
    r"\bff\s*5\b|\bfama[- ]french\s+5\b|\bfive[- ]factors?\b",
    r"\bcarhart\b",
    r"\bmkt[- ]rf\b|\bsmb\b|\bhml\b|\brmw\b|\bcma\b",
]

_CROSS_SECTIONAL_SORT_SORT_PATTERNS = [
    r"\bsort(?:ed|s|ing)?\s+(?:stocks?|assets?|firms?|entities|securities|the\s+cross[- ]section)?\s*(?:by|on)\b",
    r"\bportfolio(?:s)?\s+sort(?:ed|s)?\s+(?:by|on)\b",
    r"\bcross[- ]sectional\s+sort\b",
    r"\brank(?:ed|s|ing)?\s+(?:stocks?|assets?|firms?|entities|securities)?\s*(?:by|on)\b",
]
_CROSS_SECTIONAL_SORT_SPREAD_PATTERNS = [
    r"\btop[- ]minus[- ]bottom\b",
    r"\btop\s+(?:decile|quintile|bucket|portfolio).{0,60}\bbottom\b",
    r"\bbottom\s+(?:decile|quintile|bucket|portfolio).{0,60}\btop\b",
    r"\bhigh[- ]minus[- ]low\b|\bhigh\s+minus\s+low\b|\bhml\b",
    r"\bdecile\s+spread\b|\bquintile\s+spread\b|\blong[- ]short\s+spread\b",
    r"\blong[- ]short\b.{0,80}\b(?:decile|quintile|characteristic|sort|rank)\b",
]
_CROSS_SECTIONAL_CHARACTERISTIC_PATTERNS = [
    r"\bcharacteristic\b",
    r"\bbook[- ]to[- ]market\b|\bvalue\b",
    r"\bmomentum\b",
    r"\bsize\b|\bmarket\s+cap(?:italization)?\b",
    r"\bquality\b|\bprofitability\b|\binvestment\b",
    r"\baccruals?\b|\basset\s+growth\b|\bleverage\b|\bearnings\b",
]

_DESCRIPTIVE_PATTERNS = [
    r"\bunconditional\s+(?:mean|average|bias|frequency|correlation)\b",
    r"\b(?:mean|average)\s+(?:bias|return|effect|difference)\b",
    r"\bcorrelation\b",
    r"\bfrequency\b",
    r"\bfraction\b",
    r"\bpercent(?:age)?\s+of\b",
    r"\bobservations?\b",
    r"\bci\b|\bconfidence interval\b",
]
_TRADING_PATTERNS = [
    r"\bsignal\b",
    r"\bentry\b|\bexit\b",
    r"\bposition\b|\bpositions\b",
    r"\bpnl\b|\bp&l\b",
    r"\bsharpe\b",
    r"\btrade\b|\btrading\b",
    r"\bmomentum\b",
    r"\blong\b|\bshort\b",
    r"\bstrategy\b",
]
_STRUCTURAL_PATTERNS = [
    r"\bmarket structure\b",
    r"\bmicrostructure\b",
    r"\bcauses?\b",
    r"\bmechanism\b",
    r"\binstitutional\b",
    r"\bshould\b",
]

_PREREGISTERED_SINGLE_STAT_PATTERNS = [
    r"\b(?:exactly\s+)?one\s+pre[- ]?registered\s+statistic\b",
    r"\bsingle\s+pre[- ]?registered\s+statistic\b",
    r"\b(?:one|single)\s+(?:declared\s+)?deflation cohort\b",
    r"\bcounts?\s+as\s+one\s+pre[- ]?registered\s+search\b",
    r"\bpre[- ]?registered\s+search\s+denominator\s*(?:of\s+|is\s+|=\s*)1\b",
    r"\b(?:deflation\s+)?cohort\s+denominator\s*(?:of\s+|is\s+|=\s*)1\b",
]
_SINGLE_POOLED_TEST_ASSERTION = (
    r"\b(?:one|single)\s+(?:pooled\s+)?test\b|"
    r"\bpooled\s+test\b.{0,40}\b(?:one|single)\b"
)
_PREREGISTRATION_CONTEXT_PATTERN = (
    r"\bpre[- ]?registered\b|"
    r"\bdeflation cohort\b|"
    r"\bpre[- ]?registered search\b|"
    r"\b(?:exactly\s+)?one\s+pre[- ]?registered\s+statistic\b|"
    r"\bsingle\s+pre[- ]?registered\s+statistic\b"
)


def _claim_source_text(claim, source=None) -> str:
    parts = []
    for attr in ("statement", "mechanism", "claimed_metric_quote", "source_span"):
        try:
            parts.append(getattr(claim, attr, "") or "")
        except Exception:  # noqa: BLE001
            pass
    try:
        parts.append(getattr(source, "text", "") or "")
    except Exception:  # noqa: BLE001
        pass
    if isinstance(source, dict):
        try:
            parts.append(str(source.get("text") or ""))
        except Exception:  # noqa: BLE001
            pass
    return " ".join(str(p) for p in parts if p).lower()


def is_preregistered_single_cohort(claim, source=None) -> bool:
    """True only for an explicit one-cohort pre-registration assertion.

    This is verdict-integrity bookkeeping, not claim routing. Generic statistical
    prose such as "one-sample t-test" or "pooled mean" must not earn the reduced
    deflation denominator.
    """
    try:
        text = _claim_source_text(claim, source)
        if not text.strip():
            return False
        if any(re.search(pat, text) for pat in _PREREGISTERED_SINGLE_STAT_PATTERNS):
            return True
        return bool(
            re.search(r"\bno multiplicity correction\b", text)
            and re.search(_SINGLE_POOLED_TEST_ASSERTION, text)
            and re.search(_PREREGISTRATION_CONTEXT_PATTERN, text)
        )
    except Exception:  # noqa: BLE001
        return False


def _has_structural_predictive_regression_signature(text: str) -> bool:
    # Structural signature: a directional predictor->outcome relationship (direction cue) + an explicit
    # regression/coefficient cue + a forecast horizon. This is what makes a claim a *regression* rather
    # than a strategy or a descriptive statistic.
    if not any(re.search(pat, text) for pat in _PREDICTIVE_DIRECTION_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _PREDICTIVE_RELATIONSHIP_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _PREDICTIVE_HORIZON_PATTERNS):
        return False
    # Negative check: reject only when the claim is STRATEGY-FRAMED (performance metric or trading
    # mechanism). A bare "returns" target is NOT disqualifying — return-predictability regressions are
    # genuine regression claims. (PR3-1: replaces the old closed non-tradable-noun requirement, which
    # wrongly dropped GDP-growth/unemployment/credit-spread regressions.)
    if any(re.search(pat, text) for pat in _STRATEGY_FRAMING_PATTERNS):
        return False
    return True


def _has_structural_forecast_skill_signature(text: str) -> bool:
    # Forecast skill is a model-vs-benchmark accuracy comparison. It must name
    # forecast accuracy and a benchmark structure before it can bypass the
    # ordinary trading tally; single-predictor coefficient/t-stat claims remain
    # predictive_regression.
    if any(re.search(pat, text) for pat in _STRATEGY_FRAMING_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FORECAST_SKILL_MODEL_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FORECAST_SKILL_ACCURACY_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FORECAST_SKILL_BENCHMARK_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FORECAST_SKILL_OOS_PATTERNS):
        return False
    return True


def _has_structural_event_study_signature(text: str) -> bool:
    # Event-study claims are paper event-response tests, not event-triggered trading rules.
    # Require event timing + abnormal-return/CAR language + a window cue, and reject
    # strategy/performance framing so "trade after earnings" stays trading_strategy.
    if any(re.search(pat, text) for pat in _STRATEGY_FRAMING_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _EVENT_STUDY_TIMING_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _EVENT_STUDY_ABNORMAL_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _EVENT_STUDY_WINDOW_PATTERNS):
        return False
    return True


def _has_structural_factor_spanning_signature(text: str) -> bool:
    # Guarded signature: factor wording alone is too broad and often means a
    # tradeable factor strategy. Require an alpha/intercept claim plus an explicit
    # controlling-for/spanning regression structure and a named benchmark family.
    if not any(re.search(pat, text) for pat in _FACTOR_SPANNING_ALPHA_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FACTOR_SPANNING_CONTROL_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _FACTOR_SPANNING_BENCHMARK_PATTERNS):
        return False
    return True


def _has_structural_cross_sectional_sort_signature(text: str) -> bool:
    # Structural signature: a named characteristic, a sort/rank operation, and a
    # top-minus-bottom spread. Long-short strategy prose without the sort-a-
    # characteristic structure stays in the trading_strategy path.
    if any(re.search(pat, text) for pat in _STRATEGY_FRAMING_PATTERNS):
        if not (
            any(re.search(pat, text) for pat in _CROSS_SECTIONAL_SORT_SORT_PATTERNS)
            and any(re.search(pat, text) for pat in _CROSS_SECTIONAL_SORT_SPREAD_PATTERNS)
            and any(re.search(pat, text) for pat in _CROSS_SECTIONAL_CHARACTERISTIC_PATTERNS)
        ):
            return False
    if not any(re.search(pat, text) for pat in _CROSS_SECTIONAL_SORT_SORT_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _CROSS_SECTIONAL_SORT_SPREAD_PATTERNS):
        return False
    if not any(re.search(pat, text) for pat in _CROSS_SECTIONAL_CHARACTERISTIC_PATTERNS):
        return False
    return True


def classify_claim_type(claim, source=None) -> str:
    """Return a deterministic claim type, failing open to trading_strategy.

    Classification is keyword/regex-based over the ENGLISH claim text. Non-English
    claims, or unusual phrasings that match no cue, fall through to the
    `trading_strategy` default — the conservative fail-open (the claim is tested as
    a strategy rather than mis-specialized), never a crash.
    """
    try:
        claim_text = " ".join([
            getattr(claim, "statement", "") or "",
            getattr(claim, "mechanism", "") or "",
            getattr(claim, "claimed_metric_quote", "") or "",
            getattr(claim, "source_span", "") or "",
        ]).lower()
    except Exception:  # noqa: BLE001
        return DEFAULT_CLAIM_TYPE
    if not claim_text.strip():
        return DEFAULT_CLAIM_TYPE

    source_text = (getattr(source, "text", "") or "").lower()
    event_text = " ".join([claim_text, source_text])
    # M-3: the declared-pricing-model cue (the strongest signal this IS a bracket strategy) must appear
    # in the CLAIM itself, not merely somewhere in the source body — otherwise a paper whose main claim
    # is unrelated but whose data/related-work section mentions a bracket pricing model gets misrouted to
    # event_market_strategy and parks at a false needs_data. Venue+bracket cues may still come from either.
    if (
        any(re.search(pat, event_text) for pat in _EVENT_MARKET_MARKET_PATTERNS)
        and any(re.search(pat, event_text) for pat in _EVENT_MARKET_BRACKET_PATTERNS)
        and any(re.search(pat, claim_text) for pat in _EVENT_MARKET_PRICING_PATTERNS)
    ):
        return "event_market_strategy"

    if _has_structural_event_study_signature(claim_text):
        return "event_study"

    if _has_structural_factor_spanning_signature(claim_text):
        return "factor_spanning"

    if _has_structural_cross_sectional_sort_signature(claim_text):
        return "cross_sectional_sort"

    if _has_structural_forecast_skill_signature(claim_text):
        return "forecast_skill"

    # PR3-1: strategy-framing rejection now lives inside the signature check (a negative test on
    # performance/mechanism cues), so the routing is just the structural signature. This drops the old
    # standalone trade/trading/momentum/long-short blockers, which wrongly rejected genuine regression
    # claims that merely mention those words without being strategy-framed.
    if _has_structural_predictive_regression_signature(claim_text):
        return "predictive_regression"

    # Checked FIRST and independently of the descriptive/trading tally, but only for
    # explicit provided-series declarations. Ambiguous pooled/cohort/one-sample prose is
    # handled by the declaration-gated weak branch below.
    if any(re.search(pat, claim_text) for pat in _PROVIDED_SERIES_STAT_STRONG_PATTERNS):
        return "provided_series_statistic"
    if any(re.search(pat, claim_text) for pat in _PROVIDED_SERIES_STAT_PROVENANCE_PATTERNS):
        trading_construction = sum(
            1 for pat in _TRADING_CONSTRUCTION_PATTERNS if re.search(pat, claim_text)
        )
        high_confidence_encoded = any(
            re.search(pat, claim_text)
            for pat in _PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS
        )
        if trading_construction >= 2 and not high_confidence_encoded:
            return "trading_strategy"
        return "provided_series_statistic"
    stat_text = " ".join([claim_text, source_text])
    has_weak_stat = any(
        re.search(pat, stat_text) for pat in _PROVIDED_SERIES_STAT_WEAK_PATTERNS
    )
    if has_weak_stat:
        declaration_text = stat_text
        if any(re.search(pat, declaration_text) for pat in _PROVIDED_SERIES_DECLARATION_PATTERNS):
            trading_construction = sum(
                1 for pat in _TRADING_CONSTRUCTION_PATTERNS if re.search(pat, claim_text)
            )
            high_confidence_encoded = any(
                re.search(pat, declaration_text)
                for pat in _PROVIDED_SERIES_HIGH_CONFIDENCE_ENCODED_PATTERNS
            )
            if trading_construction >= 2 and not high_confidence_encoded:
                return "trading_strategy"
            return "provided_series_statistic"

    descriptive = sum(1 for pat in _DESCRIPTIVE_PATTERNS if re.search(pat, claim_text))
    trading = sum(1 for pat in _TRADING_PATTERNS if re.search(pat, claim_text))
    structural = sum(1 for pat in _STRUCTURAL_PATTERNS if re.search(pat, claim_text))

    if descriptive and descriptive >= trading:
        return "descriptive_statistical"
    if trading:
        return "trading_strategy"
    if structural:
        return "structural_proposition"
    return DEFAULT_CLAIM_TYPE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store_path() -> Path:
    return Path(getattr(config, "FIDELITY_REJECTIONS", config.REPORTS / "fidelity_rejections.jsonl"))


def append_rejection(*, strategy_class: str, claim_type: str, divergences, note: str = "") -> None:
    """Persist one faithful=false rejection using flock + tmp + replace.

    Corrupt or missing existing rows are ignored rather than surfacing into the
    pipeline. This is learning feedback, not a verdict dependency.
    """
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "strategy_class": str(strategy_class or "unspecified"),
        "claim_type": claim_type if claim_type in CLAIM_TYPES else DEFAULT_CLAIM_TYPE,
        "divergences": [str(d)[:500] for d in (divergences or []) if str(d).strip()][:5],
        "note": str(note or "")[:500],
        "ts": _now(),
    }
    lock_path = Path(str(path) + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        rows: list[str] = []
        if path.exists():
            rows = [line for line in path.read_text().splitlines() if line.strip()]
        rows.append(json.dumps(row, sort_keys=True))
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text("\n".join(rows) + "\n")
        tmp.replace(path)


def rejection_guidance(strategy_class: str, claim_type: str, *, limit: int = 3) -> str:
    """Return a capped prompt block for prior divergences, or "" fail-open."""
    path = _store_path()
    if not path.exists():
        return ""
    try:
        wanted_class = str(strategy_class or "").strip()
        wanted_type = claim_type if claim_type in CLAIM_TYPES else DEFAULT_CLAIM_TYPE
        seen: set[str] = set()
        items: list[str] = []
        for line in reversed(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("strategy_class") != wanted_class and row.get("claim_type") != wanted_type:
                continue
            for div in row.get("divergences") or []:
                div = str(div).strip()
                if div and div not in seen:
                    seen.add(div)
                    items.append(div[:300])
                if len(items) >= limit:
                    break
            if len(items) >= limit:
                break
        if not items:
            return ""
        bullets = "\n".join(f"- {item}" for item in items)
        return (
            "AVOID THESE PAST FIDELITY FAILURES for this strategy/claim type:\n"
            f"{bullets}\n"
        )
    except Exception:  # noqa: BLE001
        return ""
