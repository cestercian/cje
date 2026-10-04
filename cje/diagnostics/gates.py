"""Canonical diagnostic gate thresholds and status helpers.

Single source of truth for the CJE paper's diagnostic gates
(arXiv:2512.11150). Every surface that grades diagnostics must import
thresholds from this module so that a given number receives the same
verdict everywhere.

Diagnostic summary:

- Scalar range support: >= 5% of target judge-score mass outside the labeled
  calibration range triggers REFUSE-LEVEL for absolute claims. This check does
  not establish residual transport or the validity of policy rankings.
- Scope of calibration-map gates: a REFUSE-LEVEL badge or a residual-transport
  FAIL gates a policy only when its reported level depends on the calibration
  map (``level_gate_scope``). Complete-oracle means never do; residual-
  corrected estimates do not when their correction design check passes
  (``correction_design_check``).
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .models import Status

# ---------------------------------------------------------------------------
# Coverage badge (boundary_card)
# ---------------------------------------------------------------------------
# Fraction of target judge-score mass outside the oracle calibration range
# at or above which level claims are refused (REFUSE-LEVEL).
OUT_OF_RANGE_REFUSE_THRESHOLD = 0.05

# Fraction of calibrated rewards near the oracle reward bounds at or above
# which the card downgrades to CAUTION (boundary effects likely) when the
# out-of-range mass is below the refuse threshold.
SATURATION_CAUTION_THRESHOLD = 0.20

# Canonical boundary-card status -> per-policy Status ladder: a CAUTION card
# is a real WARNING tier between GOOD and CRITICAL (it used to leave the
# policy GOOD, making CAUTION invisible in overall_status).
BOUNDARY_CARD_STATUS_TO_STATUS: Dict[str, Status] = {
    "OK": Status.GOOD,
    "CAUTION": Status.WARNING,
    "REFUSE-LEVEL": Status.CRITICAL,
    "INCONCLUSIVE": Status.WARNING,
}

# ---------------------------------------------------------------------------
# Transport audit (audit_transportability)
# ---------------------------------------------------------------------------
# Deprecated compatibility constant. Residual audits are graded against an
# explicit, user-declared delta_max; this value is not used by the classifier.
TRANSPORT_FAIL_DELTA_THRESHOLD = 0.05

# Canonical transport status -> Status ladder for consumers that grade
# TransportDiagnostics alongside other diagnostics.
TRANSPORT_STATUS_TO_STATUS: Dict[str, Status] = {
    "PASS": Status.GOOD,
    "FAIL": Status.CRITICAL,
    "INCONCLUSIVE": Status.WARNING,
    "NOT_GRADED": Status.WARNING,
    # NOT_CHECKED means no probe was supplied at all — informational, not a
    # defect: it maps to GOOD so probe-less runs do not warn in worst-status
    # rollups. The per-policy audit record still says NOT_CHECKED explicitly.
    "NOT_CHECKED": Status.GOOD,
    # Compatibility for previously serialized diagnostics only.
    "WARN": Status.WARNING,
}


_STATUS_ORDER = {Status.GOOD: 0, Status.WARNING: 1, Status.CRITICAL: 2}


def worst_status(*statuses: Optional[Status]) -> Status:
    """Combine statuses, returning the worst. None entries are ignored.

    Returns Status.GOOD when no non-None status is provided.
    """
    worst = Status.GOOD
    for status in statuses:
        if status is not None and _STATUS_ORDER[status] > _STATUS_ORDER[worst]:
            worst = status
    return worst


# ---------------------------------------------------------------------------
# Which estimates the calibration-map gates apply to
# ---------------------------------------------------------------------------
# A transport FAIL and a REFUSE-LEVEL badge are statements about the
# calibration map. They gate a policy's level only when that level depends on
# the map. A complete-oracle mean ("direct_oracle") never does. A residual-
# corrected estimate ("augmented") does not either, provided its labelled rows
# really are the declared design's sample of the evaluation rows; then the
# exemption swaps the hard gate for the corrected interval, so it needs enough
# labels for that interval to be trusted. Evidence for the two constants:
#
# * Floor (deep dive E1, ``cov_i``, design A: one labelled row per prompt at
#   equal propensity): on the 0.9.0 analytic interval, weight-one coverage of
#   a nominal 95% interval was 0.895-0.927 at 10 labelled prompts and
#   0.932-0.942 at 20. The floor counts *effective* labelled prompts (Kish
#   count over prompt clusters of the inverse-propensity weights), which
#   equals the distinct-prompt count in E1's design and is smaller when a few
#   prompts carry most of the labels or the weight.
# * Known propensities also need a design effective sample size
#   N^2 / sum(1 / p) of at least the floor. In ``kp_rare`` (400 rows, a 10%
#   stratum at p=0.005, p=0.08 elsewhere) labelled prompts had median 29 but
#   the design effective n was 12.8 and the interval covered 0.167 of the time.
# * Balance threshold (design-check simulations): under random representative
#   labels (discrete and continuous judges, 1-2 draws per prompt, 20/30/60
#   labelled prompts) P(max |t| >= 3) was 0.1-0.65%, and 0.8-1.3% under
#   correct known propensities; labels on the top, bottom, middle band or
#   tails by score were caught in every run at 20+ labelled prompts. The check
#   is a falsification test only: it cannot see selection unrelated to the
#   judge score, and a mild exp(0.5 z) score tilt is caught only 11/28/76% of
#   the time at 20/30/60 labels.
CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS = 20
CORRECTION_DESIGN_BALANCE_T = 3.0

# Design-check failure codes that log a WARNING and add a gate note. The
# other codes (too few labels, constant labelled outcomes) only withhold the
# exemption: 0.9.0 already warns about constant labelled outcomes, and a small
# label count is visible in the interval.
_CORRECTION_WARNED_FAILURES = ("labelled_rows_unbalanced", "low_design_effective_n")


def level_gate_scope(
    route: Optional[str], check: Optional[Mapping[str, Any]]
) -> Tuple[bool, Optional[str]]:
    """Whether calibration-map gates apply to a policy's reported level.

    Returns ``(applies, scope)``. ``scope`` is ``"direct_oracle"`` or
    ``"residual_corrected"`` when the gates do not apply;
    ``"correction_unchecked"`` (no design check recorded) or
    ``"correction_not_exempt"`` (the check failed) for an augmented policy
    that keeps the gates; and None for every other route (``plug_in``,
    ``plug_in_targeted_unknown``, ``no_data``, unknown), which keep the gates
    exactly as in 0.9.0.
    """
    if route == "direct_oracle":
        return False, "direct_oracle"
    if route == "augmented":
        if check is None:
            return True, "correction_unchecked"
        if check.get("passed") is True:
            return False, "residual_corrected"
        return True, "correction_not_exempt"
    return True, None


def level_basis(route: Optional[str]) -> str:
    """What a policy's reported level rests on, from its point route."""
    return {"direct_oracle": "oracle", "augmented": "residual_corrected"}.get(
        str(route), "calibration_map"
    )


def correction_design_check(
    judge_scores: Any,
    observed: Any,
    cluster_ids: Any,
    label_design: str,
    propensities: Optional[Any] = None,
    labelled_outcomes_constant: bool = False,
) -> Dict[str, Any]:
    """Check that a residual correction's labels can lift map gates.

    Inputs are one policy's evaluation rows. The check requires at least
    ``CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS`` effective labelled prompts
    (Kish count over prompt clusters of the inverse-propensity weights), a
    known-propensity design effective sample size ``N^2 / sum(1/p)`` of at
    least the same floor, non-constant labelled outcomes, and judge-score
    balance: the Horvitz-Thompson weighted labelled rows must reproduce the
    mean and spread of the judge score over all rows (and, under known
    propensities, the row count) within ``CORRECTION_DESIGN_BALANCE_T``
    design standard errors.

    It is a falsification check, not proof: it catches labels selected by
    judge score, but not selection unrelated to the score, and it does not
    balance calibrator covariates.

    Returns a JSON-safe dict with ``design``, ``labelled_prompts``,
    ``effective_labelled_prompts``, ``design_effective_n`` (None unless
    known-propensity), ``balance_t`` ({variable: t}), ``passed`` and
    ``failed`` (failure codes, in order).
    """
    if label_design not in ("representative", "known_propensity"):
        raise ValueError(
            "correction_design_check needs label_design 'representative' or "
            f"'known_propensity', got {label_design!r}"
        )
    scores = np.asarray(judge_scores, dtype=float).reshape(-1)
    mask = np.asarray(observed, dtype=bool).reshape(-1)
    n_rows = len(scores)
    clusters = np.asarray(cluster_ids)
    if mask.shape != (n_rows,) or clusters.shape[:1] != (n_rows,):
        raise ValueError("judge_scores, observed and cluster_ids must be row-aligned")
    if n_rows == 0:
        raise ValueError("correction_design_check needs at least one row")
    if not np.all(np.isfinite(scores)):
        raise ValueError("judge_scores must be finite")
    _, cluster_index = np.unique(clusters, return_inverse=True)
    cluster_index = np.asarray(cluster_index).reshape(-1)
    n_labelled = int(np.sum(mask))

    known = label_design == "known_propensity"
    if known:
        if propensities is None:
            raise ValueError("known_propensity design check needs propensities")
        inclusion = np.asarray(propensities, dtype=float).reshape(-1)
        if inclusion.shape != (n_rows,):
            raise ValueError("propensities must align to the policy's rows")
        if not np.all(np.isfinite(inclusion)) or np.any(
            (inclusion <= 0) | (inclusion > 1)
        ):
            raise ValueError("propensities must be finite in (0, 1]")
    else:
        inclusion = np.full(n_rows, n_labelled / n_rows, dtype=float)

    weights = np.zeros(n_rows, dtype=float)
    weights[mask] = 1.0 / inclusion[mask]
    labelled_clusters = np.unique(cluster_index[mask])
    if n_labelled:
        cluster_weights = np.bincount(cluster_index, weights=weights)[labelled_clusters]
        effective = float(np.sum(cluster_weights) ** 2 / np.sum(cluster_weights**2))
    else:
        effective = 0.0
    # Round away float noise so the floor is inclusive (20 equal clusters
    # must give exactly 20, not 19.999999999999996).
    effective = round(effective, 9)
    design_effective_n = (
        round(float(n_rows**2 / np.sum(1.0 / inclusion)), 9) if known else None
    )

    spread_raw = np.abs(scores - np.median(scores))
    balance_variables: Dict[str, np.ndarray] = {}
    if known:
        balance_variables["label_count"] = np.ones(n_rows, dtype=float)
    balance_variables["judge_level"] = scores - np.mean(scores)
    balance_variables["judge_spread"] = spread_raw - np.mean(spread_raw)
    score_range = float(np.max(scores) - np.min(scores))
    balance_t: Dict[str, float] = {}
    for name, values in balance_variables.items():
        x = values
        if name != "label_count":
            # Centred deviations at float-noise size (a constant score, or a
            # spread that is identical on every row) are exactly zero.
            x = np.where(np.abs(values) <= 1e-9 * score_range, 0.0, values)
        deviation = (weights - 1.0) * x
        numerator = float(np.sum(deviation))
        if known:
            # Poisson-sampling design variance over all rows.
            variance = float(np.sum((1.0 - inclusion) / inclusion * x**2))
        else:
            # Prompt-cluster variance over all clusters.
            variance = float(np.sum(np.bincount(cluster_index, weights=deviation) ** 2))
        balance_t[name] = (
            float(numerator / np.sqrt(variance))
            if variance > 0 and np.isfinite(variance)
            else 0.0
        )

    failed: List[str] = []
    if effective < CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS:
        failed.append("too_few_effective_labelled_prompts")
    if (
        design_effective_n is not None
        and design_effective_n < CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS
    ):
        failed.append("low_design_effective_n")
    if labelled_outcomes_constant:
        failed.append("labelled_outcomes_constant")
    if max(abs(t) for t in balance_t.values()) >= CORRECTION_DESIGN_BALANCE_T:
        failed.append("labelled_rows_unbalanced")
    return {
        "design": str(label_design),
        "labelled_prompts": int(len(labelled_clusters)),
        "effective_labelled_prompts": float(effective),
        "design_effective_n": design_effective_n,
        "balance_t": balance_t,
        "passed": not failed,
        "failed": failed,
    }


def _design_words(check: Optional[Mapping[str, Any]]) -> str:
    design = check.get("design") if check is not None else None
    return "known-propensity" if design == "known_propensity" else "representative"


def _balance_text(check: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{name} t={float(value):+.2f}"
        for name, value in (check.get("balance_t") or {}).items()
    )


def corrected_gate_note(finding: str, check: Optional[Mapping[str, Any]]) -> str:
    """Why a map finding does not gate a residual-corrected policy."""
    effective = float((check or {}).get("effective_labelled_prompts", 0.0))
    return (
        f"{finding} describes the calibration map, not this estimate: the "
        f"estimate is residual-corrected with {effective:.0f} effective "
        f"labelled prompts ({_design_words(check)} design; labelled rows passed "
        "the design balance check), so it does not gate this policy"
    )


def correction_not_exempt_reason(check: Optional[Mapping[str, Any]]) -> str:
    """Extra gate reason for an augmented policy that keeps the map gates."""
    prefix = "residual correction does not lift calibration-map gates: "
    if check is None:
        return prefix + "no correction design check was recorded"
    failed = list(check.get("failed") or [])
    parts: List[str] = []
    if "too_few_effective_labelled_prompts" in failed:
        parts.append(
            f"{float(check.get('effective_labelled_prompts', 0.0)):.1f} effective "
            f"labelled prompts ({int(check.get('labelled_prompts', 0))} "
            f"labelled), fewer than {CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS}"
        )
    if "low_design_effective_n" in failed:
        parts.append(
            "known-propensity design effective sample size "
            f"{float(check.get('design_effective_n') or 0.0):.1f}, below "
            f"{CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS}"
        )
    if "labelled_outcomes_constant" in failed:
        parts.append("every labelled outcome is identical")
    if "labelled_rows_unbalanced" in failed:
        parts.append(
            "labelled rows do not match the declared "
            f"{_design_words(check)} design (judge-score balance "
            + _balance_text(check)
            + ")"
        )
    if not parts:
        parts.append("the correction design check did not pass")
    return prefix + "; ".join(parts)


def correction_caution(
    policy: str, check: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """WARNING text for a design check that failed on a warned code, or None."""
    if check is None:
        return None
    failed = list(check.get("failed") or [])
    sentences: List[str] = []
    if "labelled_rows_unbalanced" in failed:
        design = _design_words(check)
        sentences.append(
            f"its labelled rows do not look like a {design} sample of its "
            f"evaluation rows ({_balance_text(check)}). The corrected estimate "
            "assumes they are; labels chosen by score, availability or "
            "difficulty can bias it in either direction."
        )
    if "low_design_effective_n" in failed:
        text = (
            "the declared label propensities give a design effective sample "
            f"size of {float(check.get('design_effective_n') or 0.0):.1f} "
            f"(< {CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS}); a few rare "
            "high-weight rows dominate the Horvitz-Thompson correction and its "
            "interval can under-cover badly."
        )
        sentences.append(text if not sentences else "Also, " + text)
    if not sentences:
        return None
    return (
        f"Residual correction for policy {policy!r}: "
        + " ".join(sentences)
        + " Calibration-map gates still apply to this policy."
    )


def correction_check_note(check: Mapping[str, Any]) -> str:
    """Short gate note for a design check that failed on a warned code."""
    codes = ", ".join(str(code) for code in (check.get("failed") or []))
    return (
        f"correction design check failed ({codes}); see "
        "metadata['correction_checks']"
    )


def correction_check_limitation(check: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Winner limitation for a design check that failed on a warned code."""
    if check is None:
        return None
    failed = [str(code) for code in (check.get("failed") or [])]
    if not any(code in _CORRECTION_WARNED_FAILURES for code in failed):
        return None
    return f"residual correction design check failed ({', '.join(failed)})"


def transport_limitation(audit: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Limitation text for a displayed policy's transport audit, or None."""
    if not audit:
        return None
    status = str(audit.get("status", "NOT_CHECKED"))
    parts: List[str] = []
    if status != "PASS":
        if audit.get("gate_exemption") == "residual_corrected":
            parts.append(
                f"residual transport {status} (calibration map only; estimate "
                "residual-corrected)"
            )
        else:
            parts.append(f"residual transport {status}")
    if audit.get("probes_independent") is False:
        overlap = audit.get("probe_overlap") or {}
        parts.append(
            f"transport probes reuse {int(overlap.get('n_reused', 0))} labelled "
            "evaluation rows (audit not independent)"
        )
    return "; ".join(parts) if parts else None
