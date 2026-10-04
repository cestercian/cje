"""Reliability gates respect residual correction.

A residual-transport FAIL and a REFUSE-LEVEL score-support badge describe the
calibration map. In 0.9.0 they demoted every policy whose route was not
``direct_oracle``, including augmented policies whose level is residual-
corrected and does not depend on that map. They now gate a policy only when
its level depends on the map: plug-in routes, or an augmented route whose
correction design check fails (too few effective labelled prompts, a small
known-propensity design effective sample size, constant labelled outcomes, or
labelled rows that are unbalanced on the judge score under the declared
design). Plug-in routes keep 0.9.0 behaviour exactly.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest

from cje import TransportAuditConfig, analyze_dataset, calibrated_mean_ci
from cje.data.models import EstimationResult
from cje.data.normalization import unit_scale
from cje.diagnostics import DirectDiagnostics, Status
from cje.diagnostics.gates import (
    CORRECTION_DESIGN_BALANCE_T,
    CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS,
    correction_design_check,
    correction_not_exempt_reason,
    level_basis,
)
from cje.interface.cli import best_policy_lines

DELTA = 0.03
NOT_EXEMPT = "residual correction does not lift calibration-map gates"
DESIGN_WARNING = "Residual correction for policy"
EXEMPT_SUFFIX = "  [corrected: map gates not applied]"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _anchor(
    tmp_path: Path, *, seed: int = 0, lo: float = 0.2, hi: float = 0.6, n: int = 40
) -> Path:
    """External calibration file: y = 0.2 + 0.5 s on [lo, hi]."""
    rng = np.random.default_rng(seed)
    scores = np.r_[lo, rng.uniform(lo, hi, n - 2), hi]
    labels = 0.2 + 0.5 * scores + rng.normal(0, 0.025, n)
    path = tmp_path / f"anchor-{seed}-{lo}-{hi}.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"fit-{i}",
                    "judge_score": float(s),
                    "oracle_label": float(np.clip(y, 0, 1)),
                }
            )
            + "\n"
            for i, (s, y) in enumerate(zip(scores, labels))
        )
    )
    return path


def _external(path: Path, **overrides: Any) -> Dict[str, Any]:
    """One-call options with an external calibration file (guide workflow)."""
    options: Dict[str, Any] = dict(
        calibration_data_path=str(path),
        combine_oracle_sources=False,
        calibration_judge_scale=(0, 1),
        calibration_oracle_scale=(0, 1),
        fresh_judge_scale=(0, 1),
        fresh_oracle_scale=(0, 1),
        output_scale=(0, 1),
        estimator_config={"inference_method": "cluster_robust"},
    )
    options.update(overrides)
    return options


def _probes(
    rng: np.random.Generator,
    policy: str,
    *,
    shift: Any = 0.0,
    n: int = 60,
    lo: float = 0.2,
    hi: float = 0.6,
) -> List[Dict[str, Any]]:
    """Independent probes on new prompts; ``shift`` may be a function of s."""
    shift_fn = shift if callable(shift) else (lambda s: shift)
    scores = rng.uniform(lo, hi, n)
    return [
        {
            "prompt_id": f"probe-{policy}-{j}",
            "judge_score": float(s),
            "oracle_label": float(
                np.clip(0.2 + 0.5 * s + shift_fn(s) + rng.normal(0, 0.025), 0, 1)
            ),
        }
        for j, s in enumerate(scores)
    ]


def _transport(probes: Dict[str, List[Any]]) -> TransportAuditConfig:
    return TransportAuditConfig(
        probes_by_policy=probes,
        delta_max_by_policy={policy: DELTA for policy in probes},
    )


def _rows(
    scores: Any,
    labelled: Any,
    label_fn: Callable[[float], float],
    *,
    prompt_ids: Optional[Sequence[str]] = None,
    prefix: str = "e",
) -> List[Dict[str, Any]]:
    rows = []
    for i, score in enumerate(scores):
        row: Dict[str, Any] = {
            "prompt_id": prompt_ids[i] if prompt_ids is not None else f"{prefix}{i}",
            "judge_score": float(score),
        }
        if labelled[i]:
            row["oracle_label"] = float(np.clip(label_fn(float(score)), 0, 1))
        rows.append(row)
    return rows


def _even_rank_mask(scores: np.ndarray, k: int, margin: int = 5) -> np.ndarray:
    """Label k rows at evenly spaced score ranks (balanced by construction)."""
    order = np.argsort(scores)
    ranks = np.round(np.linspace(margin, len(scores) - 1 - margin, k)).astype(int)
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[ranks]] = True
    return mask


def _guide(tmp_path: Path, *, seed: int = 42, n_lab: int = 60) -> Dict[str, Any]:
    """The guide's audit-then-correct data: external map, 60 random labels.

    The candidate's outcomes sit 0.15 above the calibration map, so its
    independent probes FAIL; its 60 uniformly sampled labels correct it.
    """
    rng = np.random.default_rng(seed)
    path = _anchor(tmp_path, seed=seed)
    scores = rng.uniform(0.2, 0.6, 200)
    outcomes = {
        "baseline": 0.2 + 0.5 * scores + rng.normal(0, 0.025, 200),
        "candidate": 0.2 + 0.5 * scores + 0.15 + rng.normal(0, 0.025, 200),
    }
    selected = rng.choice(200, size=n_lab, replace=False)
    draws: Dict[str, List[Dict[str, Any]]] = {
        policy: [
            {
                "prompt_id": f"eval-{i}",
                "response_id": f"{policy}:eval-{i}",
                "judge_score": float(s),
            }
            for i, s in enumerate(scores)
        ]
        for policy in outcomes
    }
    for policy, rows in draws.items():
        for i in selected:
            rows[i]["oracle_label"] = float(np.clip(outcomes[policy][i], 0, 1))
    probes = {
        "baseline": _probes(rng, "baseline"),
        "candidate": _probes(rng, "candidate", shift=0.15),
    }
    reused = {policy: [dict(draws[policy][i]) for i in selected] for policy in draws}
    truth = {policy: float(np.mean(values)) for policy, values in outcomes.items()}
    return {
        "path": path,
        "draws": draws,
        "probes": probes,
        "reused": reused,
        "truth": truth,
    }


def _s1(tmp_path: Path, *, seed: int = 7) -> Dict[str, Any]:
    """Convenience labels: the candidate's 40 lowest-scored rows (default design).

    The map overstates the candidate's high-score rows by 0.25, so the
    labels on its low-score rows see no residual and the corrected estimate
    is biased upward; independent probes FAIL.
    """
    rng = np.random.default_rng(seed)
    path = _anchor(tmp_path, seed=seed, hi=0.7, n=60)
    n = 300
    s_c = rng.uniform(0.25, 0.65, n)
    s_b = rng.uniform(0.2, 0.6, n)
    y_c = np.clip(0.2 + 0.5 * s_c - 0.25 * (s_c > 0.45) + rng.normal(0, 0.025, n), 0, 1)
    y_b = np.clip(0.2 + 0.5 * s_b + rng.normal(0, 0.025, n), 0, 1)
    lab_c = np.zeros(n, dtype=bool)
    lab_c[np.argsort(s_c)[:40]] = True
    lab_b = np.zeros(n, dtype=bool)
    lab_b[rng.choice(n, 40, replace=False)] = True
    draws = {
        "candidate": _rows(s_c, lab_c, lambda s: 0.0, prefix="c"),
        "baseline": _rows(s_b, lab_b, lambda s: 0.0, prefix="b"),
    }
    for rows, labelled, outcome in (
        (draws["candidate"], lab_c, y_c),
        (draws["baseline"], lab_b, y_b),
    ):
        for i, row in enumerate(rows):
            if labelled[i]:
                row["oracle_label"] = float(outcome[i])
    probes = {
        "candidate": _probes(
            rng, "candidate", shift=lambda s: -0.25 * (s > 0.45), lo=0.25, hi=0.65
        ),
        "baseline": _probes(rng, "baseline"),
    }
    truth = {"candidate": float(y_c.mean()), "baseline": float(y_b.mean())}
    return {"path": path, "draws": draws, "probes": probes, "truth": truth}


def _s2(
    tmp_path: Path,
    *,
    seed: int = 11,
    candidate_shift: float = 0.06,
    baseline_probe_shift: Optional[float] = None,
) -> EstimationResult:
    """Corrected candidate vs plug-in baseline sharing a map offset.

    Both policies' outcomes sit above the map (candidate +0.06, baseline
    +0.12): the candidate's labels correct its level, the baseline's
    plug-in level keeps the offset.
    """
    rng = np.random.default_rng(seed)
    path = _anchor(tmp_path, seed=seed)
    n = 300
    s_c = rng.uniform(0.2, 0.6, n)
    s_b = rng.uniform(0.2, 0.6, n)
    y_c = 0.2 + 0.5 * s_c + candidate_shift + rng.normal(0, 0.025, n)
    lab_c = np.zeros(n, dtype=bool)
    lab_c[rng.choice(n, 60, replace=False)] = True
    prompts = [f"q{i}" for i in range(n)]
    candidate_rows = _rows(s_c, lab_c, lambda s: 0.0, prompt_ids=prompts)
    for i, row in enumerate(candidate_rows):
        if lab_c[i]:
            row["oracle_label"] = float(np.clip(y_c[i], 0, 1))
    draws = {
        "candidate": candidate_rows,
        "baseline": _rows(
            s_b, np.zeros(n, dtype=bool), lambda s: 0.0, prompt_ids=prompts
        ),
    }
    probes = {"candidate": _probes(rng, "candidate", shift=candidate_shift)}
    if baseline_probe_shift is not None:
        probes["baseline"] = _probes(rng, "baseline", shift=baseline_probe_shift)
    return analyze_dataset(
        fresh_draws_data=draws, transport=_transport(probes), **_external(path)
    )


def _single_policy_fail(
    tmp_path: Path,
    rows: List[Dict[str, Any]],
    *,
    seed: int = 5,
    probe_shift: float = 0.1,
    probe_label: Optional[float] = None,
    **overrides: Any,
) -> EstimationResult:
    rng = np.random.default_rng(seed)
    probes = _probes(rng, "candidate", shift=probe_shift)
    if probe_label is not None:
        for probe in probes:
            probe["oracle_label"] = probe_label
    return analyze_dataset(
        fresh_draws_data={"candidate": rows},
        transport=_transport({"candidate": probes}),
        **_external(_anchor(tmp_path, seed=seed), **overrides),
    )


def _status(result: EstimationResult, policy: str) -> Status:
    assert result.diagnostics is not None
    assert result.diagnostics.status_per_policy is not None
    return result.diagnostics.status_per_policy[policy]


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


# ---------------------------------------------------------------------------
# Corrected policies are not demoted by a calibration-map FAIL
# ---------------------------------------------------------------------------


def test_transport_fail_does_not_demote_corrected_policy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _guide(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(data["probes"]),
            **_external(data["path"]),
        )

    checks = result.metadata["correction_checks"]
    assert checks["candidate"]["passed"] is True
    assert checks["baseline"]["passed"] is True
    assert checks["candidate"]["effective_labelled_prompts"] == 60
    assert result.metadata["level_basis"] == {
        "baseline": "residual_corrected",
        "candidate": "residual_corrected",
    }

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["route"] == "augmented"
    assert audit["applies_to_current_estimate"] is False
    assert audit["gate_exemption"] == "residual_corrected"
    assert audit["probes_independent"] is True
    # The map-level rollup still reports the FAIL.
    assert result.metadata["transport_status"] == "FAIL"

    gate = result.gates["candidate"]
    assert gate.flagged is False
    assert gate.refuse_level_claims is False
    assert gate.reasons == []
    assert gate.exemption == "residual_corrected"
    assert len(gate.notes) == 1
    assert gate.notes[0].startswith("residual transport FAIL (simultaneous CI [")
    assert "describes the calibration map, not this estimate" in gate.notes[0]
    assert "60 effective labelled prompts (representative design" in gate.notes[0]

    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.GOOD
    assert result.diagnostics.overall_status != Status.CRITICAL

    lo, hi = result.confidence_interval()
    index = result.target_policies.index("candidate")
    assert lo[index] <= data["truth"]["candidate"] <= hi[index]
    verdict = result.best_policy()
    assert verdict.name == "candidate"
    assert verdict.runner_up is None
    comparison = result.compare_policies(index, 1 - index)
    assert comparison["gate_flagged"] == []
    assert "map_offset_uncancelled" not in comparison  # baseline corrected too

    text = result.summary()
    candidate_line = next(
        line for line in text.splitlines() if line.startswith("  candidate")
    )
    assert candidate_line.endswith(EXEMPT_SUFFIX)
    assert f"    note: {gate.notes[0]}" in text.splitlines()
    assert (
        "Limitations: residual transport FAIL (calibration map only; estimate "
        "residual-corrected)" in text
    )
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        "Limitations: residual transport FAIL (calibration map only; estimate "
        "residual-corrected)",
    ]
    assert not any(
        "best_policy()" in message for message in _messages(caplog, logging.WARNING)
    )


def test_convenience_labels_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """S1: labels on the candidate's lowest-scored rows under the default design.

    Exempting this policy (the draft rule) would report 0.43 against a truth
    of 0.30 as unflagged and crown it; the design check keeps 0.9.0 gating.
    """
    data = _s1(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(data["probes"]),
            **_external(data["path"]),
        )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_rows_unbalanced"]
    assert check["balance_t"]["judge_level"] < -CORRECTION_DESIGN_BALANCE_T
    assert result.metadata["correction_checks"]["baseline"]["passed"] is True

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert audit["gate_exemption"] is None
    gate = result.gates["candidate"]
    assert gate.flagged is True
    assert gate.exemption is None
    assert len(gate.reasons) == 2
    assert gate.reasons[0].startswith(
        "residual transport FAIL: simultaneous CI ["
    ) and gate.reasons[0].endswith(f"is outside margin +/-{DELTA:.3f}")
    assert gate.reasons[1].startswith(
        f"{NOT_EXEMPT}: labelled rows do not match the declared representative "
        "design (judge-score balance judge_level t=-"
    )
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.CRITICAL
    assert result.metadata["level_basis"] == {
        "baseline": "residual_corrected",
        "candidate": "calibration_map",
    }
    # The design-check note is recorded, but summary() prints notes only for
    # gates that did not flag the policy.
    assert gate.notes == [
        "correction design check failed (labelled_rows_unbalanced); see "
        "metadata['correction_checks']"
    ]
    assert "    note: " not in result.summary()

    verdict = result.best_policy()
    assert verdict.name == "baseline"
    assert verdict.runner_up == "candidate"
    design_warnings = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(design_warnings) == 1
    assert design_warnings[0].startswith(
        "Residual correction for policy 'candidate': its labelled rows do not look "
        "like a representative sample of its evaluation rows (judge_level t=-"
    )
    assert design_warnings[0].endswith(
        "Calibration-map gates still apply to this policy."
    )


def test_design_check_warns_without_map_finding(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _s1(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data={"candidate": data["draws"]["candidate"]},
            **_external(data["path"]),
        )

    gate = result.gates["candidate"]
    assert gate.flagged is False
    assert gate.exemption is None
    assert gate.notes == [
        "correction design check failed (labelled_rows_unbalanced); see "
        "metadata['correction_checks']"
    ]
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.GOOD
    assert result.best_policy().name == "candidate"
    assert (
        sum(m.startswith(DESIGN_WARNING) for m in _messages(caplog, logging.WARNING))
        == 1
    )

    text = result.summary()
    assert "    note: correction design check failed (labelled_rows_unbalanced)" in text
    assert EXEMPT_SUFFIX not in text
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        "Limitations: residual transport NOT_CHECKED; residual correction design "
        "check failed (labelled_rows_unbalanced)",
    ]


# ---------------------------------------------------------------------------
# What the design check requires
# ---------------------------------------------------------------------------


def _floor_rows(rng: np.random.Generator, layout: str) -> List[Dict[str, Any]]:
    """Rows with 0.1 above the map and labels balanced on the score."""

    def outcome(s: float) -> float:
        return float(0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))

    if layout in ("19", "20"):
        scores = rng.uniform(0.2, 0.6, 300)
        return _rows(scores, _even_rank_mask(scores, int(layout)), outcome)
    if layout == "20_rows_10_prompts":
        prompts = [f"p{j}" for j in range(150) for _ in range(2)]
        scores = rng.uniform(0.2, 0.6, 300)
        labelled = np.zeros(300, dtype=bool)
        for j in np.round(np.linspace(3, 146, 10)).astype(int):
            labelled[2 * j : 2 * j + 2] = True
        return _rows(scores, labelled, outcome, prompt_ids=prompts)
    assert layout == "one_prompt_30_draws"
    prompts = ["big"] * 30 + [f"p{j}" for j in range(270)]
    scores = rng.uniform(0.2, 0.6, 300)
    labelled = np.zeros(300, dtype=bool)
    labelled[:30] = True
    labelled[30 + np.round(np.linspace(3, 266, 19)).astype(int)] = True
    return _rows(scores, labelled, outcome, prompt_ids=prompts)


@pytest.mark.parametrize(
    "layout, exempt, effective",
    [
        ("19", False, 19.0),
        ("20", True, 20.0),
        ("20_rows_10_prompts", False, 10.0),
        ("one_prompt_30_draws", False, 2.6),
    ],
)
def test_below_effective_floor_keeps_gate(
    tmp_path: Path, layout: str, exempt: bool, effective: float
) -> None:
    rows = _floor_rows(np.random.default_rng(13), layout)
    result = _single_policy_fail(tmp_path, rows)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["effective_labelled_prompts"] == pytest.approx(effective, abs=0.05)
    assert result.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    gate = result.gates["candidate"]
    assert result.diagnostics is not None
    status = _status(result, "candidate")
    if exempt:
        assert check["passed"] is True
        assert gate.flagged is False
        assert gate.exemption == "residual_corrected"
        assert status == Status.GOOD
        return
    assert check["failed"] == ["too_few_effective_labelled_prompts"]
    assert gate.flagged is True
    assert status == Status.CRITICAL
    assert len(gate.reasons) == 2
    assert gate.reasons[1].startswith(f"{NOT_EXEMPT}: ")
    assert f"fewer than {CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS}" in gate.reasons[1]
    if layout == "19":
        assert gate.reasons[1] == (
            f"{NOT_EXEMPT}: 19.0 effective labelled prompts (19 labelled), "
            "fewer than 20"
        )
        # Too few labels only withholds the exemption: no design-check
        # limitation, note or warning, with or without an audit.
        design_limitation = "residual correction design check failed"
        assert gate.notes == []
        assert design_limitation not in result.summary()
        assert not any(design_limitation in line for line in best_policy_lines(result))
        no_audit = analyze_dataset(
            fresh_draws_data={"candidate": rows},
            **_external(_anchor(tmp_path, seed=5)),
        )
        assert no_audit.metadata["correction_checks"]["candidate"]["failed"] == [
            "too_few_effective_labelled_prompts"
        ]
        assert (
            "candidate" not in (no_audit.metadata.get("reliability_gates") or {})
            or not no_audit.gates["candidate"].notes
        )
        assert design_limitation not in no_audit.summary()
        assert best_policy_lines(no_audit) == [
            "Best by point estimate: candidate",
            "Limitations: residual transport NOT_CHECKED",
        ]


def test_constant_labelled_outcomes_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A binary oracle whose labelled outcomes are all 1 cannot lift the gate."""
    scores = np.random.default_rng(17).uniform(0.2, 0.6, 300)
    rows = _rows(scores, _even_rank_mask(scores, 40), lambda s: 1.0)
    with caplog.at_level(logging.INFO):
        result = _single_policy_fail(tmp_path, rows, probe_label=1.0)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_outcomes_constant"]
    gate = result.gates["candidate"]
    assert gate.flagged is True
    assert gate.reasons[-1] == f"{NOT_EXEMPT}: every labelled outcome is identical"
    # 0.9.0's provisional-interval warning covers this case; no design warning.
    warnings_logged = _messages(caplog, logging.WARNING)
    assert any("Every labelled outcome is identical" in m for m in warnings_logged)
    assert not any(m.startswith(DESIGN_WARNING) for m in warnings_logged)


# ---------------------------------------------------------------------------
# Plug-in routes keep 0.9.0 behaviour exactly
# ---------------------------------------------------------------------------


def _plug_in_data(tmp_path: Path, *, labelled: bool) -> Dict[str, Any]:
    rng = np.random.default_rng(23)
    n = 200
    s_b = rng.uniform(0.2, 0.5, n)
    s_c = rng.uniform(0.3, 0.6, n)
    mask = np.zeros(n, dtype=bool)
    if labelled:
        mask[rng.choice(n, 60, replace=False)] = True
    draws = {
        "baseline": _rows(s_b, mask, lambda s: 0.2 + 0.5 * s, prefix="b"),
        "candidate": _rows(s_c, mask, lambda s: 0.2 + 0.5 * s + 0.1, prefix="c"),
    }
    probes = {"candidate": _probes(rng, "candidate", shift=0.1)}
    return {"path": _anchor(tmp_path, seed=23), "draws": draws, "probes": probes}


@pytest.mark.parametrize(
    "case, route",
    [
        ("no_labels", "plug_in"),
        ("augmentation_off", "plug_in"),
        ("targeted_unknown", "plug_in_targeted_unknown"),
    ],
)
def test_plug_in_routes_unchanged(tmp_path: Path, case: str, route: str) -> None:
    data = _plug_in_data(tmp_path, labelled=case != "no_labels")
    overrides: Dict[str, Any] = {}
    if case == "augmentation_off":
        overrides["estimator_config"] = {
            "inference_method": "cluster_robust",
            "use_augmented_estimator": False,
        }
    if case == "targeted_unknown":
        overrides["label_design"] = "targeted_unknown"
    result = analyze_dataset(
        fresh_draws_data=data["draws"],
        transport=_transport(data["probes"]),
        **_external(data["path"], **overrides),
    )

    assert result.metadata["point_estimator"]["routes"] == [route, route]
    assert "correction_checks" not in result.metadata
    assert result.metadata["level_basis"] == {
        "baseline": "calibration_map",
        "candidate": "calibration_map",
    }
    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert audit["gate_exemption"] is None
    assert audit["route"] == route
    lo_d, hi_d = audit["delta_ci"]
    fail_reason = (
        f"residual transport FAIL: simultaneous CI [{lo_d:+.3f}, {hi_d:+.3f}] "
        f"is outside margin +/-{DELTA:.3f}"
    )
    # 0.9.0 gate dict, byte for byte (no notes or exemption keys).
    assert result.metadata["reliability_gates"]["candidate"] == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [fail_reason],
    }
    assert result.metadata["reliability_gates"]["baseline"] == {
        "flagged": False,
        "refused": False,
        "refuse_level_claims": False,
        "reasons": [],
    }
    assert result.diagnostics is not None
    assert _status(result, "candidate") == Status.CRITICAL
    verdict = result.best_policy()
    assert (verdict.name, verdict.runner_up) == ("baseline", "candidate")
    comparison = result.compare_policies(0, 1)
    assert comparison["gate_flagged"] == ["candidate"]
    assert "map_offset_uncancelled" not in comparison

    lo, hi = result.confidence_interval()
    est = result.estimates
    demoted = (
        "Best reliable policy: baseline — raw argmax candidate was flagged "
        f"({fail_reason}; diagnostics status CRITICAL); pass reliable_only=False "
        "for the raw argmax"
    )
    assert result.summary() == "\n".join(
        [
            f"CJE Estimation Results (method: {result.method})",
            f"  baseline   {est[0]:.3f}  95% CI [{lo[0]:.3f}, {hi[0]:.3f}]",
            f"  candidate  {est[1]:.3f}  95% CI [{lo[1]:.3f}, {hi[1]:.3f}]"
            "  [gate: FLAGGED]",
            "Best by point estimate: candidate",
            "Limitations: flagged by the reliability gates; residual transport FAIL",
            demoted,
            "Status: critical",
        ]
    )
    assert best_policy_lines(result) == [
        "Best by point estimate: candidate",
        "Limitations: reliability gates flagged this policy; residual transport FAIL",
        demoted,
    ]


def test_transport_fail_does_not_gate_direct_oracle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A complete-oracle mean ignores a map FAIL on its held-out probes."""
    rng = np.random.default_rng(79)
    n = 200
    scores = rng.uniform(0.2, 0.6, n)
    draws = {
        "oracle": _rows(
            scores,
            np.ones(n, dtype=bool),
            lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02),
            prefix="o",
        ),
        "baseline": _rows(
            rng.uniform(0.2, 0.6, n), np.zeros(n, dtype=bool), lambda s: 0.0
        ),
    }
    probes = {"oracle": _probes(rng, "oracle", shift=0.1)}
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data=draws,
            transport=_transport(probes),
            **_external(_anchor(tmp_path, seed=79)),
        )

    index = result.target_policies.index("oracle")
    assert result.metadata["point_estimator"]["routes"][index] == "direct_oracle"
    assert result.metadata["level_basis"]["oracle"] == "oracle"
    audit = result.metadata["transport_audits"]["oracle"]
    assert audit["status"] == "FAIL"
    assert audit["route"] == "direct_oracle"
    assert audit["applies_to_current_estimate"] is False
    assert audit["gate_exemption"] == "direct_oracle"
    gates = result.metadata.get("reliability_gates") or {}
    assert not (gates.get("oracle") or {}).get("flagged", False)
    assert not (gates.get("oracle") or {}).get("refuse_level_claims", False)
    assert "oracle" not in result._unreliable_policies()
    assert _status(result, "oracle") == Status.GOOD
    verdict = result.best_policy()
    assert verdict.name == "oracle"
    assert verdict.runner_up is None
    assert not any(
        "best_policy()" in message for message in _messages(caplog, logging.WARNING)
    )


# ---------------------------------------------------------------------------
# Known-propensity designs
# ---------------------------------------------------------------------------


def test_known_propensity_correction_is_exempt(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    propensities = 0.2 + 0.2 * rng.uniform(size=n)
    observed = rng.uniform(size=n) < propensities
    assert observed.sum() >= 60
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    result = _single_policy_fail(
        tmp_path,
        rows,
        label_design="known_propensity",
        label_propensities={"candidate": propensities.tolist()},
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["design"] == "known_propensity"
    assert check["passed"] is True
    assert check["design_effective_n"] >= CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS
    assert set(check["balance_t"]) == {"label_count", "judge_level", "judge_spread"}
    assert result.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    gate = result.gates["candidate"]
    assert gate.flagged is False
    assert gate.exemption == "residual_corrected"
    assert "(known-propensity design;" in gate.notes[0]


def _rare_stratum_rows(rng: np.random.Generator) -> Dict[str, Any]:
    """kp_rare layout: a 10% stratum at p=0.005 (unsampled), p=0.08 elsewhere."""
    n = 400
    scores = rng.uniform(0.2, 0.6, n)
    propensities = np.where(np.arange(n) < 40, 0.005, 0.08)
    observed = np.zeros(n, dtype=bool)
    rest = 40 + np.argsort(scores[40:])
    observed[rest[np.round(np.linspace(3, 356, 29)).astype(int)]] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    return {"rows": rows, "propensities": propensities.tolist()}


def test_known_propensity_low_design_n_keeps_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _rare_stratum_rows(np.random.default_rng(29))
    options: Dict[str, Any] = dict(
        label_design="known_propensity",
        label_propensities={"candidate": data["propensities"]},
    )
    result = _single_policy_fail(tmp_path, data["rows"], **options)

    check = result.metadata["correction_checks"]["candidate"]
    assert check["labelled_prompts"] == 29
    assert check["design_effective_n"] == pytest.approx(12.8)
    assert check["failed"] == ["low_design_effective_n"]
    gate = result.gates["candidate"]
    assert gate.flagged is True
    assert gate.reasons[-1] == (
        f"{NOT_EXEMPT}: known-propensity design effective sample size 12.8, " "below 20"
    )

    # The design warning is logged even without any map finding.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        no_audit = analyze_dataset(
            fresh_draws_data={"candidate": data["rows"]},
            **_external(_anchor(tmp_path, seed=5), **options),
        )
    assert no_audit.gates["candidate"].flagged is False
    warnings_logged = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(warnings_logged) == 1
    assert "design effective sample size of 12.8 (< 20)" in warnings_logged[0]


def test_known_propensity_misdeclared_is_caught(tmp_path: Path) -> None:
    """Labels taken from the top of the score range, declared as p = 0.2."""
    rng = np.random.default_rng(31)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[np.argsort(scores)[-60:]] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    result = _single_policy_fail(
        tmp_path,
        rows,
        label_design="known_propensity",
        label_propensities={"candidate": [0.2] * n},
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["failed"] == ["labelled_rows_unbalanced"]
    assert check["balance_t"]["label_count"] == pytest.approx(0.0, abs=1e-9)
    assert check["balance_t"]["judge_level"] > CORRECTION_DESIGN_BALANCE_T
    assert result.gates["candidate"].flagged is True


def test_known_propensity_certain_unlabelled_rows_keep_gate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Propensity 1 on every row ("no reweighting") with 40 random labels.

    Propensity 1 means "always labelled", so the 260 unlabelled rows
    contradict the design, and the correction stays near the map's level.
    """
    rng = np.random.default_rng(67)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 40, replace=False)] = True
    rows = _rows(scores, observed, lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02))
    with caplog.at_level(logging.INFO):
        result = _single_policy_fail(
            tmp_path,
            rows,
            label_design="known_propensity",
            label_propensities={"candidate": [1.0] * n},
        )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["design_effective_n"] == n
    assert check["effective_labelled_prompts"] == 40
    assert check["unlabelled_certain_rows"] == 260
    assert check["failed"] == [
        "unlabelled_rows_declared_certain",
        "labelled_rows_unbalanced",
    ]
    assert check["balance_t"]["label_count"] < -CORRECTION_DESIGN_BALANCE_T
    assert result.metadata["level_basis"] == {"candidate": "calibration_map"}

    audit = result.metadata["transport_audits"]["candidate"]
    assert audit["status"] == "FAIL"
    assert audit["applies_to_current_estimate"] is True
    assert audit["gate_exemption"] is None
    gate = result.gates["candidate"]
    assert gate.flagged is True
    assert gate.refuse_level_claims is True
    assert gate.reasons[-1].startswith(
        f"{NOT_EXEMPT}: 260 unlabelled rows have a declared label propensity of 1 "
        "(certain to be labelled); labelled rows do not match the declared "
        "known-propensity design"
    )
    assert _status(result, "candidate") == Status.CRITICAL
    design_warnings = [
        m for m in _messages(caplog, logging.WARNING) if m.startswith(DESIGN_WARNING)
    ]
    assert len(design_warnings) == 1
    assert design_warnings[0].startswith(
        "Residual correction for policy 'candidate': 260 unlabelled rows have a "
        "declared label propensity of 1 (certain to be labelled), which the "
        "declared design rules out"
    )

    # Propensity 1 on the labelled rows only is a valid declaration.
    unit = correction_design_check(
        scores,
        observed,
        np.arange(n),
        "known_propensity",
        propensities=np.where(observed, 1.0, 0.1),
    )
    assert unit["unlabelled_certain_rows"] == 0
    assert "unlabelled_rows_declared_certain" not in unit["failed"]


def _whole_prompt_rows(
    seed: int, *, draws: int = 4, propensity: float = 0.25, n_prompts: int = 150
) -> List[Dict[str, Any]]:
    """Known-propensity design that labels every draw of a sampled prompt."""
    rng = np.random.default_rng(seed)
    prompts = [f"q{j}" for j in range(n_prompts) for _ in range(draws)]
    scores = np.clip(
        np.repeat(rng.uniform(0.2, 0.6, n_prompts), draws)
        + rng.normal(0, 0.03, n_prompts * draws),
        0.2,
        0.6,
    )
    sampled = np.repeat(rng.uniform(size=n_prompts) < propensity, draws)
    rows = _rows(
        scores,
        sampled,
        lambda s: 0.2 + 0.5 * s + 0.1 + rng.normal(0, 0.02),
        prompt_ids=prompts,
    )
    for i, row in enumerate(rows):
        row["draw_idx"] = i % draws
    return rows


def test_known_propensity_whole_prompt_labels_are_exempt(tmp_path: Path) -> None:
    """Prompt-level sampling, 4 draws each, correctly declared p = 0.25.

    A row-level Poisson variance gave judge_level t = -3.09 here (34 effective
    labelled prompts) and kept the gate; the cluster variance does not.
    """
    rows = _whole_prompt_rows(9)
    rng = np.random.default_rng(9)
    probes = _probes(rng, "candidate", shift=0.1)
    result = analyze_dataset(
        fresh_draws_data={"candidate": rows},
        transport=_transport({"candidate": probes}),
        **_external(
            _anchor(tmp_path, seed=5),
            label_design="known_propensity",
            label_propensities={"candidate": [0.25] * len(rows)},
        ),
    )

    check = result.metadata["correction_checks"]["candidate"]
    assert check["effective_labelled_prompts"] == pytest.approx(34.0)
    assert check["passed"] is True
    assert max(abs(t) for t in check["balance_t"].values()) < 2.0
    assert result.metadata["transport_audits"]["candidate"]["status"] == "FAIL"
    assert result.gates["candidate"].flagged is False
    assert result.gates["candidate"].exemption == "residual_corrected"


@pytest.mark.parametrize("prompt_level", [True, False], ids=["prompts", "rows"])
@pytest.mark.parametrize("draws", [2, 4])
def test_known_propensity_balance_false_fail_rate(
    draws: int, prompt_level: bool
) -> None:
    """Correct known propensities rarely fail balance, whole prompts or rows.

    A row-level Poisson variance failed 7% (2 draws) and 27% (4 draws) of
    whole-prompt designs; the cluster variance stays near 1%.
    """
    n_prompts, propensity = 150, 0.25
    clusters = np.repeat(np.arange(n_prompts), draws)
    failures = 0
    for seed in range(200):
        rng = np.random.default_rng([seed, draws])
        scores = np.clip(
            0.7 * np.repeat(rng.uniform(size=n_prompts), draws)
            + 0.3 * rng.uniform(size=n_prompts * draws),
            0,
            1,
        )
        if prompt_level:
            observed = np.repeat(rng.uniform(size=n_prompts) < propensity, draws)
        else:
            observed = rng.uniform(size=n_prompts * draws) < propensity
        check = correction_design_check(
            scores,
            observed,
            clusters,
            "known_propensity",
            propensities=np.full(len(scores), propensity),
        )
        failures += "labelled_rows_unbalanced" in check["failed"]
    assert failures <= 6


@pytest.mark.slow
def test_design_check_false_positive_rate() -> None:
    """Random representative labels at the floor rarely fail balance."""
    failures = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(size=200)
        observed = np.zeros(200, dtype=bool)
        observed[
            rng.choice(200, CORRECTION_EXEMPT_MIN_LABELLED_PROMPTS, replace=False)
        ] = True
        check = correction_design_check(
            scores, observed, np.arange(200), "representative"
        )
        failures += "labelled_rows_unbalanced" in check["failed"]
        assert check == correction_design_check(
            scores, observed, np.arange(200), "representative"
        )
    assert failures <= 6


def test_design_check_is_json_safe_and_deterministic() -> None:
    scores = np.linspace(0.2, 0.6, 30)
    observed = np.r_[np.ones(20), np.zeros(10)].astype(bool)
    check = correction_design_check(
        scores, observed, [f"p{i}" for i in range(30)], "representative"
    )
    assert json.loads(json.dumps(check)) == check
    assert check == correction_design_check(
        scores, observed, [f"p{i}" for i in range(30)], "representative"
    )
    # The existing transport fixture: labels on the 20 lowest of 30 rows.
    assert check["balance_t"]["judge_level"] == pytest.approx(-4.05, abs=0.01)
    assert check["failed"] == ["labelled_rows_unbalanced"]
    with pytest.raises(ValueError, match="label_design"):
        correction_design_check(scores, observed, np.arange(30), "targeted_unknown")


@pytest.mark.parametrize(
    "case", ["two_point_0.3_0.7", "two_point_0.1_0.9", "constant", "constant_kp"]
)
def test_design_check_float_noise_guard(case: str) -> None:
    """Centred values that are float noise give t == 0, not a noise ratio.

    A two-point symmetric judge has the same |s - median| on every row, and a
    constant judge has no deviation at all; without the guard their t values
    are ratios of rounding noise (judge_spread t = -0.77 for the first case).
    """
    rng = np.random.default_rng(71)
    n = 300
    if case.startswith("two_point"):
        lo, hi = (float(v) for v in case.split("_")[2:])
        scores = np.r_[np.full(n // 2, lo), np.full(n // 2, hi)]
        variables = ["judge_spread"]
    else:
        scores = np.full(n, 0.1)
        variables = ["judge_level", "judge_spread"]
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 37, replace=False)] = True
    options: Dict[str, Any] = {}
    design = "representative"
    if case == "constant_kp":
        design = "known_propensity"
        options["propensities"] = np.full(n, 0.13)
    check = correction_design_check(scores, observed, np.arange(n), design, **options)
    for variable in variables:
        assert check["balance_t"][variable] == 0.0
    assert "labelled_rows_unbalanced" not in check["failed"]
    assert check["passed"] is True


def test_design_check_without_design_variance_fails_balance() -> None:
    """A deviation whose design variance overflows is unbalanced, t None."""
    rng = np.random.default_rng(73)
    n = 300
    scores = rng.uniform(0.2, 0.6, n)
    observed = np.zeros(n, dtype=bool)
    observed[rng.choice(n, 40, replace=False)] = True
    propensities = np.full(n, 0.2)
    propensities[np.flatnonzero(observed)[0]] = 1e-200
    check = correction_design_check(
        scores, observed, np.arange(n), "known_propensity", propensities=propensities
    )
    assert check["balance_t"]["label_count"] is None
    assert "labelled_rows_unbalanced" in check["failed"]
    assert check["effective_labelled_prompts"] == 1.0
    # Strict JSON (no NaN or Infinity).
    assert json.loads(json.dumps(check, allow_nan=False)) == check
    reason = correction_not_exempt_reason(check)
    assert "label_count t undefined (no design variance)" in reason


def test_level_basis_follows_gate_scope() -> None:
    passed = {"passed": True, "failed": []}
    failed = {"passed": False, "failed": ["labelled_rows_unbalanced"]}
    assert level_basis("direct_oracle") == "oracle"
    assert level_basis("augmented", passed) == "residual_corrected"
    assert level_basis("augmented", failed) == "calibration_map"
    assert level_basis("augmented", None) == "calibration_map"
    assert level_basis("plug_in", passed) == "calibration_map"
    assert level_basis("plug_in_targeted_unknown") == "calibration_map"
    assert level_basis(None) == "calibration_map"


# ---------------------------------------------------------------------------
# Score-support badge (boundary card)
# ---------------------------------------------------------------------------


def _badge_data(labels_on: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fresh-only: policy 'corrected' labels 30 of 300 rows, 'uncorrected' none."""
    rng = np.random.default_rng(37)
    n = 300
    s_a = rng.uniform(0, 1, n)
    s_b = rng.uniform(0, 1, n)
    if labels_on == "even":
        # Evenly spaced score ranks 10..289: >= 5% of rows fall outside the
        # labelled (calibration) range, and the labels are balanced.
        order = np.argsort(s_a)
        mask = np.zeros(n, dtype=bool)
        mask[order[np.round(np.linspace(10, 289, 30)).astype(int)]] = True
    else:
        mask = np.zeros(n, dtype=bool)
        mask[np.argsort(s_a)[-30:]] = True
    return {
        "corrected": _rows(
            s_a, mask, lambda s: 0.1 + 0.6 * s + rng.normal(0, 0.02), prefix="a"
        ),
        "uncorrected": _rows(s_b, np.zeros(n, dtype=bool), lambda s: 0.0, prefix="b"),
    }


def test_refuse_level_badge_exempt_for_corrected_policy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=_badge_data("even"),
            estimator_config={"inference_method": "cluster_robust"},
        )
    diagnostics = result.diagnostics
    assert diagnostics is not None and diagnostics.boundary_cards is not None

    card = diagnostics.boundary_cards["corrected"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["out_of_range"] >= 0.05
    assert card["applies_to_current_estimate"] is False
    assert card["gate_exemption"] == "residual_corrected"
    gate = result.gates["corrected"]
    assert gate.flagged is False
    assert gate.exemption == "residual_corrected"
    assert gate.notes[0].startswith(
        f"boundary REFUSE-LEVEL ({card['out_of_range']:.1%} of judge scores outside "
        "the oracle calibration range) describes the calibration map"
    )
    assert _status(result, "corrected") == Status.GOOD
    assert "corrected" not in diagnostics.refuse_level_policies
    assert not any("for corrected:" in issue for issue in diagnostics.validate())
    warnings_logged = _messages(caplog, logging.WARNING)
    assert not any("REFUSE-LEVEL for policy 'corrected'" in m for m in warnings_logged)
    assert any(
        m.startswith("boundary REFUSE-LEVEL") for m in _messages(caplog, logging.INFO)
    )

    # The plug-in policy keeps 0.9.0 behaviour.
    other = diagnostics.boundary_cards["uncorrected"]
    assert other["status"] == "REFUSE-LEVEL"
    assert other["applies_to_current_estimate"] is True
    assert other["gate_exemption"] is None
    assert result.metadata["reliability_gates"]["uncorrected"] == {
        "flagged": True,
        "refused": False,
        "refuse_level_claims": True,
        "reasons": [
            f"boundary: {other['out_of_range']:.1%} of judge scores outside the "
            "oracle calibration range"
        ],
    }
    assert _status(result, "uncorrected") == Status.CRITICAL
    assert diagnostics.refuse_level_policies == ["uncorrected"]
    assert any("REFUSE-LEVEL for policy 'uncorrected'" in m for m in warnings_logged)


def test_refuse_level_with_unbalanced_labels_keeps_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=_badge_data("top"),
            estimator_config={"inference_method": "cluster_robust"},
        )
    diagnostics = result.diagnostics
    assert diagnostics is not None and diagnostics.boundary_cards is not None
    card = diagnostics.boundary_cards["corrected"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["applies_to_current_estimate"] is True
    assert card["gate_exemption"] is None
    assert _status(result, "corrected") == Status.CRITICAL
    gate = result.gates["corrected"]
    assert gate.flagged is True
    assert len(gate.reasons) == 2
    assert gate.reasons[0] == (
        f"boundary: {card['out_of_range']:.1%} of judge scores outside the oracle "
        "calibration range"
    )
    assert gate.reasons[1].startswith(f"{NOT_EXEMPT}: labelled rows do not match")
    warnings_logged = _messages(caplog, logging.WARNING)
    assert any("REFUSE-LEVEL for policy 'corrected'" in m for m in warnings_logged)
    assert any(
        m.startswith("Residual correction for policy 'corrected'")
        for m in warnings_logged
    )


def test_direct_oracle_card_not_listed(tmp_path: Path) -> None:
    unit = DirectDiagnostics(
        estimator_type="Direct",
        method="direct_oracle",
        n_samples_total=10,
        n_samples_valid=10,
        policies=["pi"],
        estimates={"pi": 0.5},
        standard_errors={"pi": 0.01},
        n_samples_used={"pi": 10},
        boundary_cards={
            "pi": {
                "status": "REFUSE-LEVEL",
                "out_of_range": 0.12,
                "applies_to_current_estimate": False,
                "gate_exemption": "direct_oracle",
            }
        },
    )
    assert unit.refuse_level_policies == []
    assert not any("REFUSE-LEVEL" in issue for issue in unit.validate())

    # End to end: a fully labelled policy outside a narrow external map.
    rng = np.random.default_rng(41)
    scores = rng.uniform(0.2, 0.6, 50)
    rows = _rows(scores, np.ones(50, dtype=bool), lambda s: 0.2 + 0.5 * s)
    result = analyze_dataset(
        fresh_draws_data={"oracle": rows},
        **_external(_anchor(tmp_path, seed=41, lo=0.4, hi=0.5)),
    )
    assert result.diagnostics is not None
    assert result.diagnostics.boundary_cards is not None
    card = result.diagnostics.boundary_cards["oracle"]
    assert card["status"] == "REFUSE-LEVEL"
    assert card["applies_to_current_estimate"] is False
    assert card["gate_exemption"] == "direct_oracle"
    assert result.diagnostics.refuse_level_policies == []
    assert not any("REFUSE-LEVEL" in issue for issue in result.diagnostics.validate())
    assert result.metadata["level_basis"] == {"oracle": "oracle"}
    assert "correction_checks" not in result.metadata


# ---------------------------------------------------------------------------
# calibrated_mean_ci
# ---------------------------------------------------------------------------


def _array_case(case: str) -> Dict[str, Any]:
    rng = np.random.default_rng(43)
    scores = rng.uniform(0, 1, 400)
    outcomes = np.clip(0.1 + 0.6 * scores + rng.normal(0, 0.03, 400), 0, 1)
    mask = np.zeros(400, dtype=bool)
    if case == "random_30":
        mask[rng.choice(400, 30, replace=False)] = True
    elif case == "top_30":
        mask[np.argsort(scores)[-30:]] = True
    else:
        mask[rng.choice(400, 12, replace=False)] = True
    return {"scores": scores, "labels": np.where(mask, outcomes, np.nan), "mask": mask}


@pytest.mark.parametrize(
    "case, applies, estimate",
    [
        ("random_30", False, 0.38663548),
        ("top_30", True, 0.65535997),
        ("few_12", True, 0.39765386),
    ],
)
def test_calibrated_mean_ci_badge_scope(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    case: str,
    applies: bool,
    estimate: float,
) -> None:
    data = _array_case(case)
    with caplog.at_level(logging.INFO):
        result = calibrated_mean_ci(data["scores"], data["labels"], data["mask"])

    card = result.diagnostics["boundary_card"]
    check = result.diagnostics["correction_check"]
    assert card["applies_to_current_estimate"] is applies
    assert check["passed"] is (not applies)
    warnings_logged = _messages(caplog, logging.WARNING)
    refuse_warned = any("REFUSE-LEVEL" in m for m in warnings_logged)
    design_warned = any(m.startswith(DESIGN_WARNING) for m in warnings_logged)
    if case == "random_30":
        assert card["status"] == "REFUSE-LEVEL" and card["out_of_range"] >= 0.05
        assert card["gate_exemption"] == "residual_corrected"
        assert not refuse_warned and not design_warned
        assert any(
            m.startswith("boundary REFUSE-LEVEL")
            for m in _messages(caplog, logging.INFO)
        )
    elif case == "top_30":
        assert card["status"] == "REFUSE-LEVEL"
        assert refuse_warned and design_warned
        assert "labelled_rows_unbalanced" in check["failed"]
    else:
        assert check["failed"] == ["too_few_effective_labelled_prompts"]
        assert not design_warned

    # The estimate equals 0.9.0's, and the gate never moves the numbers:
    # forcing the opposite design verdict leaves estimate and interval intact.
    assert result.estimate == pytest.approx(estimate, abs=1e-7)
    flipped = dict(check, passed=applies, failed=[] if applies else ["forced"])
    monkeypatch.setattr(
        "cje.array_api.correction_design_check", lambda *a, **k: flipped
    )
    forced = calibrated_mean_ci(data["scores"], data["labels"], data["mask"])
    assert forced.diagnostics["boundary_card"]["applies_to_current_estimate"] is (
        not applies
    )
    assert forced.estimate == result.estimate
    assert forced.ci == result.ci
    assert forced.se == result.se


# ---------------------------------------------------------------------------
# Rankings that mix a lifted FAIL with map-based levels
# ---------------------------------------------------------------------------


def test_mixed_basis_caution(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """S2: a lifted FAIL beside a plug-in baseline that shares the map offset."""
    result = _s2(tmp_path)
    audits = result.metadata["transport_audits"]
    assert audits["candidate"]["status"] == "FAIL"
    assert audits["candidate"]["gate_exemption"] == "residual_corrected"
    assert audits["baseline"]["status"] == "NOT_CHECKED"
    assert result.metadata["level_basis"] == {
        "baseline": "calibration_map",
        "candidate": "residual_corrected",
    }

    # No demotion: the caution is a stated limitation, not a gate.
    verdict = result.best_policy()
    assert verdict.name == "candidate"
    assert verdict.runner_up is None
    assert result.gates["candidate"].flagged is False

    with caplog.at_level(logging.WARNING, logger="cje.data.models"):
        comparison = result.compare_policies(1, 0)
    assert comparison["gate_flagged"] == []
    assert comparison["map_offset_uncancelled"] == ["baseline"]
    cautions = [m for m in _messages(caplog, logging.WARNING) if "map offset" in m]
    assert cautions == [
        "compare_policies: 'candidate' is residual-corrected but 'baseline' rests "
        "on the calibration map, which failed its residual audit for "
        "'candidate'; a shared map offset does not cancel in this difference, so "
        "its sign is not established until 'baseline' is audited (PASS) or "
        "corrected."
    ]
    assert result.compare_policies(0, 1)["map_offset_uncancelled"] == ["baseline"]

    limitation = (
        "ranking against map-based 'baseline' not established (the calibration "
        "map failed its residual audit for 'candidate'; that offset does not "
        "cancel against a map-based level)"
    )
    assert limitation in result.summary()
    assert limitation in best_policy_lines(result)[1]
    assert result._ranking_basis_limitation("baseline") == (
        "'baseline' rests on a calibration map that failed its residual audit "
        "for 'candidate'"
    )

    # Serialized pairwise state never stores the key; a restored result
    # re-derives it from the saved audits and level bases.
    payload = result.to_dict(detail="portable")
    stored = payload["pairwise_inference_state"]["comparisons"]
    assert stored and all("map_offset_uncancelled" not in c for c in stored.values())
    restored = EstimationResult.from_dict(json.loads(json.dumps(payload)))
    assert restored.compare_policies(1, 0)["map_offset_uncancelled"] == ["baseline"]


@pytest.mark.parametrize(
    "candidate_shift, baseline_probe_shift",
    [(0.06, 0.0), (0.0, None)],
    ids=["baseline_audit_passes", "no_lifted_fail"],
)
def test_mixed_basis_caution_negative_cases(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    candidate_shift: float,
    baseline_probe_shift: Optional[float],
) -> None:
    result = _s2(
        tmp_path,
        candidate_shift=candidate_shift,
        baseline_probe_shift=baseline_probe_shift,
    )
    audits = result.metadata["transport_audits"]
    if baseline_probe_shift is not None:
        assert audits["baseline"]["status"] == "PASS"
        assert audits["candidate"]["status"] == "FAIL"
    else:
        assert audits["candidate"]["status"] == "PASS"
    with caplog.at_level(logging.WARNING, logger="cje.data.models"):
        comparison = result.compare_policies(1, 0)
    assert "map_offset_uncancelled" not in comparison
    assert not any("map offset" in m for m in _messages(caplog, logging.WARNING))
    assert "ranking against map-based" not in result.summary()
    assert not any("map-based" in line for line in best_policy_lines(result))


def test_mixed_basis_caution_includes_failed_correction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An augmented policy whose design check fails is map-based for rankings."""
    rng = np.random.default_rng(83)
    path = _anchor(tmp_path, seed=83)
    n = 300
    prompts = [f"q{i}" for i in range(n)]
    draws: Dict[str, List[Dict[str, Any]]] = {}
    for policy, labels in (
        ("corrected", "random"),
        ("low_labels", "lowest"),
        ("base", ""),
    ):
        scores = rng.uniform(0.2, 0.6, n)
        mask = np.zeros(n, dtype=bool)
        if labels == "random":
            mask[rng.choice(n, 60, replace=False)] = True
        elif labels == "lowest":
            mask[np.argsort(scores)[:60]] = True
        draws[policy] = _rows(
            scores,
            mask,
            lambda s: 0.2 + 0.5 * s + 0.06 + rng.normal(0, 0.025),
            prompt_ids=prompts,
        )
    probes = {"corrected": _probes(rng, "corrected", shift=0.06)}
    result = analyze_dataset(
        fresh_draws_data=draws, transport=_transport(probes), **_external(path)
    )

    checks = result.metadata["correction_checks"]
    assert checks["corrected"]["passed"] is True
    assert "labelled_rows_unbalanced" in checks["low_labels"]["failed"]
    assert result.metadata["level_basis"] == {
        "base": "calibration_map",
        "corrected": "residual_corrected",
        "low_labels": "calibration_map",
    }
    audits = result.metadata["transport_audits"]
    assert audits["corrected"]["gate_exemption"] == "residual_corrected"
    assert audits["low_labels"]["status"] == "NOT_CHECKED"
    assert not result._unreliable_policies()

    index = result.target_policies.index
    with caplog.at_level(logging.WARNING, logger="cje.data.models"):
        comparison = result.compare_policies(index("corrected"), index("low_labels"))
    assert comparison["map_offset_uncancelled"] == ["low_labels"]
    assert any(
        "'low_labels' rests on the calibration map" in m
        for m in _messages(caplog, logging.WARNING)
    )
    assert result.compare_policies(index("corrected"), index("base"))[
        "map_offset_uncancelled"
    ] == ["base"]
    assert "map_offset_uncancelled" not in result.compare_policies(
        index("low_labels"), index("base")
    )
    assert result.best_policy().name == "corrected"
    limitation = "ranking against map-based 'base', 'low_labels' not established"
    assert limitation in result.summary()
    assert limitation in best_policy_lines(result)[1]


# ---------------------------------------------------------------------------
# Probes that reuse labelled evaluation rows
# ---------------------------------------------------------------------------


def test_probe_overlap_with_correction_labels_warns_and_marks(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _guide(tmp_path)
    with caplog.at_level(logging.INFO):
        result = analyze_dataset(
            fresh_draws_data=data["draws"],
            transport=_transport(data["reused"]),
            **_external(data["path"]),
        )

    audits = result.metadata["transport_audits"]
    for policy in ("baseline", "candidate"):
        assert audits[policy]["probe_overlap"] == {
            "n_reused": 60,
            "n_reused_rows": 60,
            "n_calibration_fit": 0,
            "n_ambiguous": 0,
            "n_unkeyed": 0,
            "matched_by": ["prompt_and_response_id"],
        }
        assert audits[policy]["probes_independent"] is False
        overlap_warnings = [
            m
            for m in _messages(caplog, logging.WARNING)
            if m.startswith(f"Transport probes for policy {policy!r}")
        ]
        assert len(overlap_warnings) == 1
        assert (
            "not independent validation of the corrected estimate"
            in overlap_warnings[0]
        )
        assert "may also have been used" not in overlap_warnings[0]

    # Statuses and the exemption are unchanged by the overlap.
    assert audits["candidate"]["status"] == "FAIL"
    assert audits["candidate"]["gate_exemption"] == "residual_corrected"
    assert result.gates["candidate"].flagged is False
    assert result.best_policy().name == "candidate"
    assert (
        "Limitations: residual transport FAIL (calibration map only; estimate "
        "residual-corrected); transport probes reuse 60 labelled evaluation rows "
        "(audit not independent)" in result.summary()
    )


def _fresh_only(n_prompts: int = 200, draws: int = 1) -> Dict[str, Any]:
    rng = np.random.default_rng(47)
    n = n_prompts * draws
    scores = rng.uniform(0.2, 0.6, n)
    prompts = [f"p{j}" for j in range(n_prompts) for _ in range(draws)]
    labelled = np.zeros(n, dtype=bool)
    for j in rng.choice(n_prompts, 60, replace=False):
        labelled[j * draws : (j + 1) * draws] = True
    rows = _rows(
        scores,
        labelled,
        lambda s: 0.2 + 0.5 * s + rng.normal(0, 0.02),
        prompt_ids=prompts,
    )
    return {"rows": rows, "labelled": labelled}


def test_probe_in_calibration_fit_by_key_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = _fresh_only()
    probes = [
        {key: row[key] for key in ("prompt_id", "judge_score", "oracle_label")}
        for row, labelled in zip(data["rows"], data["labelled"])
        if labelled
    ]
    config = _transport({"policy": probes})
    with pytest.raises(ValueError, match="used to fit the calibrator") as raised:
        analyze_dataset(
            fresh_draws_data={"policy": data["rows"]},
            estimator_config={"inference_method": "cluster_robust"},
            transport=config,
        )
    assert "combine_oracle_sources=False" in str(raised.value)
    assert "matched by prompt_score_and_label" in str(raised.value)

    # The guide's route: external map, labels attached for correction only.
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data={"policy": data["rows"]},
            transport=config,
            **_external(_anchor(tmp_path)),
        )
    overlap = result.metadata["transport_audits"]["policy"]["probe_overlap"]
    assert overlap["n_reused"] == 60
    assert overlap["n_calibration_fit"] == 0
    assert overlap["matched_by"] == ["prompt_score_and_label"]
    assert any(
        m.startswith("Transport probes for policy 'policy'")
        for m in _messages(caplog, logging.WARNING)
    )


def test_probe_overlap_ambiguous_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Two draws per prompt and no draw_idx: an ambiguous in-sample match warns."""
    data = _fresh_only(n_prompts=150, draws=2)
    probes = [
        {key: row[key] for key in ("prompt_id", "judge_score", "oracle_label")}
        for row, labelled in zip(data["rows"][::2], data["labelled"][::2])
        if labelled
    ]
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data={"policy": data["rows"]},
            estimator_config={"inference_method": "cluster_robust"},
            transport=_transport({"policy": probes}),
        )
    overlap = result.metadata["transport_audits"]["policy"]["probe_overlap"]
    assert overlap["n_reused"] == len(probes)
    assert overlap["n_ambiguous"] == len(probes)
    assert overlap["n_calibration_fit"] == len(probes)
    message = next(
        m
        for m in _messages(caplog, logging.WARNING)
        if m.startswith("Transport probes for policy 'policy'")
    )
    assert "may also have been used to fit the calibrator" in message
    assert "biased toward PASS" in message


@pytest.mark.parametrize(
    "case, route",
    [
        ("augmented", "augmented"),
        ("plug_in", "plug_in"),
        ("targeted_unknown", "plug_in_targeted_unknown"),
        ("augmented_external", "augmented"),
        ("explicit_draw_idx", "augmented"),
    ],
)
def test_new_responses_on_labelled_prompts_are_independent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str, route: str
) -> None:
    """Probes that are new responses for labelled prompts are held out.

    One draw per prompt; each probe has a labelled prompt's id but its own
    judge score and oracle label, and no response id or text. It is not the
    evaluation row, so it neither raises (the labels fitted the calibrator
    unless the fit is external) nor marks the audit dependent.
    """
    data = _fresh_only()
    rng = np.random.default_rng(89)
    probes = []
    for row, labelled in zip(data["rows"], data["labelled"]):
        if labelled:
            score = float(rng.uniform(0.2, 0.6))
            probe = {
                "prompt_id": row["prompt_id"],
                "judge_score": score,
                "oracle_label": float(0.2 + 0.5 * score + rng.normal(0, 0.02)),
            }
            if case == "explicit_draw_idx":
                probe["draw_idx"] = 0
            probes.append(probe)
    options: Dict[str, Any] = {
        "estimator_config": {"inference_method": "cluster_robust"}
    }
    if case == "plug_in":
        options["estimator_config"]["use_augmented_estimator"] = False
    if case == "targeted_unknown":
        options["label_design"] = "targeted_unknown"
    if case == "augmented_external":
        options = _external(_anchor(tmp_path))
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data={"policy": data["rows"]},
            transport=_transport({"policy": probes}),
            **options,
        )

    assert result.metadata["point_estimator"]["routes"] == [route]
    audit = result.metadata["transport_audits"]["policy"]
    assert audit["probe_overlap"]["n_reused"] == 0
    assert audit["probe_overlap"]["n_reused_rows"] == 0
    assert audit["probes_independent"] is True
    assert not any(
        m.startswith("Transport probes for policy")
        for m in _messages(caplog, logging.WARNING)
    )
    assert "reuse" not in result.summary()


def test_duplicate_probes_count_distinct_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Each labelled row supplied twice: 120 matching probes, 60 rows reused."""
    data = _fresh_only()
    probes = [
        {key: row[key] for key in ("prompt_id", "judge_score", "oracle_label")}
        for row, labelled in zip(data["rows"], data["labelled"])
        if labelled
    ]
    with caplog.at_level(logging.WARNING):
        result = analyze_dataset(
            fresh_draws_data={"policy": data["rows"]},
            transport=_transport({"policy": probes + probes}),
            **_external(_anchor(tmp_path)),
        )
    overlap = result.metadata["transport_audits"]["policy"]["probe_overlap"]
    assert overlap["n_reused"] == 120
    assert overlap["n_reused_rows"] == 60
    message = next(
        m
        for m in _messages(caplog, logging.WARNING)
        if m.startswith("Transport probes for policy 'policy'")
    )
    assert message.startswith(
        "Transport probes for policy 'policy': 120 of 120 match the policy's own "
        "labelled evaluation rows (60 distinct rows; matched by "
        "prompt_score_and_label)."
    )
    assert "transport probes reuse 60 labelled evaluation rows" in result.summary()


def _key_rule_rows(
    *,
    draws: int = 1,
    response: bool = False,
    response_id: bool = False,
    observation_id: bool = False,
    prompt_prefix: str = "p",
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(53)
    n_prompts = 100
    rows = []
    for j in range(n_prompts):
        for d in range(draws):
            score = float(rng.uniform(0.2, 0.6))
            row: Dict[str, Any] = {
                "prompt_id": f"{prompt_prefix}{j:06d}",
                "judge_score": score,
            }
            if j % 5 < 2:
                row["oracle_label"] = float(0.2 + 0.5 * score + rng.normal(0, 0.02))
            if response:
                row["response"] = f"answer {j} {d}"
            if response_id:
                row["response_id"] = f"r-{j}-{d}"
            if observation_id:
                row["observation_id"] = f"obs-{j}-{d}"
            rows.append(row)
    return rows


def _overlap_for(
    tmp_path: Path, rows: List[Dict[str, Any]], probes: List[Any]
) -> Dict[str, Any]:
    result = analyze_dataset(
        fresh_draws_data={"policy": rows},
        transport=_transport({"policy": probes}),
        **_external(_anchor(tmp_path)),
    )
    audit: Dict[str, Any] = result.metadata["transport_audits"]["policy"]
    return audit


def test_probe_overlap_key_rules(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from cje.data.fresh_draws import fresh_draws_from_dict

    def labelled_copies(
        rows: List[Dict[str, Any]], **changes: Any
    ) -> List[Dict[str, Any]]:
        copies = []
        for row in rows:
            if "oracle_label" in row:
                probe = dict(row)
                for key, value in changes.items():
                    probe[key] = value(row) if callable(value) else value
                copies.append(probe)
        return copies

    # (a) Same prompt, different response text: not the same row.
    rows = _key_rule_rows(response=True)
    audit = _overlap_for(
        tmp_path, rows, labelled_copies(rows, response=lambda r: "other text")
    )
    assert audit["probe_overlap"]["n_reused"] == 0
    assert audit["probes_independent"] is True
    # ... and the same response text is.
    audit = _overlap_for(tmp_path, rows, labelled_copies(rows))
    assert audit["probe_overlap"]["matched_by"] == ["prompt_and_response_text"]

    # (b) Same prompt, different response_id: not the same row.
    rows = _key_rule_rows(response_id=True)
    audit = _overlap_for(
        tmp_path, rows, labelled_copies(rows, response_id=lambda r: "other")
    )
    assert audit["probe_overlap"]["n_reused"] == 0

    # (c) FreshDrawSample probes match by explicit (source_id, row_id) ...
    rows = _key_rule_rows(draws=2)
    datasets, _ = fresh_draws_from_dict(
        {"policy": rows}, judge_scale=(0, 1), oracle_scale=(0, 1)
    )
    samples = [s for s in datasets["policy"].samples if s.oracle_label is not None]
    audit = _overlap_for(tmp_path, rows, samples)
    assert audit["probe_overlap"]["n_reused"] == len(samples)
    assert audit["probe_overlap"]["matched_by"] == ["source_row_id"]
    # ... and, without row ids, keep their draw_idx: exact, never ambiguous.
    stripped = [
        s.model_copy(update={"source_id": None, "row_id": None}) for s in samples
    ]
    audit = _overlap_for(tmp_path, rows, stripped)
    assert audit["probe_overlap"]["n_reused"] == len(samples)
    assert audit["probe_overlap"]["n_ambiguous"] == 0
    assert audit["probe_overlap"]["matched_by"] == ["prompt_and_draw_idx"]

    # (d) observation_id identifies the row whatever the prompt id says.
    rows = _key_rule_rows(observation_id=True)
    audit = _overlap_for(
        tmp_path,
        rows,
        labelled_copies(rows, prompt_id=lambda r: "other-" + r["prompt_id"]),
    )
    assert audit["probe_overlap"]["matched_by"] == ["observation_id"]
    assert audit["probe_overlap"]["n_reused"] == 40

    # (e) New prompts: independent, no warning.
    rows = _key_rule_rows()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        audit = _overlap_for(
            tmp_path,
            rows,
            labelled_copies(rows, prompt_id=lambda r: "new-" + r["prompt_id"]),
        )
    assert audit["probe_overlap"]["n_reused"] == 0
    assert audit["probes_independent"] is True
    assert not any(
        m.startswith("Transport probes for policy")
        for m in _messages(caplog, logging.WARNING)
    )

    # (f) Probes without prompt_id or prompt text get index-fallback ids,
    # which are never key-matched even when an evaluation row is named alike.
    rows = _key_rule_rows(prompt_prefix="fresh_policy_")
    probes = [
        {"judge_score": row["judge_score"], "oracle_label": row["oracle_label"]}
        for row in rows
        if "oracle_label" in row
    ]
    audit = _overlap_for(tmp_path, rows, probes)
    assert audit["probe_overlap"]["n_reused"] == 0
    assert audit["probe_overlap"]["n_unkeyed"] == len(probes)
    assert audit["probes_independent"] is True

    # (g) Same prompt and values but a different explicit draw_idx: the row
    # cannot be identified exactly, so the match is ambiguous (never raises).
    rows = _key_rule_rows()
    audit = _overlap_for(tmp_path, rows, labelled_copies(rows, draw_idx=1))
    assert audit["probe_overlap"]["n_reused"] == 40
    assert audit["probe_overlap"]["n_ambiguous"] == 40
    assert audit["probe_overlap"]["matched_by"] == ["prompt_score_and_label"]

    # (h) Same prompt and explicit draw_idx, but a different judge score or
    # oracle label: a different observation, not the row.
    for field in ("judge_score", "oracle_label"):
        audit = _overlap_for(
            tmp_path,
            rows,
            labelled_copies(rows, draw_idx=0, **{field: lambda r: r[field] / 2}),
        )
        assert audit["probe_overlap"]["n_reused"] == 0
        assert audit["probes_independent"] is True


# ---------------------------------------------------------------------------
# Private entry point and serialization
# ---------------------------------------------------------------------------


class _IdentityCalibrator:
    covariate_names: List[str] = []

    def predict(self, scores: Any, covariates: Optional[Any] = None) -> np.ndarray:
        return np.asarray(scores, dtype=float)


def test_private_attach_without_evaluation_rows() -> None:
    from cje.interface.analysis import _attach_transport_audits

    policies = ["augmented_unchecked", "plug_in"]
    result = EstimationResult(
        estimates=np.array([0.4, 0.4]),
        standard_errors=np.array([0.01, 0.01]),
        n_samples_used={policy: 30 for policy in policies},
        method="calibrated_direct",
        influence_functions=None,
        diagnostics=None,
        calibrator=_IdentityCalibrator(),
        metadata={
            "target_policies": policies,
            "point_estimator": {"routes": ["augmented", "plug_in"]},
        },
    )
    rng = np.random.default_rng(59)
    config = _transport(
        {policy: _probes(rng, policy, shift=0.2) for policy in policies}
    )
    _attach_transport_audits(
        result,
        config=config,
        target_policies=policies,
        calibration_dataset=None,
        judge_field="judge_score",
        oracle_field="oracle_label",
        oracle_input_scale=unit_scale(),
        output_scale=unit_scale(),
    )

    audits = result.metadata["transport_audits"]
    for policy in policies:
        assert audits[policy]["status"] == "FAIL"
        assert audits[policy]["probe_overlap"] is None
        assert audits[policy]["probes_independent"] is None
        assert audits[policy]["applies_to_current_estimate"] is True
        assert audits[policy]["gate_exemption"] is None
        assert result.gates[policy].flagged is True
    assert result.gates["augmented_unchecked"].reasons[-1] == (
        f"{NOT_EXEMPT}: no correction design check was recorded"
    )
    assert len(result.gates["plug_in"].reasons) == 1


@pytest.mark.parametrize("detail", ["portable", "full"])
def test_gate_notes_round_trip(tmp_path: Path, detail: str) -> None:
    data = _guide(tmp_path)
    result = analyze_dataset(
        fresh_draws_data=data["draws"],
        transport=_transport(data["reused"]),
        **_external(data["path"]),
    )
    restored = EstimationResult.from_dict(
        json.loads(json.dumps(result.to_dict(detail=detail)))
    )

    assert restored.gates == result.gates
    assert restored.gates["candidate"].exemption == "residual_corrected"
    assert restored.gates["candidate"].notes
    for key in ("correction_checks", "level_basis"):
        assert restored.metadata[key] == result.metadata[key]
    for policy, audit in result.metadata["transport_audits"].items():
        saved = restored.metadata["transport_audits"][policy]
        for key in ("gate_exemption", "probe_overlap", "probes_independent", "route"):
            assert saved[key] == audit[key]
    assert restored.best_policy() == result.best_policy()
    assert restored.summary() == result.summary()


def test_cli_marks_exempt_audit_and_failed_design_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from cje.interface.cli import main

    rng = np.random.default_rng(61)
    n = 300
    s_a = rng.uniform(0.2, 0.6, n)
    s_b = rng.uniform(0.2, 0.6, n)
    top = np.zeros(n, dtype=bool)
    top[np.argsort(s_b)[-40:]] = True
    draws = {
        "corrected": _rows(
            s_a,
            _even_rank_mask(s_a, 40),
            lambda s: 0.3 + 0.5 * s + rng.normal(0, 0.02),
            prefix="a",
        ),
        "unbalanced": _rows(
            s_b, top, lambda s: 0.3 + 0.5 * s + rng.normal(0, 0.02), prefix="b"
        ),
    }
    directory = tmp_path / "responses"
    directory.mkdir()
    for policy, rows in draws.items():
        (directory / f"{policy}_responses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    probe_path = tmp_path / "corrected_probe.jsonl"
    probe_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _probes(rng, "corrected", shift=0.25))
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cje",
            "analyze",
            str(directory),
            "--transport-probe",
            f"corrected={probe_path}",
            "--transport-margin",
            "corrected=0.03",
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert (
        "    residual transport: FAIL (calibration map only; estimate "
        "residual-corrected)\n" in out
    )
    assert (
        "    residual correction design check: FAILED (labelled_rows_unbalanced)\n"
        in out
    )
    assert "    residual transport: NOT_CHECKED\n" in out
    assert "Best by point estimate: " in out


def test_cli_marks_probe_reuse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two draws per prompt, probes are draw-0 labelled rows without draw_idx.

    The match is ambiguous, so the run proceeds; the CLI marks the audit as
    not independent with the distinct reused-row count.
    """
    import sys

    from cje.interface.cli import main

    data = _fresh_only(n_prompts=150, draws=2)
    directory = tmp_path / "responses"
    directory.mkdir()
    (directory / "policy_responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in data["rows"])
    )
    probes = [
        {key: row[key] for key in ("prompt_id", "judge_score", "oracle_label")}
        for row, labelled in zip(data["rows"][::2], data["labelled"][::2])
        if labelled
    ]
    probe_path = tmp_path / "policy_probe.jsonl"
    probe_path.write_text("".join(json.dumps(row) + "\n" for row in probes))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cje",
            "analyze",
            str(directory),
            "--transport-probe",
            f"policy={probe_path}",
            "--transport-margin",
            "policy=0.05",
            "--estimator-config",
            '{"inference_method": "cluster_robust"}',
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    status_line = next(
        line for line in out.splitlines() if line.startswith("    residual transport:")
    )
    assert status_line.endswith(
        f"; probes reuse {len(probes)} labelled evaluation rows (not independent)"
    )
    assert (
        f"transport probes reuse {len(probes)} labelled evaluation rows "
        "(audit not independent)" in out
    )
