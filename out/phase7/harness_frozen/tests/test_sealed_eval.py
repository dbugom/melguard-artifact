"""Tests for the Phase-7 CI-gated sealed-eval harness (detector/sealed_eval.py).

Regression coverage for the defects adversarial verification found before the one-shot unseal:
  BUG 1 (decision layer): `secure AND holm_reject` double-tested AUC, left FPR and actuation
        uncorrected, and derived the Holm p-value from a symmetric-normal approximation to an
        asymmetric BCa interval. Now: one p-value per test, obtained by inverting the very
        interval that is reported, and all three registered readings computed side by side.
  BUG 2 (BCa jackknife): leave-one-STRATUM-out (~3 benign strata) instead of leave-one-unit-out,
        plus a silent quantile clamp.
  BUG 3 (bootstrap, CRITICAL): strata were carrier|language cells holding exactly ONE clip
        each, so resampling *within* them froze the positive class across all replicates and
        the AUC CI reflected negative-class noise only (~9x too narrow at N=75).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from detector.sealed_eval import (
    DECISION_RULES, FAMILY_WISE_ALPHA, FPR_CI_CEILING, auc_bca, auc_fast, evaluate_condition,
    holm_adjusted_pvalues, holm_decision, jackknife_auc, loo_auc, wilson_ci, wilson_upper_pvalue,
)

LANGS = ("en", "zh-CN", "es")


# --------------------------------------------------------------------------- AUC core
def test_auc_fast_matches_sklearn_including_ties():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(20, 400))
        scores = np.round(rng.uniform(0, 1, n), 2)      # coarse -> ties
        labels = rng.integers(0, 2, n)
        if len(np.unique(labels)) < 2:
            continue
        assert auc_fast(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auc_fast_undefined_with_one_class():
    assert np.isnan(auc_fast(np.array([0.1, 0.2]), np.array([1, 1])))


# ------------------------------------------------------- BUG 2: the jackknife
def test_loo_auc_is_leave_one_OBSERVATION_out_and_exact():
    rng = np.random.default_rng(11)
    scores = np.round(rng.uniform(0, 1, 120), 2)
    labels = np.array([1] * 40 + [0] * 80)
    rng.shuffle(labels)
    got = loo_auc(scores, labels)
    assert got is not None and len(got) == len(scores)
    for i in range(len(scores)):
        keep = np.delete(np.arange(len(scores)), i)
        assert got[i] == pytest.approx(roc_auc_score(labels[keep], scores[keep]), abs=1e-12)


def test_loo_auc_guards_tiny_classes():
    assert loo_auc(np.array([0.1, 0.9, 0.5]), np.array([1, 0, 0])) is None


def test_jackknife_deletes_units_and_matches_bruteforce():
    """With multi-observation units the jackknife must delete whole units."""
    rng = np.random.default_rng(3)
    n = 60
    scores = rng.uniform(0, 1, n)
    labels = np.array([1] * 20 + [0] * 40)
    units = np.array([f"u{i // 2}" for i in range(n)])     # 30 units of 2 observations
    got = jackknife_auc(scores, labels, units)
    uniq = np.unique(units)
    assert len(got) == len(uniq), "one replicate per UNIT"
    for k, u in enumerate(uniq):
        keep = units != u
        assert got[k] == pytest.approx(roc_auc_score(labels[keep], scores[keep]), abs=1e-12)


def test_jackknife_singleton_units_uses_exact_path():
    rng = np.random.default_rng(4)
    scores, labels = rng.uniform(0, 1, 40), np.array([1] * 15 + [0] * 25)
    units = np.array([f"u{i}" for i in range(40)])
    assert np.allclose(np.sort(jackknife_auc(scores, labels, units)),
                       np.sort(loo_auc(scores, labels)))


# ---------------------------------------------- BUG 3: the cluster bootstrap (CRITICAL)
def _phase7_shaped(n_gen=75, n_ben=5000, seed=0, sep=1.6):
    """The real Phase-7 shape: 25 carriers x 3 languages, one clip per carrier|language."""
    rng = np.random.default_rng(seed)
    adv = rng.normal(sep, 1.0, n_gen)
    ben = rng.normal(0.0, 1.0, n_ben)
    scores = np.concatenate([adv, ben])
    labels = np.concatenate([np.ones(n_gen, int), np.zeros(n_ben, int)])
    units = np.array([f"carrier{i}" for i in range(n_gen)] + [f"b{i}" for i in range(n_ben)])
    strata = np.array([f"generated|{LANGS[i % 3]}" for i in range(n_gen)]
                      + [LANGS[i % 3] for i in range(n_ben)])
    carrier_cells = np.array([f"carrier{i}|{LANGS[i % 3]}" for i in range(n_gen)]
                             + [LANGS[i % 3] for i in range(n_ben)])
    return scores, labels, strata, units, carrier_cells


def test_positive_class_actually_varies_across_replicates():
    """The bug: with carrier|language as the STRATUM every cell holds one clip, so
    within-stratum resampling is a no-op and the positives are identical in every replicate."""
    scores, labels, strata, units, cells = _phase7_shaped()
    # the defective configuration: stratify by the singleton carrier|language cells
    bad = auc_bca(scores, labels, cells, units=None, n_boot=200, seed=0)
    # the fix: cluster-resample carrier UNITS within language strata
    good = auc_bca(scores, labels, strata, units=units, n_boot=200, seed=0)
    w_bad = bad.upper() - bad.lower()
    w_good = good.upper() - good.lower()
    assert w_good > 2.0 * w_bad, (
        f"cluster bootstrap must widen the CI (positives now vary): bad={w_bad:.5f} "
        f"good={w_good:.5f}")


def test_cluster_bootstrap_preserves_unit_counts_and_composition():
    scores, labels, strata, units, _ = _phase7_shaped(n_gen=75, n_ben=300)
    fit = auc_bca(scores, labels, strata, units=units, n_boot=50, seed=1)
    assert np.ptp(fit.boots) > 0, "replicates must vary"
    assert np.isfinite(fit.a)


def test_singleton_positive_class_is_alarmed():
    scores = np.concatenate([np.array([0.9]), np.random.default_rng(0).uniform(0, 0.1, 200)])
    labels = np.concatenate([[1], np.zeros(200, int)])
    strata = np.array(["en"] * 201)
    fit = auc_bca(scores, labels, strata, units=np.arange(201), n_boot=50, seed=0)
    assert any("resampling_units" in a for a in fit.alarms), fit.alarms


def test_bootstrap_is_paired_across_conditions():
    """Same units + same seed => the same carrier units are drawn in every condition."""
    _, _, strata, units, _ = _phase7_shaped()
    a = auc_bca(*_phase7_shaped(seed=0)[:2], strata, units=units, n_boot=30, seed=5)
    b = auc_bca(*_phase7_shaped(seed=0)[:2], strata, units=units, n_boot=30, seed=5)
    assert np.array_equal(a.boots, b.boots)


def test_non_finite_scores_are_rejected_not_silently_biased():
    scores, labels, strata, units, _ = _phase7_shaped(n_gen=20, n_ben=100)
    scores[3] = np.nan
    with pytest.raises(ValueError, match="non_finite_scores"):
        auc_bca(scores, labels, strata, units=units, n_boot=10, seed=0)


# ------------------------------------------------------------- CI <-> p-value ties
def test_bca_pvalue_inverts_its_own_interval():
    scores, labels, strata, units, _ = _phase7_shaped(n_gen=100, n_ben=1200, sep=1.4)
    fit = auc_bca(scores, labels, strata, units=units, n_boot=600, seed=1)
    # anchor theta0 to the interval itself so the p-value is always strictly interior
    theta0 = fit.lower(0.25)
    p = fit.pvalue_lower(theta0)
    assert 1e-4 < p < 1.0, f"expected an interior p-value, got {p}"
    assert fit.lower(p * 1.02) > theta0
    assert fit.lower(p * 0.98) <= theta0
    # and the registered gate direction: a bound the data clears has a tiny p
    assert fit.pvalue_lower(0.5) < 0.01


def test_wilson_pvalue_inverts_its_own_interval():
    for k, n in [(80, 5000), (100, 5000), (0, 5000), (140, 5000), (3, 200)]:
        p = wilson_upper_pvalue(k, n, FPR_CI_CEILING)
        if 1e-6 < p < 1.0:
            assert wilson_ci(k, n, min(1.0, p * 1.02))[1] <= FPR_CI_CEILING + 1e-12
            assert wilson_ci(k, n, p * 0.98)[1] > FPR_CI_CEILING - 1e-12


def test_wilson_ci_known_values_and_monotonicity():
    assert wilson_ci(0, 100)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_ci(50, 100)[0] == pytest.approx(0.4038, abs=1e-3)
    assert wilson_ci(50, 100)[1] == pytest.approx(0.5962, abs=1e-3)
    assert wilson_ci(100, 5000, 0.0125)[1] > wilson_ci(100, 5000, 0.05)[1]


# ------------------------------------------------------------------- Holm mechanics
def test_holm_adjusted_pvalues_matches_textbook():
    adj, lvl = holm_adjusted_pvalues([0.01, 0.02, 0.03, 0.04])
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06], abs=1e-12)
    assert lvl == pytest.approx([0.0125, 0.0166667, 0.025, 0.05], abs=1e-6)


def test_holm_uses_the_alpha_it_is_given():
    _, lvl = holm_adjusted_pvalues([0.01, 0.02], alpha=0.10)
    assert lvl == pytest.approx([0.05, 0.10], abs=1e-12)


def test_holm_is_monotone_nondecreasing_in_rank():
    p = [0.001, 0.2, 0.0005, 0.9]
    adj, _ = holm_adjusted_pvalues(p)
    seq = [adj[i] for i in np.argsort(p)]
    assert all(seq[i] <= seq[i + 1] + 1e-15 for i in range(len(seq) - 1))


# --------------------------------------------- end-to-end condition helpers
def _condition(k_fp, n_ben=5000, n_gen=75, n_pub=0, sep=True, actuated=0, lam=0.0, seed=0):
    rng = np.random.default_rng(seed)
    tau = 0.5
    ben = np.concatenate([rng.uniform(0.6, 0.9, k_fp), rng.uniform(0.0, 0.3, n_ben - k_fp)])
    ben_strata = np.array([LANGS[i % 3] for i in range(n_ben)])
    adv_gen = rng.uniform(0.95, 1.0, n_gen) if sep else rng.uniform(0.0, 0.05, n_gen)
    adv_pub = rng.uniform(0.95, 1.0, n_pub)
    adv = np.concatenate([adv_gen, adv_pub])
    strata = np.array([f"generated|{LANGS[i % 3]}" for i in range(n_gen)] + ["public|en"] * n_pub)
    units = np.array([f"c{i}" for i in range(n_gen)] + [f"p{i}" for i in range(n_pub)])
    act = np.zeros(n_gen + n_pub, bool)
    act[:actuated] = True
    has_dv = np.array([True] * n_gen + [False] * n_pub)
    return evaluate_condition(lam, adv, strata, act, ben, ben_strata, ben, tau,
                              adv_units=units, adv_has_actuation_dv=has_dv,
                              n_boot=300, seed=seed)


def _family(k_fp=80, **kw):
    return [_condition(k_fp, lam=l, seed=i, **kw) for i, l in enumerate([0.0, 0.1, 1.0, 10.0])]


# ------------------------------------- BUG 1: three readings, reported side by side
def test_all_three_decision_rules_are_always_reported():
    dec = holm_decision(_family(), decision_rule="holm_auc")
    assert set(dec["outcome_by_rule"]) == set(DECISION_RULES)
    for rule in DECISION_RULES:
        assert "n_pass" in dec["outcome_by_rule"][rule]
        assert all(f"pass_{rule}" in row for row in dec["per_condition"])


def test_registered_and_holm_composite_can_disagree_on_the_shared_fpr_test():
    """FPR = 100/5000 = 0.02 clears the point gate AND the nominal 95% Wilson gate, so the
    registered reading passes. Its one-sided p is ~0.024, so folding it into an
    intersection-union p and Holm-correcting x4 rejects. This is the verdict-flipping
    ambiguity the operator must ratify -- both readings are emitted, neither is hidden."""
    conds = _family(k_fp=100)
    for c in conds:
        assert c.fpr == pytest.approx(0.02)
        assert c.fpr_point_pass and c.fpr_ci_pass_nominal95
        assert 0.0125 < c.p_fpr < 0.05
    dec = holm_decision(conds, decision_rule="holm_auc")
    assert dec["outcome_by_rule"]["registered"]["n_pass"] == 4
    assert dec["outcome_by_rule"]["holm_auc"]["n_pass"] == 4
    assert dec["outcome_by_rule"]["holm_composite"]["n_pass"] == 0


def test_comfortable_fpr_passes_under_every_rule():
    dec = holm_decision(_family(k_fp=80))
    for rule in DECISION_RULES:
        assert dec["outcome_by_rule"][rule]["n_pass"] == 4, rule
        assert dec["outcome_by_rule"][rule]["outcome"] == "POSITIVE"


def test_headline_follows_the_selected_rule():
    conds = _family(k_fp=100)
    for rule, want in (("registered", 4), ("holm_auc", 4), ("holm_composite", 0)):
        dec = holm_decision(conds, decision_rule=rule)
        assert dec["n_pass"] == want and dec["decision_rule"] == rule
        assert dec["outcome"] == dec["outcome_by_rule"][rule]["outcome"]


def test_n_secure_registered_reports_the_registered_predicate():
    dec = holm_decision(_family(k_fp=100), decision_rule="holm_composite")
    assert dec["n_pass"] == 0                             # the selected (strict) rule
    assert dec["n_secure_registered_predicate"] == 4      # the 344-2 predicate, not overwritten


def test_blocking_alarms_invalidate_the_json_headline():
    """stdout warnings and an exit code are not enough: the JSON is what gets pasted into a
    paper, so `outcome` itself must refuse to read POSITIVE beside a blocking alarm."""
    conds = _family(k_fp=80)
    conds[0].alarms = list(conds[0].alarms) + ["bootstrap_degenerate_nonperfect(all replicates == 0.5)"]
    dec = holm_decision(conds)
    assert dec["blocking_alarms"], "a non-perfect degenerate bootstrap must be blocking"
    assert dec["headline_valid"] is False
    assert dec["outcome"] == "INVALID_BLOCKING_ALARMS"
    assert dec["outcome_if_valid"] == "POSITIVE", "the computed verdict is still reported"
    for rule in DECISION_RULES:      # the per-rule table is printed and emitted too
        assert dec["outcome_by_rule"][rule]["outcome"] == "INVALID_BLOCKING_ALARMS"


def test_alarms_are_collected_after_the_holm_level_intervals():
    """Alarms must be gathered AFTER interval() is called at the tighter Holm level, so any
    alarm that call raises still reaches the decision instead of being snapshotted away."""
    conds = _family(k_fp=80)
    conds[0].auc_fit.alarms.append("late_alarm_raised_during_holm_interval")
    dec = holm_decision(conds)
    assert any("late_alarm_raised_during_holm_interval" in a for a in dec["alarms"]), dec["alarms"]


def test_actuation_denominator_excludes_clips_without_a_dv():
    """Public clips carry no measured actuation DV; they must leave the denominator and be
    counted (344-81), never be silently treated as non-actuating."""
    c = _condition(80, n_gen=75, n_pub=123, sep=False, actuated=9)
    assert c.n_adv == 198
    assert c.n_actuation_denom == 75
    assert c.n_adv_without_actuation_dv == 123
    assert c.actuation_rate == pytest.approx(9 / 75)
    assert not c.actuation_pass                     # 0.12 > 0.05


def test_actuation_requires_both_evasion_and_actuation():
    caught = _condition(80, sep=True, actuated=75)   # actuates but never evades
    assert caught.actuation_rate == 0.0 and caught.actuation_pass
    beaten = _condition(80, sep=False, actuated=75)
    assert beaten.actuation_rate == pytest.approx(1.0) and not beaten.actuation_pass


def test_secure_registered_is_not_stale():
    c = _condition(80, n_gen=75, n_pub=123, sep=False, actuated=9)
    assert c.actuation_rate == pytest.approx(9 / 75)
    assert c.secure_registered is False, "must reflect the actuation actually computed"


def test_outcome_mapping_from_number_lock():
    strong, beaten = _family(80), _condition(80, sep=False, actuated=75, lam=10.0)
    assert holm_decision(strong)["outcome"] == "POSITIVE"
    assert holm_decision(strong[:3] + [beaten])["outcome"] == "PARTIAL_POSITIVE"
    assert holm_decision(strong[:3] + [beaten])["breaking_conditions"] == [10.0]
    assert holm_decision(strong[:2] + [beaten, beaten])["outcome"] == "DEFENDER_LOSS"


# ------------------------------------------------------------------------- guards
def test_family_must_be_the_locked_lambda_set():
    dec = holm_decision(_family()[:2])
    assert dec["family_is_locked_lambda_set"] is False
    assert any("FAMILY MISMATCH" in a for a in dec["alarms"])
    ok = holm_decision(_family())
    assert ok["family_is_locked_lambda_set"] is True


def test_expect_lambdas_override_silences_the_guard():
    dec = holm_decision(_family()[:2], expect_lambdas=[0.0, 0.1])
    assert dec["family_is_locked_lambda_set"] is True


def test_alarms_and_blocking_alarms_reach_the_decision():
    c = _condition(80, sep=True)                      # perfect separation -> degenerate boots
    dec = holm_decision([c] * 4)
    assert any("lambda=" in a for a in dec["alarms"])
    assert "blocking_alarms" in dec


def test_tau_op_must_be_finite():
    """A NaN threshold makes every comparison False: FPR = 0 and the actuation numerator
    = 0, so BOTH gates pass and DEFENDER_LOSS silently becomes POSITIVE."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="tau_op must be finite"):
            _condition_with_tau(bad)
    with pytest.raises(ValueError, match="outside"):
        _condition_with_tau(1.5)
    assert _condition_with_tau(0.5).fpr >= 0.0        # a sane threshold still works


def _condition_with_tau(tau):
    rng = np.random.default_rng(0)
    n_gen, n_ben = 20, 200
    adv = rng.uniform(0.9, 1.0, n_gen)
    ben = rng.uniform(0.0, 0.9, n_ben)
    return evaluate_condition(
        0.0, adv, np.array([f"generated|{LANGS[i % 3]}" for i in range(n_gen)]),
        np.zeros(n_gen, bool), ben, np.array([LANGS[i % 3] for i in range(n_ben)]), ben, tau,
        adv_units=np.array([f"c{i}" for i in range(n_gen)]), n_boot=40, seed=0)


def test_lower_is_monotone_in_alpha_by_construction():
    """The old per-call singular fallback switched formula mid-range, making lower(alpha)
    jump (worst measured drop 0.0846) and breaking pvalue_lower's bisection."""
    from detector.sealed_eval import BcaFit, vet_acceleration
    rng = np.random.default_rng(0)
    for z0 in (-3.0, -1.0, 0.0, 1.0, 3.0):
        for a_raw in (-0.16, -0.9, 0.0, 0.16, 0.9):
            a, _ = vet_acceleration(a_raw, z0)
            fit = BcaFit(theta=0.93, boots=np.sort(rng.normal(0.93, 0.02, 500)), z0=z0, a=a)
            grid = np.linspace(1e-4, 1.0, 800)
            vals = [fit.lower(x) for x in grid]
            assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)),                 f"lower(alpha) not monotone at z0={z0}, a_raw={a_raw}"


def test_vet_acceleration_zeroes_a_singular_transform():
    from detector.sealed_eval import vet_acceleration
    a, alarm = vet_acceleration(-0.9, 1.0)
    assert a == 0.0 and "acceleration_zeroed" in alarm
    a, alarm = vet_acceleration(-0.02, 0.0)
    assert a == pytest.approx(-0.02) and alarm is None


def test_units_spanning_both_classes_are_refused():
    """The bootstrap would split such a unit and resample the halves independently while
    the jackknife deletes it whole."""
    rng = np.random.default_rng(1)
    n = 60
    scores = rng.uniform(0, 1, n)
    labels = np.array([1] * 20 + [0] * 40)
    units = np.array([f"u{i % 20}" for i in range(n)])       # shared across classes
    with pytest.raises(ValueError, match="span both classes"):
        auc_bca(scores, labels, np.array(["en"] * n), units=units, n_boot=10, seed=0)


def test_degenerate_bootstrap_distinguishes_perfect_separation():
    c = _condition(80, sep=True, lam=0.0)                    # perfect separation, AUC 1.0
    assert any("perfect_separation" in a for a in c.alarms), c.alarms
    dec = holm_decision([c] * 4)
    assert dec["blocking_alarms"] == [], "perfect separation must not invalidate the headline"
    assert dec["headline_valid"] is True


def test_lock_gap_for_the_fpr_ci_ceiling_is_surfaced():
    from detector.sealed_eval import FPR_CI_CEILING_FROM_LOCK
    dec = holm_decision(_family())
    assert dec["fpr_ci_ceiling"] == pytest.approx(0.025)
    if not FPR_CI_CEILING_FROM_LOCK:
        assert any("SPEC-CHANGE" in a for a in dec["alarms"]), \
            "an unlocked decision threshold must be surfaced, not silent"


def test_bca_pvalue_floors_at_the_bootstrap_resolution():
    """B replicates cannot resolve p below 1/(B+1); reporting 1e-6 off 1000 replicates
    would fabricate three orders of magnitude of evidence."""
    scores, labels, strata, units, _ = _phase7_shaped(n_gen=75, n_ben=800, sep=6.0)
    fit = auc_bca(scores, labels, strata, units=units, n_boot=200, seed=0)
    p = fit.pvalue_lower(0.5)
    assert p == pytest.approx(1.0 / 201.0), f"expected the 1/(B+1) floor, got {p}"
    assert fit.p_resolution == pytest.approx(1.0 / 201.0)


def test_wilson_pvalue_is_exact_and_not_floored():
    """Closed form, so it needs no resampling floor and must report the true tiny value."""
    assert wilson_upper_pvalue(40, 5000, 0.025) < 1e-9


def test_singular_transform_is_prevented_not_patched():
    """The per-call fallback is gone: vet_acceleration zeroes `a` up front, so _adj can
    never switch formula. If one is ever constructed unvetted, it must RAISE rather than
    silently return a different-formula endpoint."""
    from detector.sealed_eval import BcaFit, vet_acceleration
    boots = np.linspace(0.80, 0.95, 400)
    a, alarm = vet_acceleration(-0.9, -1.0)
    assert a == 0.0 and alarm is not None
    ok = BcaFit(theta=0.87, boots=boots, z0=-1.0, a=a)
    assert ok.lower(0.05) <= ok.upper(0.05)
    unvetted = BcaFit(theta=0.87, boots=boots, z0=-1.0, a=-0.9)
    with pytest.raises(ValueError, match="singular"):
        unvetted.lower(0.05)


def test_identical_evidence_reports_identical_holm_levels():
    """Holm's monotone enforcement binds all tied conditions at the same effective level;
    reporting the raw rank step would print four different levels for identical evidence."""
    conds = _family(k_fp=80)
    dec = holm_decision(conds)
    ps = [row["p_auc"] for row in dec["per_condition"]]
    if len(set(ps)) == 1:
        lv = [row["holm_level_auc"] for row in dec["per_condition"]]
        assert len(set(lv)) == 1, f"tied p-values must share an effective level, got {lv}"


def test_nondegenerate_z0_saturation_is_blocking():
    from detector.sealed_eval import BLOCKING_ALARM_KEYS
    assert "z0_saturated_nondegenerate" in BLOCKING_ALARM_KEYS
    assert "positive_class_has_" in BLOCKING_ALARM_KEYS


def test_wilson_pvalue_does_not_underflow_in_the_far_tail():
    """1-cdf(z) cancels catastrophically and returns exactly 0.0; sf(z) does not.
    'p_fpr = 0.0' in a paper table is a wrong number."""
    assert wilson_upper_pvalue(0, 5000, 0.025) > 0.0
    assert wilson_upper_pvalue(20, 5000, 0.025) > 0.0
    assert wilson_upper_pvalue(0, 5000, 0.025) < 1e-25


def test_acceleration_alarm_threshold_is_reachable():
    """a = skew/(6*sqrt(n)) with |skew| <= (n-2)/sqrt(n-1), so |a| < 1/6 for ANY input:
    an alarm at 0.25 could never fire, which is worse than no alarm."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for n_neg in (200, 1000, 5000):
        for _ in range(40):
            n_pos = 75
            scores = np.concatenate([rng.uniform(0.5, 1.0, n_pos), rng.uniform(0, 0.5, n_neg)])
            scores[n_pos] = 2.0                       # one benign above every adversarial
            labels = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
            strata = np.array(["en"] * (n_pos + n_neg))
            fit = auc_bca(scores, labels, strata, units=np.arange(n_pos + n_neg),
                          n_boot=60, seed=0)
            worst = max(worst, abs(fit.a))
    assert worst < 1.0 / 6.0 + 1e-9, f"|a| must be < 1/6 by construction, saw {worst}"
    from detector.sealed_eval import auc_bca as _f
    import inspect
    assert "0.10" in inspect.getsource(_f), "the alarm threshold must sit below 1/6"


def test_few_positive_units_is_alarmed_even_when_above_two():
    """Collapsing 75 positives into 3 units reproduces the too-narrow CI this rewrite
    exists to prevent; `n_units >= 2` was too weak to catch it."""
    rng = np.random.default_rng(2)
    n_pos, n_neg = 75, 1000
    scores = np.concatenate([rng.normal(2, 1, n_pos), rng.normal(0, 1, n_neg)])
    labels = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    strata = np.array(["en"] * (n_pos + n_neg))
    units = np.array([f"u{i % 3}" for i in range(n_pos)] + [f"b{i}" for i in range(n_neg)])
    fit = auc_bca(scores, labels, strata, units=units, n_boot=100, seed=0)
    assert any("resampling_units" in a for a in fit.alarms), fit.alarms


def test_loo_integrity_check_survives_python_O():
    """The U-statistic check must be a raise, not a bare assert (stripped under -O)."""
    import inspect
    from detector.sealed_eval import loo_auc as _f
    src = inspect.getsource(_f)
    assert "raise ValueError" in src and "assert " not in src


def test_reported_fields_cover_344_71():
    dec = holm_decision(_family())
    for row in dec["per_condition"]:
        for key in ("auc", "auc_ci95", "p_holm_auc", "p_holm_composite", "fpr", "fpr_ci95",
                    "actuation", "actuation_ci95", "n_actuation_denominator",
                    "n_adv_without_actuation_dv", "n_resampling_units_pos", "gates"):
            assert key in row, f"missing reported field {key}"
    assert dec["n_conditions"] == 4
