import pytest

from kumo_label.training.multilabel_predict import (
    multilabel_uncertainty,
    multilabel_confidence,
    multilabel_is_wrong,
    multilabel_imbalance,
)
from kumo_label.training.predict import compute_al_score


# --- multilabel_uncertainty ---

def test_uncertainty_all_half_is_maximally_uncertain():
    assert multilabel_uncertainty([0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_uncertainty_all_extreme_is_zero():
    assert multilabel_uncertainty([0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_uncertainty_mixed_spot_check():
    # (1 - 2*|0.5-0.5|) = 1.0, (1 - 2*|1.0-0.5|) = 0.0 -> mean = 0.5
    assert multilabel_uncertainty([0.5, 1.0]) == pytest.approx(0.5)


# --- multilabel_confidence ---

def test_confidence_mean_of_predicted_set():
    # predicted indices 0 and 2 -> mean(0.9, 0.7) = 0.8
    assert multilabel_confidence([0.9, 0.2, 0.7], [0, 2]) == pytest.approx(0.8)


def test_confidence_empty_predicted_set_uses_one_minus_max():
    assert multilabel_confidence([0.3, 0.4, 0.45], []) == pytest.approx(0.55)


# --- multilabel_is_wrong ---

def test_is_wrong_exact_match_is_false():
    assert multilabel_is_wrong({"cat", "dog"}, {"cat", "dog"}, True) is False


def test_is_wrong_subset_is_true():
    assert multilabel_is_wrong({"cat"}, {"cat", "dog"}, True) is True


def test_is_wrong_superset_is_true():
    assert multilabel_is_wrong({"cat", "dog"}, {"cat"}, True) is True


def test_is_wrong_disjoint_is_true():
    assert multilabel_is_wrong({"cat"}, {"dog"}, True) is True


def test_is_wrong_unlabeled_is_always_false():
    # Even with a mismatched predicted/actual set, unlabeled images are never "wrong".
    assert multilabel_is_wrong({"cat"}, set(), False) is False
    assert multilabel_is_wrong({"cat"}, {"dog"}, False) is False


# --- multilabel_imbalance ---

def test_imbalance_balanced_counts_is_zero_for_any_tag_set():
    counts = {"cat": 10, "dog": 10}
    assert multilabel_imbalance({"cat"}, counts) == pytest.approx(0.0)
    assert multilabel_imbalance({"cat", "dog"}, counts) == pytest.approx(0.0)


def test_imbalance_rare_tag_is_high_score():
    counts = {"cat": 10, "dog": 1}
    assert multilabel_imbalance({"dog"}, counts) == pytest.approx(0.9)


def test_imbalance_uses_rarest_tag_in_set():
    counts = {"cat": 10, "dog": 1, "bird": 5}
    # rarest of {"cat", "dog"} is dog (count=1) -> 1 - 1/10 = 0.9
    assert multilabel_imbalance({"cat", "dog"}, counts) == pytest.approx(0.9)


def test_imbalance_empty_set_is_zero():
    assert multilabel_imbalance(set(), {"cat": 10, "dog": 1}) == pytest.approx(0.0)


def test_imbalance_zero_max_count_is_zero():
    assert multilabel_imbalance({"cat"}, {}) == pytest.approx(0.0)


# --- compute_al_score lockstep guard (imported unchanged from predict.py) ---

def test_compute_al_score_documented_weighted_sum():
    # 0.35*uncertainty + 0.25*is_unlabeled + 0.15*is_wrong + 0.10*(1-confidence) + 0.15*class_imbalance
    score = compute_al_score(
        uncertainty=0.4, is_unlabeled=True, is_wrong=False, confidence=0.6, class_imbalance=0.2,
    )
    expected = 0.35 * 0.4 + 0.25 * 1.0 + 0.15 * 0.0 + 0.10 * (1.0 - 0.6) + 0.15 * 0.2
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.46)
