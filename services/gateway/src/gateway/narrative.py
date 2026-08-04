"""Facts to English.

The intelligence service stays numeric; this layer turns `ExplanationFact`
objects into sentences. Two consequences, both deliberate:

* wording changes without redeploying the ML service
* the narrative is unit-testable against fixed facts

**Templates, not an LLM.** Deterministic, no latency, and no possibility of a
generated sentence stating a number that differs from the one computed. For a
project whose entire pitch is correctness, that trade is not close. An LLM could
later be added *behind* the template with the figures pinned so it can only
rephrase, never restate.

The copy pattern throughout: state the probability, then immediately state what
it means in failure terms. A probability shown alone reads as a promise.
"""

from __future__ import annotations

from intraday_contracts import Direction, ExplanationFact, Prediction, Side

_SIDE_WORD = {
    Side.LONG: "higher",
    Side.SHORT: "lower",
    Side.FLAT: "unchanged",
}

_STRENGTH_BANDS = (
    (0.75, "strongly"),
    (0.65, "moderately"),
    (0.58, "mildly"),
    (0.0, "marginally"),
)


def strength_word(certainty: float) -> str:
    for threshold, word in _STRENGTH_BANDS:
        if certainty >= threshold:
            return word
    return "marginally"


def headline(prediction: Prediction) -> str:
    """One sentence. Always paired with `failure_framing`."""
    if prediction.side is Side.FLAT:
        return (
            f"{prediction.symbol}: no call. The model is not confident enough "
            f"in either direction to justify a position."
        )

    word = strength_word(prediction.certainty)
    direction = _SIDE_WORD[prediction.side]
    return (
        f"{prediction.symbol} looks {word} likely to trade {direction} "
        f"over the next {prediction.horizon_bars} bars."
    )


def failure_framing(prediction: Prediction) -> str:
    """The sentence that stops a probability reading as a promise."""
    if prediction.side is Side.FLAT:
        return "No position is suggested, which is itself a decision."

    percent = round(prediction.certainty * 100)
    wrong = 100 - percent
    return (
        f"The model puts this at {percent}% — but {percent}% is not a promise. "
        f"Out of every 100 similar calls, about {wrong} have gone the other way."
    )


def calibration_caveat(prediction: Prediction) -> str | None:
    """Raw model scores are not probabilities and must not be shown as percentages."""
    if prediction.is_calibrated:
        return None
    return (
        "This score has not been calibrated, so it should be read as a ranking "
        "rather than as a probability."
    )


def meta_note(prediction: Prediction) -> str | None:
    """What the second model thought of the first one's call."""
    if prediction.meta_probability is None:
        return None
    percent = round(prediction.meta_probability * 100)
    if prediction.side is Side.FLAT:
        return (
            f"A second model, which judges when the first one is unreliable, "
            f"put the odds of this call being correct at {percent}% and declined it."
        )
    return (
        f"A second model, which judges when the first one is unreliable, "
        f"put the odds of this specific call being correct at {percent}%."
    )


def fact_sentence(fact: ExplanationFact) -> str:
    verb = "points upward" if fact.direction is Direction.SUPPORTS_UP else "points downward"
    return f"{fact.display_name} is {fact.display_value}, which {verb}."


def explain(prediction: Prediction, *, max_facts: int = 3) -> dict[str, object]:
    """The full narrative payload for one prediction.

    Top three facts only. A list of forty SHAP values is not an explanation.
    """
    facts = sorted(prediction.facts, key=lambda f: f.rank)[:max_facts]

    return {
        "headline": headline(prediction),
        "failure_framing": failure_framing(prediction),
        "meta_note": meta_note(prediction),
        "calibration_caveat": calibration_caveat(prediction),
        "reasons": [fact_sentence(f) for f in facts],
    }


def significance_banner(p_value: float, n_permutations: int) -> dict[str, object]:
    """The status-bar verdict, shown on every screen.

    A system that reports predictions without reporting whether they are
    distinguishable from chance is hiding the most important number about
    itself. Putting it beside the ticker means it cannot be read past.
    """
    significant = p_value < 0.05
    return {
        "p_value": p_value,
        "n_permutations": n_permutations,
        "is_significant": significant,
        "label": f"p={p_value:.4f}",
        "verdict": "significant" if significant else "NOT significant",
        "detail": (
            "The measured edge is statistically distinguishable from chance."
            if significant
            else "The measured edge is NOT distinguishable from chance."
        ),
    }
