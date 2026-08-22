import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_trace.run_intent import load_intents, select_experiment_intents


def test_select_experiment_intents_balances_categories():
    intents = load_intents()
    subset = select_experiment_intents(intents, max_per_category=2)

    assert len(subset) <= 8
    categories = [item.get("intent_category") for item in subset]
    assert categories.count("standard") <= 2
    assert categories.count("ambiguous") <= 2
    assert categories.count("boundary") <= 2
    assert categories.count("trap") <= 2
    assert "standard" in categories
    assert "ambiguous" in categories
