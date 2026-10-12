"""Rule-based waste segregation logic (Practical 2 companion module).

Categorises a waste item into one of three bins and gives a collection note.
This is a rule-based decision, NOT an image-classification result.
"""

from __future__ import annotations

CATEGORIES = {
    "wet": "Wet / Biodegradable waste",
    "dry": "Dry / Recyclable waste",
    "residual": "Residual waste",
}

_RULES = [
    # (item substring, condition predicate, category, reason)
    ("food", lambda c: True, "wet", "Food scraps are biodegradable."),
    ("vegetable", lambda c: True, "wet", "Vegetable peels are biodegradable."),
    ("fruit", lambda c: True, "wet", "Fruit waste is biodegradable."),
    ("paper", lambda c: c.get("soiled"), "residual",
     "Soiled paper cannot be recycled and goes to residual waste."),
    ("paper", lambda c: not c.get("soiled"), "dry",
     "Clean paper is recyclable."),
    ("plastic", lambda c: True, "dry", "Clean plastic is recyclable."),
    ("bottle", lambda c: True, "dry", "Bottles are recyclable."),
    ("can", lambda c: True, "dry", "Metal cans are recyclable."),
    ("glass", lambda c: True, "dry", "Glass is recyclable."),
    ("leaf", lambda c: True, "wet", "Leaves and garden waste are biodegradable."),
    ("meat", lambda c: True, "wet", "Meat and bones are biodegradable."),
    ("diaper", lambda c: True, "residual", "Used hygiene products go to residual waste."),
    ("broken", lambda c: True, "residual", "Broken non-recyclable items go to residual waste."),
]


def classify_item(item: str, condition: dict | None = None) -> dict:
    """Return category, bin label, and reason for a waste item.

    condition: optional dict, e.g. {"soiled": True}.
    """
    condition = condition or {}
    key = item.strip().lower()
    for substring, pred, cat, reason in _RULES:
        if substring in key and pred(condition):
            return {
                "item": item,
                "category_key": cat,
                "category": CATEGORIES[cat],
                "reason": reason,
            }
    return {
        "item": item,
        "category_key": "residual",
        "category": CATEGORIES["residual"],
        "reason": "No specific rule matched; placed in residual waste as a safe default.",
    }


def needs_collection(category_key: str, fill_level: float) -> bool:
    """Return True if the bin should be collected (fill_level 0-100 %)."""
    return fill_level >= 80.0


if __name__ == "__main__":
    for item, cond in [("food scraps", {}), ("paper", {"soiled": True}),
                       ("plastic bottle", {}), ("unknown item", {})]:
        print(classify_item(item, cond))
