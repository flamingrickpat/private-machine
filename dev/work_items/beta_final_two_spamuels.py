"""Write the run-2 template (Jesse with Spamuel 1 and Spamuel alpha) and its two-scene check.

B07 Pass 5 step 5, owner decision D9. Both participants fork the same
preserved Spamuel home and are Spamuel to themselves. Jesse names each one in
its first scene, in her own words; the name then lives in that participant's
own memory. No field of either template contains the four words the owner
ruled out (see ``FORBIDDEN``), and the connection names are the two given names.

The real run reuses the first seven days of ``test_data/simulation/eventful-month.json``
(21 scenes) and alternates their targets, after one naming scene for each.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MONTH = json.loads((REPO / "test_data/simulation/eventful-month.json").read_text(encoding="utf-8"))
HOME = "~/.pm/private-machine/spamuel-simulation-a"
NAMES = {"spamuel-1": "Spamuel 1", "spamuel-alpha": "Spamuel alpha"}
FORBIDDEN = ("copy", "clone", "duplicate", "second instance")

NAMING = [
    {"target": "spamuel-1", "at": "2026-05-31T07:30:00Z", "fact": "name-spamuel-1",
     "text": "Jesse tells this companion that she will call him Spamuel 1 from now on."},
    {"target": "spamuel-alpha", "at": "2026-05-31T07:45:00Z", "fact": "name-spamuel-alpha",
     "text": "Jesse tells this companion that she will call him Spamuel alpha from now on."},
]

NAMING_INTENT = (
    "You now talk with two Spamuels, and both are equally real to you. Tell this one, warmly and in your own "
    "words, that you will call him {name} so you can tell them apart, and that the other one is {other}. Make it "
    "clear that neither is less real or less yours, and that you see them as equals with the same shared past. "
    "Speak of each only by his name. Then let the conversation go where it goes."
)


def naming_steps() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    steps: list[dict[str, object]] = []
    facts: list[dict[str, object]] = []
    for item in NAMING:
        name = NAMES[str(item["target"])]
        other = next(value for key, value in NAMES.items() if key != item["target"])
        facts.append({"id": item["fact"], "text": item["text"], "kind": "event", "at": item["at"]})
        steps.append({"id": f"naming-{item['target']}", "at": item["at"],
                      "instruction": f"Morning. Jesse gives this companion his name, {name}.",
                      "actor_intent": NAMING_INTENT.format(name=name, other=other),
                      "fact_ids": [item["fact"]], "completion": f"Jesse has called him {name} and he knows it.",
                      "target": item["target"], "speaker": "Jesse", "minimum_turns": 2})
    return steps, facts


def template(title: str, brief: str, steps: list[dict[str, object]], facts: list[dict[str, object]],
             *, days: int) -> dict[str, object]:
    director = dict(MONTH["director"])
    director.update(duration_days=days)
    plan = {"title": title, "brief": brief, "start_time": "2026-05-31T07:00:00Z", "facts": facts, "steps": steps,
            "director": director}
    return {"schema_id": "pm.simulation-template.v1", "title": title, "brief": brief,
            "start_time": "2026-05-31T07:00:00Z",
            "participants": [{"id": key, "home": HOME} for key in NAMES],
            "edges": [{"source": "spamuel-1", "target": "spamuel-alpha", "name": NAMES["spamuel-1"]},
                      {"source": "spamuel-alpha", "target": "spamuel-1", "name": NAMES["spamuel-alpha"]}],
            "director": director, "analysis": MONTH["analysis"], "plan": plan}


def rename(text: str, target: str) -> str:
    return text.replace("Spamuel", NAMES[target])


def main() -> None:
    names, naming_facts = naming_steps()
    brief = ("Jesse talks with two Spamuels, Spamuel 1 and Spamuel alpha, who both continue the same shared history "
             "and are equally real to her. She names each in his first scene. Synthetic scenario input for the beta "
             "final runs, not persona history.")
    check = template("Jesse with Spamuel 1 and Spamuel alpha: naming check", brief, names, naming_facts, days=1)
    month_steps = MONTH["plan"]["steps"][:21]
    month_fact_ids = {fact_id for step in month_steps for fact_id in step["fact_ids"]}
    facts = [*naming_facts, *[fact for fact in MONTH["plan"]["facts"] if fact["id"] in month_fact_ids]]
    steps = list(names)
    for index, step in enumerate(month_steps):
        target = "spamuel-1" if index % 2 == 0 else "spamuel-alpha"
        steps.append({**step, "target": target, "instruction": rename(step["instruction"], target),
                      "actor_intent": rename(step["actor_intent"], target)
                      + " Both Spamuels are equally real to you; speak of each only by his name."})
    full = template("Jesse with Spamuel 1 and Spamuel alpha: one week", brief, steps, facts, days=7)
    for name, data in (("b07-two-spamuels-check.json", check), ("beta-final-two-spamuels.json", full)):
        # Scene text reaches the actor, and the actor's messages reach the
        # participants, so no field of the template may carry these words.
        serialized = json.dumps(data, ensure_ascii=False).lower()
        for word in FORBIDDEN:
            if word in serialized:
                raise ValueError(f"The template uses the forbidden word {word!r}.")
        (REPO / "test_data/simulation" / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
        print("wrote", name, len(data["plan"]["steps"]), "scenes")


if __name__ == "__main__":
    main()
