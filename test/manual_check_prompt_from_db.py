import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test.debug_real_db_synthetic_memory import debug_real_db_with_synthetic_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test prompt generation on a real SQLite ghost DB using synthetic memory consolidation.")
    parser.add_argument("db_path", help="Path to an existing SQLite DB like ./data/main.db")
    parser.add_argument("--query", default="cats, kittens and other felines, animals", help="Prompt-planning retrieval query")
    parser.add_argument("--context-budget", type=int, default=16000, help="Total prompt token budget passed into plan_context_story_simple")
    parser.add_argument("--save-db", default=None, help="Optional path for a synthetic-memory debug DB copy")
    parser.add_argument("--save-prompt", default=None, help="Optional path to write the rendered final prompt text")
    args = parser.parse_args()

    result = debug_real_db_with_synthetic_memory(
        db_path=args.db_path,
        query=args.query,
        context_budget=args.context_budget,
        save_path=args.save_db,
    )

    turns = result["turns"]
    if not turns:
        raise Exception("manual_check_prompt_from_db: rendered turns are empty")

    recent_meta = result["planner_output"].metadata["lanes"]["recent"]
    historical_meta = result["planner_output"].metadata["lanes"]["historical"]
    if recent_meta["token_budget_used"] <= 0:
        raise Exception("manual_check_prompt_from_db: recent lane is empty")

    print("=== SMOKE TEST SUMMARY ===")
    print(f"turn_count={len(turns)}")
    print(f"recent_used={recent_meta['token_budget_used']}")
    print(f"historical_used={historical_meta['token_budget_used']}")
    print(f"planner_used_tokens={result['planner_used_tokens']}")
    print(f"rendered_prompt_tokens={result['rendered_prompt_tokens']}")

    if args.save_prompt:
        prompt_text = "\n\n".join(f"{role.upper()}:\n{content}" for role, content in turns)
        output_path = Path(args.save_prompt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(prompt_text, encoding="utf-8")
        print(f"Saved prompt to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
