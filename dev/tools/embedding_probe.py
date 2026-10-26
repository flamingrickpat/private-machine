"""Probe an embedding endpoint by itself. Development tool, not product code.

Use it before you point a character at a new encoder, or after you restart one:

```powershell
uv run --locked python dev/tools/embedding_probe.py --url http://127.0.0.1:8081/v1 `
    --model qwen3-embedding-0.6b --dimensions 1024
uv run --locked python dev/tools/embedding_probe.py --config <home>/config/model.yaml
```

The tool prints what the server reports about itself, then embeds the five
probe sentences of :data:`pm.memory.embeddings.SMOKE_TEST_TEXTS` and prints the
similarity report. It exits ``1`` when the vectors fail the same quality rules
that the runtime applies at start, so a launcher can gate on it. It never
touches a checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.llm.config import EmbeddingConfig
from pm.memory.embeddings import (SMOKE_MAX_MEAN_VECTOR_NORM, SMOKE_MAX_UNRELATED_MEAN, SMOKE_MIN_PARAPHRASE_MARGIN,
                                  EmbeddingQualityError, EmbeddingService)
from pm.utils import yaml_util


def embedding_settings(args) -> EmbeddingConfig:
    """Build the endpoint settings from ``--config`` or from the command line."""
    if args.config is not None:
        section = yaml_util.safe_load(Path(args.config).read_text(encoding="utf-8"))["embedding"]
        return EmbeddingConfig.model_validate(section)
    missing = [name for name, value in (("--url", args.url), ("--model", args.model)) if value is None]
    if missing:
        raise SystemExit(f"Give --config, or {', '.join(missing)} with --url and --model and --dimensions.")
    return EmbeddingConfig(base_url=args.url, model=args.model, dimensions=args.dimensions,
                           api_key=args.api_key, response_model=args.response_model)


def server_props(config: EmbeddingConfig) -> dict:
    """Return what the server reports about the loaded model. Empty when it has no ``/props``."""
    parts = urlsplit(config.base_url)
    url = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1") + "/props", "", ""))
    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check one embedding endpoint for usable vector quality.")
    parser.add_argument("--config", help="Model config file. Its embedding section supplies the endpoint.")
    parser.add_argument("--url", help="Embedding endpoint, for example http://127.0.0.1:8081/v1")
    parser.add_argument("--model", help="Model name the request sends, and the name the reply must carry.")
    parser.add_argument("--response-model", default=None, help="Alias the server answers with, if it differs.")
    parser.add_argument("--dimensions", type=int, default=1024, help="Vector size the config expects. Default: 1024")
    parser.add_argument("--api-key", default="none", help="Bearer token. Default: none")
    args = parser.parse_args(argv)

    config = embedding_settings(args)
    props = server_props(config)
    served = props.get("model_path") or props.get("model_alias") or config.model
    print(f"endpoint  : {config.base_url}")
    print(f"model     : request '{config.model}'"
          + (f", reply '{config.response_model}'" if config.response_model else "")
          + f", served '{served}'")
    if props:
        print(f"server    : n_ctx={props.get('default_generation_settings', {}).get('n_ctx')}"
              f" alias={props.get('model_alias')} ftype={props.get('model_ftype')}"
              f" build={props.get('build_info')}")
        declared = props.get("model_path")
        if declared:
            print(f"model file: {declared}")
    else:
        print("server    : no /props route")

    service = EmbeddingService(config, live_test=True)
    try:
        report = asyncio.run(service.smoke_test())
    except EmbeddingQualityError as error:
        print(f"\nFAIL\n{error}")
        return 1
    except Exception as error:  # A wrong name, a wrong size, or a dead endpoint is a fail, not a traceback.
        print(f"\nFAIL: {type(error).__name__}: {error}")
        print(f"Check the endpoint, the model name, and --dimensions ({config.dimensions}).")
        return 1
    print("probe     : " + ", ".join(f"{key}={value:.4f}" for key, value in report.items()))
    print(f"limits    : unrelated mean <= {SMOKE_MAX_UNRELATED_MEAN:.2f},"
          f" paraphrase margin >= {SMOKE_MIN_PARAPHRASE_MARGIN:.2f},"
          f" mean vector norm <= {SMOKE_MAX_MEAN_VECTOR_NORM:.2f}")
    print(f"PASS      : {service.dimensions} dimension vectors from"
          f" '{service.last_response_model or config.model}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
