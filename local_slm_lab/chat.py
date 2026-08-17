"""Send a single prompt to the local model server."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import LocalModelConfig, chat_completion


SYSTEM_PROMPT = (
    "You are a forecasting requirements assistant. Be concise, do not invent "
    "missing values, and ask at most one clarification question at a time."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--config", default="config.example.json")
    args = parser.parse_args()

    config = LocalModelConfig.from_file(Path(args.config))
    answer = chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ],
        config,
    )
    print(answer)


if __name__ == "__main__":
    main()

