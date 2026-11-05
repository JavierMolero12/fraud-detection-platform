"""Documentation that points nowhere is worse than none: every relative link must resolve."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\]\(([^)\s]+)\)")
SKIP_DIRS = {".git", ".venv", "data", "mlruns", "node_modules", ".pytest_cache"}


def _markdown_files():
    for path in ROOT.rglob("*.md"):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def test_every_relative_markdown_link_resolves():
    broken = []
    for doc in _markdown_files():
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.DOTALL)
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue  # http(s), mailto, or an anchor in the same page
            path = target.split("#", 1)[0]
            if path and not (doc.parent / path).resolve().exists():
                broken.append(f"{doc.relative_to(ROOT).as_posix()} -> {target}")
    assert not broken, broken
