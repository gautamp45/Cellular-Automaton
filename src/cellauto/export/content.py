"""Explainer prose: ``content/*.md`` with a minimal ``key: value`` front-matter block."""

from __future__ import annotations

from pathlib import Path

from cellauto.contract.models import ContentPage

_REQUIRED = ("title", "page", "order")


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` fenced ``key: value`` lines (no YAML) from the body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("content file must start with a '---' front-matter fence")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise ValueError("unterminated front-matter block") from None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"bad front-matter line {line!r}")
        meta[key.strip()] = value.strip()
    return meta, "\n".join(lines[end + 1 :]).lstrip("\n")


def load_pages(content_dir: Path) -> list[tuple[ContentPage, bytes]]:
    """``(page, raw bytes)`` for every ``*.md``, sorted by ``(order, slug)``."""
    pages: list[tuple[ContentPage, bytes]] = []
    for path in sorted(content_dir.glob("*.md")):
        raw = path.read_bytes()
        meta, _ = parse_front_matter(raw.decode("utf-8"))
        missing = [k for k in _REQUIRED if k not in meta]
        if missing:
            raise ValueError(f"{path.name}: missing front-matter keys {missing}")
        pages.append(
            (
                ContentPage(
                    slug=path.stem,
                    title=meta["title"],
                    page=meta["page"],
                    order=int(meta["order"]),
                    path=f"content/{path.name}",
                ),
                raw,
            )
        )
    pages.sort(key=lambda p: (p[0].order, p[0].slug))
    return pages
