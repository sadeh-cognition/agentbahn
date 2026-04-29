from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from django.conf import settings


IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
SEARCH_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_MATCHES = 10
MAX_LINE_LENGTH = 240


def run_codebase_agent(query: str, *, base_dir: Path | None = None) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query cannot be blank.")

    root = base_dir or settings.BASE_DIR
    matches = list(_search_codebase(root, normalized_query))
    if not matches:
        return f"No codebase matches found for: {normalized_query}"

    rendered_matches = "\n".join(matches[:MAX_MATCHES])
    if len(matches) > MAX_MATCHES:
        rendered_matches += (
            f"\n... {len(matches) - MAX_MATCHES} additional matches omitted."
        )
    return rendered_matches


def _search_codebase(root: Path, query: str) -> Iterable[str]:
    casefolded_query = query.casefold()
    for path in sorted(root.rglob("*")):
        if not _should_search_file(root, path):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if casefolded_query not in line.casefold():
                continue
            relative_path = path.relative_to(root)
            snippet = line.strip()
            if len(snippet) > MAX_LINE_LENGTH:
                snippet = f"{snippet[:MAX_LINE_LENGTH].rstrip()}..."
            yield f"{relative_path}:{line_number}: {snippet}"


def _should_search_file(root: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    relative_parts = path.relative_to(root).parts
    if any(part in IGNORED_DIRECTORIES for part in relative_parts):
        return False
    return path.suffix in SEARCH_EXTENSIONS
