# AGENTS.md

## Testing

- Ruff Lint: `uv run ruff check [files]`
- Lint: `uv run basedpyright [files]`
  - Pyright Warnings are okay but not nice- you don't have to fix existing ones, but try not to introduce additional ones.

## Code style

- Use type hints where ever appropriate.
- The codebase uses a modern version of Python. Use `list[...]` instead of `typing.List[...]`. No need to import `annotations`.
- Try to minimize indentation.
- Keep checks minimal. Defensive guardrail should not be included in your code.

## Notes for agents

- All python commands should be invoked through `uv run`.
- If some code looks out of place or could be simplified, go ahead and fix the code, even if that is not directly part of the request.
