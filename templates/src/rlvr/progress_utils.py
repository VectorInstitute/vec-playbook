"""Progress bar utils."""

from contextlib import contextmanager

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


@contextmanager
def spinner(message: str, transient: bool = False):
    """Show spinner and timer."""
    with Progress(
        TextColumn("[bold blue][progress.description]{task.description}"),
        TimeElapsedColumn(),
        SpinnerColumn(),
        transient=transient,
    ) as progress:
        progress.add_task(message, total=None)
        yield
