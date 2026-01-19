"""Optional LangFuse intergations."""

import logging
from os import getenv
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Sequence, TypeVar

import backoff
import httpx

from .progress_utils import spinner


if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse._client.datasets import DatasetItemClient

    from .grpo.data_types import RLVRDataItem
    from .grpo.rollout_generation import EvalResult

V = TypeVar("V")
TraceID = str

logger = logging.getLogger(__name__)


def get_langfuse_client() -> "Langfuse | None":
    """Set up LangFuse client *only when requested* and not on every import.

    Might raise exception if LangFuse is not configured.
    """
    from langfuse import get_client  # noqa: PLC0415

    client = get_client()

    try:
        client.auth_check()
        return client
    except Exception as e:
        logger.info(
            "LangFuse isn't set up correctly; "
            f"observability features would be unavailable: {e}"
        )


def maybe_setup_langfuse_instrumentation(service_name: str = "rlvr-grpo") -> None:
    """Configure Logfire to instrument OpenAI Agents and initialize Langfuse.

    Follows guidance from Langfuse docs on OpenAI Agents SDK instrumentation.
    This is a no-op if required packages or credentials are missing.
    """
    # Only attempt setup if Langfuse credentials seem present
    if not (getenv("LANGFUSE_PUBLIC_KEY") and getenv("LANGFUSE_SECRET_KEY")):
        return

    # Required before logfire
    # to avoid https://github.com/orgs/langfuse/discussions/9263#
    get_langfuse_client()

    try:
        import logfire  # type: ignore  # noqa: PLC0415
    except Exception:
        return

    try:
        # Configure logfire to export spans via OTLP (Langfuse client handles export)
        logfire.configure(
            service_name=getenv("LANGFUSE_SERVICE_NAME", service_name),
            send_to_logfire=False,
            console=False,
        )
        # Patch OpenAI Agents SDK to emit spans
        logfire.instrument_openai_agents()

    except Exception:
        # Swallow all errors to keep core functionality intact
        return


@backoff.on_exception(backoff.expo, httpx.TransportError)
async def maybe_traced(
    async_fn: Callable[[], Coroutine[None, None, V]],
    dataset_item: "DatasetItemClient | None",
    run_name: str,
) -> tuple[V, TraceID | None]:
    """Invoke async_fn using dataset_item.run context manager if available.

    Returns
    -------
        awaited output from async_fn, and
        LangFuse trace_id if dataset_item is provided.
    """
    if not dataset_item:
        return (await async_fn()), None

    with dataset_item.run(run_name=run_name) as root_span:
        result = await async_fn()
        root_span.update(
            input=dataset_item.input, output=getattr(result, "final_output", None)
        )

    return result, root_span.trace_id


def add_score(
    eval_results: "Sequence[EvalResult]",
    trace_ids: Sequence[TraceID | None],
    metric_name: str,
):
    """Apply scores to LangFuse run results; run this once for each metric."""
    langfuse_client = get_langfuse_client()
    if not langfuse_client:
        return

    for _result, _trace_id in zip(eval_results, trace_ids):
        if not _trace_id:
            continue

        langfuse_client.create_score(
            name=metric_name,
            value=_result.score,
            trace_id=_trace_id,
            comment=_result.explanation,
        )

    with spinner("Adding scores"):
        langfuse_client.flush()


def initialize_lf_dataset(
    items: "Sequence[RLVRDataItem]", dataset_name: str, metadata: Any | None = None
) -> None:
    """Create LangFuse dataset from the given items.

    Updates RLVRDataItem `items` in-place to make LangFuse dataset client available.
    """
    from langfuse._client.datasets import DatasetItemClient  # noqa: PLC0415

    langfuse_client = get_langfuse_client()
    if not langfuse_client:
        return

    langfuse_client.create_dataset(name=dataset_name, metadata=metadata)
    for _item in items:
        _lf_dataset_item = backoff.on_exception(backoff.expo, httpx.TransportError)(
            langfuse_client.create_dataset_item
        )(
            dataset_name=dataset_name,
            input=_item.query,
            expected_output=_item.target,
            metadata=_item.metadata,
        )
        _item.lf_dataset_client = DatasetItemClient(_lf_dataset_item, langfuse_client)

    with spinner("Uploading dataset"):
        langfuse_client.flush()


def silence_langfuse():
    """Hide tracing messages from LangFuse."""
    langfuse_logger = logging.getLogger("langfuse")
    langfuse_logger.setLevel(logging.ERROR)
    langfuse_logger.propagate = False

    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry.sdk").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry.instrumentation").setLevel(logging.WARNING)

    for _handler in list(langfuse_logger.handlers):
        langfuse_logger.removeHandler(_handler)
