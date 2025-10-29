"""Launch `vLLM serve` using submitit."""

import asyncio
import dataclasses
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Mapping,
    TypeVar,
)

import httpx
import openai
import pydantic
import submitit
from vllm.engine.arg_utils import EngineArgs


T = TypeVar("T")
VLLM_CLI_PREFIX = "uv run vllm serve"
SERVED_MODEL_NAME = "model"
VLLM_URL_PATTERN = re.compile(r"http:\/\/0\.0\.0\.0[:\.](\d{1,5})", re.DOTALL)
VLLM_READY_PATTERN = re.compile(r"startup complete", re.DOTALL)


class SubmititArgs(pydantic.BaseModel):
    """Submitit args."""

    job_name: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    mem: str | None = "72GB"
    cpus_per_task: str | None = "16"
    gres: str | None = "gpu:1"
    time: str = "1:00:00"
    use_srun: bool = False

    def to_submitit_parameters(self) -> dict[str, int | str]:
        """Produce submit-compatible dict consisting of non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


def _flatten_dict(nested: Mapping[str, Any]) -> dict[tuple[str, ...], Any]:
    """
    Flatten a nested dictionary.

    Paths use original mapping keys for sequences.
    """
    out: dict[tuple[str, ...], Any] = {}
    stack: list[tuple[tuple[str, ...], Any]] = [((), nested)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, Mapping):
            for k, v in value.items():
                stack.append((path + (k,), v))
        else:
            out[path] = value
    return out


def _serialize_config_element(element):
    """Serialize various objects in vllm_config.

    Handles of a mix of Pydantic dataclass and built-in dataclass.
    """
    if isinstance(element, set):
        return [_serialize_config_element(_item) for _item in element]

    return getattr(element, "__dict__", element)


def _serialize_vllm_config(engine_args: EngineArgs) -> dict[tuple[str, ...], Any]:
    """Serialize vllm engine args."""
    return _flatten_dict(
        json.loads(
            json.dumps(
                dataclasses.asdict(engine_args), default=_serialize_config_element
            )
        )
    )


def get_vllm_cli_args(
    engine_args: EngineArgs,
    vllm_cli_prefix: str | list[str] = VLLM_CLI_PREFIX,
    port: int | None = None,
) -> list[str]:
    """Obtain vLLM serve command given engine args.

    Port is randomly generated.
    """
    if isinstance(vllm_cli_prefix, str):
        vllm_cli_prefix = vllm_cli_prefix.split(" ")

    engine_args_dict = _serialize_vllm_config(engine_args)
    engine_args_dict_ref = _serialize_vllm_config(EngineArgs())

    engine_args_patch = {
        k: v for k, v in engine_args_dict.items() if v != engine_args_dict_ref.get(k)
    }

    # Add only non-default values to cli.
    # "model" flag must be specified as a position arg.
    output = [*vllm_cli_prefix, engine_args_patch.pop(("model",))]
    for k, v in engine_args_patch.items():
        if isinstance(v, bool):
            if v:
                output.append(f"--{".".join(k)}")
        else:
            output.extend((f"--{".".join(k)}", str(v)))

    if port is None:
        port = uuid.uuid4().int % (65536 - 10000) + 10000

    output.extend(("--port", str(port)))

    return output


async def _follow(
    path: Path, callback: Callable[[str], Coroutine[None, None, None]], poll: float
) -> None:
    """
    Tail a file and invoke `callback` for every new line until cancelled.

    Args:
        path: Path to the log file.
        callback: Callback invoked per line (without trailing newline).
        poll: Sleep interval when no new data is available.
    """
    f = path.open("r", encoding="utf-8", errors="ignore")
    pending_tasks: list[asyncio.Task] = []
    try:
        while True:
            line = f.readline()
            if line:
                pending_tasks.append(asyncio.create_task(callback(line.rstrip("\n"))))
            else:
                await asyncio.sleep(poll)
    except asyncio.CancelledError:
        # Final quick drain before exiting.
        for line in f.readlines():
            pending_tasks.append(asyncio.create_task(callback(line.rstrip("\n"))))

        await asyncio.gather(*pending_tasks)
        raise

    finally:
        await asyncio.gather(*pending_tasks)
        f.close()


async def async_tail_job_logs(
    job: submitit.Job,
    on_stdout: Callable[[str], Coroutine[None, None, None]],
    on_stderr: Callable[[str], Coroutine[None, None, None]],
    on_interrupt: Callable[[], Coroutine[None, None, None]],
    poll: float = 0.25,
) -> None:
    """
    Asynchronously follow stdout/stderr and invoke callbacks for each new line.

    Args:
        job: Submitit job returned by executor.submit(...).
        on_stdout: Callback for stdout lines.
        on_stderr: Callback for stderr lines.
        on_interrupt: Callback when job exits (not when this task is cancelled.)
        poll: Poll interval for file growth and completion checks.
    """
    out_p, err_p = job.paths.stdout, job.paths.stderr

    # Wait for any log file to appear or the job to finish early.
    while not ((out_p.exists() and err_p.exists()) or job.done()):
        await asyncio.sleep(poll)

    # Both files will exist (or at least one); create tasks concisely.
    tasks = [
        asyncio.create_task(_follow(_path, _callback, poll))
        for _path, _callback in [(out_p, on_stdout), (err_p, on_stderr)]
    ]

    try:
        while not job.done():
            await asyncio.sleep(poll)
        await asyncio.sleep(poll * 2)  # allow final flush
        await on_interrupt()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=False)


class SubmititVLLMWorker:
    """Watcher for one replica of vLLM launched via submitit."""

    def __init__(
        self,
        engine_args: "EngineArgs",
        submitit_executor: submitit.SlurmExecutor,
        max_num_timeouts: int,
        vllm_cli_prefix: str | list[str] = VLLM_CLI_PREFIX,
        worker_name: str = __name__,
        ping_sticky_seconds: float = 1.0,
    ):
        """
        Create Submitit watcher.

        Note that the job is started automatically.
        If the job is interrupted/preempted, the watcher will launch a new one.

        To launch multiple workers, create one instance of this class
        for each worker instance.

        Invoke the `stop` function to actually stop the watcher
        and any related jobs.

        Params:
        - engine_args: vLLM engine args. Must be CLI serializable.
        - submitit_args: configs for submitit
        - vllm_cli_prefix: shell command for launching `vllm serve`.
        - worker_name: for logging.
        - max_num_timeouts: restart worker if it times out after this many checks.
        - ping_sticky_seconds: each successful ping is valid for this many seconds.
        """
        self.logger = logging.getLogger(worker_name)
        self.metrics_logger = logging.getLogger(f"{worker_name}.metrics")

        # remote backend: url is None when backend is not ready.
        self.remote_status_signal = asyncio.Event()
        self.remote_url_signal = asyncio.Event()
        self.base_url: str | None = None
        self.executor = submitit_executor
        self.engine_args = engine_args
        self.vllm_cli_prefix = vllm_cli_prefix

        self.watcher_task: asyncio.Task[None]
        self.submitit_job, self.watcher_task = self._launch_worker()

        self.ping_lock = asyncio.Lock()
        self.ping_is_ready: bool = False
        self.ping_sticky_seconds = ping_sticky_seconds
        self.max_num_timeouts = max_num_timeouts
        self.num_timeouts = 0

    def _launch_worker(self):
        """Launch vLLM worker and return job handle as well as watcher task."""
        args = get_vllm_cli_args(self.engine_args, vllm_cli_prefix=self.vllm_cli_prefix)
        submitit_job = self.executor.submit(submitit.helpers.CommandFunction(args))
        self.logger.info(f"Launched: {submitit_job}")
        self.num_timeouts = 0

        watcher_task = asyncio.get_event_loop().create_task(
            async_tail_job_logs(
                submitit_job,
                on_stdout=self._handle_job_output,
                on_stderr=self._handle_job_output,
                on_interrupt=self._handle_job_interrupted,
            )
        )
        return submitit_job, watcher_task

    def stop(self):
        """Stop worker and raise watcher exceptions (if any)."""
        if not self.watcher_task.cancel():
            self.watcher_task.result()

        self.submitit_job.cancel()
        self.remote_status_signal.set()

    async def _handle_job_interrupted(self):
        """Handle job interruption by launching again."""
        self.base_url = None
        self.remote_url_signal.clear()
        self.remote_status_signal.clear()
        self.submitit_job.cancel()

        # This callback is running within the previous self.watcher_task
        # stop the previous watcher_task only after launching a new one.
        previous_watcher_task = self.watcher_task
        previous_job = self.submitit_job
        self.submitit_job, self.watcher_task = self._launch_worker()
        self.logger.info(
            f"Job interrupted: {previous_job}. Relaunched: {self.submitit_job}"
        )

        # Raise exceptions (if any) from the previous watcher task.
        if not previous_watcher_task.cancel():
            previous_watcher_task.result()

    async def _handle_job_output(self, log_line: str):
        """Mark self as ready and update remote url when job is ready.

        Handles job output one line at a time.
        """
        self.logger.debug(log_line)
        if "throughput" in log_line:
            self.metrics_logger.info(log_line)

        url_port_match = VLLM_URL_PATTERN.search(log_line)
        ready_match = VLLM_READY_PATTERN.search(log_line)

        # Note the possible race condition- base_url might appear before or after
        # the "ready" status is seen.
        if url_port_match is not None:
            _job_host = self.submitit_job.get_info()["NodeList"]
            self.base_url = f"http://{_job_host}:{url_port_match.group(1)}"
            self.remote_url_signal.set()
            self.logger.debug(f"Base URL {self.base_url}")

        if ready_match is not None:
            # Unblock remote_status_signal only after url is ready.
            await self.remote_url_signal.wait()
            self.remote_status_signal.set()
            self.logger.info(f"Ready: {self.base_url}")

    async def _get_url(self) -> str:
        """Return base url when ready, blocking (async) until ready."""
        await self.remote_status_signal.wait()
        assert self.base_url is not None
        return self.base_url

    async def _test_if_ready(self) -> bool:
        """Check if the server is ready and reachable."""
        await self.remote_status_signal.wait()

        # Max one ping at a time
        async with self.ping_lock:
            # Each successful ping is good for ping_sticky_seconds.
            if self.ping_is_ready:
                return True

            assert self.base_url is not None
            try:
                response = await httpx.AsyncClient().get(f"{self.base_url}/ping")
                response.raise_for_status()
            except (httpx.RequestError, httpx.TimeoutException) as e:
                self.logger.warning(e)
                self.num_timeouts += 1
                return False

            self.ping_is_ready = True
            asyncio.create_task(self._reset_ping_status())
            return True

    async def _reset_ping_status(self):
        """Reset ping status to False after timer."""
        await asyncio.sleep(self.ping_sticky_seconds)
        self.ping_is_ready = False

    async def get_client(self) -> openai.AsyncOpenAI:
        """Wait until backend is ready and return client.

        The method updates the cumulative server timeout count.
        """
        base_url = await self._get_url()
        server_ready = False
        while not server_ready:
            server_ready = await self._test_if_ready()

            if self.num_timeouts >= self.max_num_timeouts:
                self.logger.warning(f"Timeouts: {self.num_timeouts}; restarting")
                await self._handle_job_interrupted()

        self.num_timeouts = 0
        return openai.AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="EMPTY", max_retries=0
        )


async def _rate_limited(
    fn: Callable[[], Coroutine[None, None, T]], semaphore: asyncio.Semaphore
) -> tuple[T, asyncio.Semaphore]:
    """Wait for fn, subject to semaphore.

    When this function returns normally, the semaphore isn't released automatically
    but kept active.

    While this function cleans up the semaphore automatically if fn did not go through,
    the caller is responsible for releasing the semaphore of all tasks that
    *did* go through.
    """
    await semaphore.acquire()
    try:
        result = await fn()
        return result, semaphore

    # Clean up if the fn wait did not go through.
    except asyncio.CancelledError as e:
        semaphore.release()
        raise e


async def indexed(index: int, coro: Coroutine[None, None, T]) -> tuple[int, T]:
    """Return (index, await coro)."""
    return index, (await coro)


class SubmititVLLM:
    """Handles multiple replicas of submitit vLLM workers."""

    def __init__(
        self,
        engine_args: "EngineArgs",
        submitit_executor: submitit.SlurmExecutor,
        concurrency_per_worker: int,
        num_replicas: int,
        vllm_cli_prefix: str | list[str] = VLLM_CLI_PREFIX,
        logger_name_prefix: str = "submitit_vllm",
    ):
        self.logger = logging.getLogger(f"{logger_name_prefix}.controller")

        if engine_args.served_model_name and (
            engine_args.served_model_name != engine_args.model
        ):
            self.logger.warning(
                "engine_args.served_model_name != engine_args.model. "
                "Overriding `served_model_name` with `model`"
            )

        # Use model_name as served_model_name
        self.served_model_name = engine_args.model
        engine_args.served_model_name = self.served_model_name

        with submitit_executor.batch():
            self.workers = [
                SubmititVLLMWorker(
                    engine_args=engine_args,
                    submitit_executor=submitit_executor,
                    vllm_cli_prefix=vllm_cli_prefix,
                    max_num_timeouts=2 * concurrency_per_worker,
                    worker_name=f"{logger_name_prefix}.worker_{_index:03d}",
                )
                for _index in range(num_replicas)
            ]

        self.worker_semaphores = [
            asyncio.Semaphore(concurrency_per_worker) for _ in range(num_replicas)
        ]

    def stop(self):
        """Spin down workers and raise any worker exceptions."""
        worker_exceptions: list[Exception] = []
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                worker_exceptions.append(e)

        if worker_exceptions:
            raise RuntimeError(*worker_exceptions)

    def __enter__(self):
        """Use SubmititVLLM in context manager mode for auto-clean-up."""
        return self

    def __exit__(self, *_):
        """Clean up."""
        self.stop()
        return False

    @asynccontextmanager
    async def get_client(self):
        """Obtain rate-limited load-balanced OpenAI Async client.

        Blocks asynchronously if no backend is available yet.
        """
        # Return first available worker, subject to per-worker rate limit.
        _tasks = [
            asyncio.create_task(
                _rate_limited(lambda _worker=_worker: _worker.get_client(), _semaphore)
            )
            for _worker, _semaphore in zip(self.workers, self.worker_semaphores)
        ]

        try:
            # Avoid await in the following block to avoid race conditions

            # take the first finished result and release the rest.
            (first, *others), pending = await asyncio.wait(
                _tasks, return_when=asyncio.FIRST_COMPLETED
            )
            # this block is synchronous,
            # so no "pending" task would become ready in the meantime!
            for _task in others:
                _, _semaphore = _task.result()
                _semaphore.release()

            for _task in pending:
                _task.cancel()

            # only one task is left- yield this one and
            # remember to release the sempahore when completed.
            client, semaphore = first.result()
            self.logger.debug(f"Yielding: {client.base_url}")
            yield client

            semaphore.release()

        # Release semaphore
        finally:
            for _task in _tasks:
                _task.cancel()

    @asynccontextmanager
    async def get_oai_agents_config(self):
        """Wrap around get_client."""
        import agents

        async with self.get_client() as client:
            yield agents.RunConfig(
                model=agents.OpenAIChatCompletionsModel(
                    model=self.served_model_name, openai_client=client
                )
            )
