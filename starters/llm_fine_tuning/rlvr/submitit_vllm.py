"""Launch `vLLM serve` using submitit."""

import asyncio
import dataclasses
import json
import logging
import re
import types
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, Sequence, TypeVar

import backoff
import httpx
import openai
import pydantic
import submitit
from httpx import URL
from rich.progress import Progress
from vllm.engine.arg_utils import EngineArgs


T = TypeVar("T")
VLLM_CLI_PREFIX = "uv run vllm serve"
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
        vllm_cli_prefix: str | list[str] = VLLM_CLI_PREFIX,
        max_timeouts: int = 5,
        worker_name: str = __name__,
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
        - max_timeouts: restart worker if it times out after this many checks.
        """
        self.logger = logging.getLogger(worker_name)

        # remote backend: url is None when backend is not ready.
        self.remote_status_signal = asyncio.Event()
        self.remote_url_signal = asyncio.Event()
        self.base_url: str | None = None
        self.executor = submitit_executor
        self.engine_args = engine_args
        self.vllm_cli_prefix = vllm_cli_prefix

        self.watcher_task: asyncio.Task[None]
        self.submitit_job, self.watcher_task = self._launch_worker()

        self.max_timeouts = max_timeouts
        self.num_timeouts = 0

    def _launch_worker(self):
        """Launch vLLM worker and return job handle as well as watcher task."""
        args = get_vllm_cli_args(self.engine_args, vllm_cli_prefix=self.vllm_cli_prefix)
        submitit_job = self.executor.submit(submitit.helpers.CommandFunction(args))
        self.logger.info(f"Launched: {submitit_job}")
        self.num_timeouts = 0

        watcher_task = asyncio.create_task(
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
        self.submitit_job, self.watcher_task = self._launch_worker()
        self.logger.info(f"Job interrupted. Relaunched {self.submitit_job}")

        # Raise exceptions (if any) from the previous watcher task.
        if not previous_watcher_task.cancel():
            previous_watcher_task.result()

    async def _handle_job_output(self, log_line: str):
        """Mark self as ready and update remote url when job is ready.

        Handles job output one line at a time.
        """
        self.logger.debug(log_line)
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
        assert self.base_url is not None
        try:
            response = await httpx.AsyncClient().get(f"{self.base_url}/ping")
            return response.status_code < 300
        except (httpx.RequestError, httpx.TimeoutException) as e:
            self.logger.info(e)
            return False

    async def get_client(self) -> openai.AsyncOpenAI:
        """Wait until backend is ready and return client.

        The method updates the cumulative server timeout count.
        """
        base_url = await self._get_url()
        server_ready = False
        while (not server_ready) and (self.num_timeouts < self.max_timeouts):
            server_ready = await self._test_if_ready()
            self.num_timeouts += 1

        if not server_ready:
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


async def gather_with_progress(
    coros: "list[types.CoroutineType[Any, Any, T]]",
    description: str = "Running tasks",
) -> Sequence[T]:
    """
    Run a list of coroutines concurrently, display a rich.Progress bar as each finishes.

    Returns the results in the same order as the input list.

    :param coros: List of coroutines to run.
    :return: List of results, ordered to match the input coroutines.
    """
    # Wrap each coroutine in a Task and remember its original index
    tasks = [
        asyncio.create_task(indexed(index=index, coro=coro))
        for index, coro in enumerate(coros)
    ]

    # Pre‐allocate a results list; we'll fill in each slot as its Task completes
    results: list[T | None] = [None] * len(tasks)

    # Create and start a Progress bar with a total equal to the number of tasks
    with Progress() as progress:
        progress_task = progress.add_task(description, total=len(tasks))

        # as_completed yields each Task as soon as it finishes
        for finished in asyncio.as_completed(tasks):
            index, result = await finished
            results[index] = result
            progress.update(progress_task, advance=1)

    # At this point, every slot in `results` is guaranteed to be non‐None
    # so we can safely cast it back to List[T]
    return results  # type: ignore


class SubmititVLLM:
    """Handles multiple replicas of submitit vLLM workers."""

    def __init__(
        self,
        engine_args: "EngineArgs",
        submitit_executor: submitit.SlurmExecutor,
        concurrency_per_worker: int,
        num_replicas: int,
        vllm_cli_prefix: str | list[str] = VLLM_CLI_PREFIX,
    ):
        self.logger = logging.getLogger("SubmititVLLM")

        self.workers = [
            SubmititVLLMWorker(
                engine_args=engine_args,
                submitit_executor=submitit_executor,
                vllm_cli_prefix=vllm_cli_prefix,
                worker_name=f"submitit_vllm.{_index}",
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


@backoff.on_exception(backoff.expo, openai.APIConnectionError)
async def generate_one(submitit_vllm: SubmititVLLM, prompt: str) -> URL:
    """Generate one request."""
    async with submitit_vllm.get_client() as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="model",
            max_completion_tokens=128,
        )
        content = response.choices[0].message.content
        assert content is not None
        return client.base_url


async def main():
    """Demonstrate full workflow."""
    engine_args = EngineArgs()

    submitit_executor = submitit.SlurmExecutor(
        folder="/scratch/ssd004/scratch/jacobtian/submitit",
        python="/h/jacobtian/vec-playbook/run_in_container.sh uv run python",
    )
    submitit_args = SubmititArgs(partition="a40", qos="m5", gres="gpu:1")

    submitit_executor.update_parameters(
        **submitit_args.to_submitit_parameters(), use_srun=False
    )
    engine_args = EngineArgs(
        model="/model-weights/Qwen3-8B",
        max_model_len=1024,
        served_model_name="model",
        enforce_eager=True,
    )

    submitit_vllm = SubmititVLLM(
        engine_args=engine_args,
        submitit_executor=submitit_executor,
        concurrency_per_worker=36,
        num_replicas=9,
    )
    coros = [
        generate_one(submitit_vllm, f"({_idx}) Introduce yourself.")
        for _idx in range(2160)
    ]

    try:
        base_urls = await gather_with_progress(coros, "Generating")
        for _url in base_urls:
            print(_url)

        print(Counter(base_urls))
    finally:
        submitit_vllm.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
