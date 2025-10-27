"""Submitit launcher for vLLM server reusing the cluster Singularity image."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import pwd
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import suppress
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Optional, Union

import submitit
from pydantic import BaseModel, Field
from vllm.engine.arg_utils import AsyncEngineArgs as VllmAsyncEngineArgs


try:
    # pydantic v2
    from pydantic import ConfigDict  # type: ignore

    _HAS_CONFIGDICT = True
except Exception:
    _HAS_CONFIGDICT = False

SINGULARITY_IMAGE = Path("/projects/llm/unsloth-vllm-trl-latest.sif")
NSSWITCH_CONF = """\
passwd:         files
group:          files
shadow:         files
gshadow:        files
hosts:          files dns
networks:       files
protocols:      files
services:       files
ethers:         files
rpc:            files
netgroup:       files
"""


class ServeConfig(BaseModel):
    """Configuration for starting a vLLM OpenAI server under submitit.

    Args:
        engine_args: vLLM AsyncEngineArgs (recommended) or a plain dict of fields.
        host: Listening host for the HTTP server (defaults to "0.0.0.0").
        port: Listening port. Use 0 to auto-pick a free port on the worker.
        log_folder: Where submitit writes logs and where we drop readiness JSON.
        cluster: Submitit cluster mode (e.g., "slurm", "local", "debug").
            If None, AutoExecutor auto-detects.
        slurm_partition: Slurm partition, if applicable.
        timeout_min: Scheduler timeout for the job.
    """

    # Allow arbitrary vLLM types (v1/v2 compatible)
    if _HAS_CONFIGDICT:
        model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore
    else:

        class Config:
            """Fallback configuration for pydantic v1."""

            arbitrary_types_allowed = True  # type: ignore

    engine_args: Union[VllmAsyncEngineArgs, dict[str, Any]]
    host: str = "0.0.0.0"
    port: int = 0
    log_folder: Path = Field(default_factory=lambda: Path("./_submitit_logs"))
    cluster: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_params: dict[str, Any] = Field(default_factory=dict)


ServeConfig.model_rebuild()


def _to_engine_kwargs(
    engine_args: Union["VllmAsyncEngineArgs", dict[str, Any]],
) -> dict[str, Any]:
    """Return a plain dict of AsyncEngineArgs fields suitable for CLI overlay.

    Accepts either:
      - an instance of vllm.engine.arg_utils.AsyncEngineArgs
      - a pre-constructed dict[str, Any] of the same fields
    """
    if isinstance(engine_args, dict):
        return engine_args

    # vLLM EngineArgs/AsyncEngineArgs are dataclasses
    try:
        if dataclasses.is_dataclass(engine_args):
            # Preserve nested dataclass instances (pydantic configs) so vLLM
            # keeps attribute access semantics during validation.
            return {
                field.name: getattr(engine_args, field.name)
                for field in dataclasses.fields(engine_args)
            }
    except Exception:
        pass

    # Fallback reflection (best effort)
    return {
        k: getattr(engine_args, k) for k in dir(engine_args) if not k.startswith("_")
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _prepare_overlay_files() -> dict[str, Path]:
    overlay_root = Path.home() / ".local/overlay"
    etc_dir = overlay_root / "etc"
    etc_dir.mkdir(parents=True, exist_ok=True)

    nsswitch = overlay_root / "nsswitch.conf"
    nsswitch.write_text(NSSWITCH_CONF, encoding="utf-8")

    uid = os.getuid()
    gid = os.getgid()
    user = os.environ.get("USER") or f"u{uid}"
    try:
        pw_entry = pwd.getpwuid(uid)
        home = pw_entry.pw_dir or os.environ.get("HOME") or f"/home/{user}"
        shell = pw_entry.pw_shell or os.environ.get("SHELL") or "/bin/sh"
    except KeyError:
        home = os.environ.get("HOME") or f"/home/{user}"
        shell = os.environ.get("SHELL") or "/bin/sh"

    passwd_entry = f"{user}:x:{uid}:{gid}:{user}:{home}:{shell}\n"
    passwd_file = etc_dir / "passwd"
    system_passwd = Path("/etc/passwd").read_text(encoding="utf-8")
    passwd_file.write_text(passwd_entry + system_passwd, encoding="utf-8")

    group_file = etc_dir / "group"
    group_lines: list[str] = []
    for group_id in sorted({gid, *os.getgroups()}):
        try:
            entry = subprocess.check_output(
                ["getent", "group", str(group_id)], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if entry:
            group_lines.append(entry)
    system_group = Path("/etc/group").read_text(encoding="utf-8")
    group_payload = "\n".join(group_lines)
    if group_payload:
        group_payload += "\n"
    group_file.write_text(group_payload + system_group, encoding="utf-8")

    return {
        "root": overlay_root,
        "etc_dir": etc_dir,
        "nsswitch": nsswitch,
        "passwd": passwd_file,
        "group": group_file,
    }


def _detect_munge_socket() -> Optional[str]:
    for candidate in (
        Path("/opt/munge/var/run/munge/munge.socket.2"),
        Path("/run/munge/munge.socket.2"),
        Path("/var/run/munge/munge.socket.2"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _invoke_in_singularity(
    engine_kwargs: dict[str, Any],
    frontend_kwargs: dict[str, Any],
    ready_file: str,
) -> None:
    payload_file = (
        Path(ready_file)
        .with_name(Path(ready_file).name + ".payload.json")
        .resolve()
    )
    payload = {
        "engine_kwargs": engine_kwargs,
        "frontend_kwargs": frontend_kwargs,
        "ready_file": ready_file,
    }
    payload_file.write_text(
        json.dumps(payload, default=_json_default, indent=2),
        encoding="utf-8",
    )

    if not SINGULARITY_IMAGE.exists():
        raise FileNotFoundError(
            f"Singularity image not found at {SINGULARITY_IMAGE}. "
            "Update SINGULARITY_IMAGE if the path has changed."
        )

    overlay = _prepare_overlay_files()
    env = os.environ.copy()
    env["SINGULARITYENV_SLURM_CONF"] = "/opt/slurm/etc/slurm.conf"
    env["SINGULARITYENV_PATH"] = f"/opt/slurm/bin:{env.get('PATH', '')}"
    env["SINGULARITYENV_LD_LIBRARY_PATH"] = (
        "/opt/slurm/lib:/opt/slurm/lib64:"
        "/opt/munge/lib:/opt/munge/lib64:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    if munge_socket := _detect_munge_socket():
        env["SINGULARITYENV_MUNGE_SOCKET"] = munge_socket
    env["SINGULARITYENV_VLLM_LAUNCHER_INSIDE_SINGULARITY"] = "1"
    env["SINGULARITYENV_VLLM_LAUNCHER_PAYLOAD"] = str(payload_file)

    binds = [
        "/model-weights:/model-weights",
        "/projects/llm:/projects/llm",
        "/scratch/ssd004/scratch/jacobtian:/scratch/ssd004/scratch/",
        "/scratch/ssd004/scratch/jacobtian:/scratch/",
        "/fs01/home/jacobtian/:/fs01/home/jacobtian/",
        "/opt/:/opt/",
        "/opt/slurm/:/opt/slurm",
        "/opt/munge/var/run/munge:/opt/munge/var/run/munge",
        f"{overlay['nsswitch']}:/etc/nsswitch.conf",
        f"{overlay['passwd']}:/etc/passwd",
        f"{overlay['group']}:/etc/group",
        "/var/run/munge:/var/run/munge",
        "/etc/munge:/etc/munge",
    ]

    singularity_cmd: list[str] = ["singularity", "exec", "--nv"]
    for bind in binds:
        singularity_cmd.extend(["--bind", bind])
    singularity_cmd.append(str(SINGULARITY_IMAGE))
    singularity_cmd.extend(
        [
            "uv",
            "run",
            "python",
            "-m",
            "starters.llm_fine_tuning.rlvr.vllm_launcher",
            "--container-worker",
            str(payload_file),
        ]
    )

    bash_script = dedent(
        f"""
        set -euo pipefail
        source ~/.bashrc >/dev/null 2>&1 || true
        if [ -f /opt/lmod/lmod/init/bash ]; then
          source /opt/lmod/lmod/init/bash || true
          export MODULEPATH=/opt/modulefiles:/pkgs/modulefiles:/pkgs/environment-modules
          module load singularity-ce >/dev/null 2>&1 || true
        fi
        {shlex.join(singularity_cmd)}
        """
    ).strip()

    process = subprocess.Popen(
        ["bash", "-lc", bash_script],
        env=env,
        cwd=os.getcwd(),
    )
    try:
        return_code = process.wait()
    except BaseException:
        process.terminate()
        with suppress(Exception):
            process.wait(timeout=15)
        raise
    finally:
        with suppress(FileNotFoundError):
            payload_file.unlink()

    if return_code:
        raise RuntimeError(f"Singularity execution failed with exit code {return_code}")


def _run_vllm_engine(
    engine_kwargs: dict[str, Any], frontend_kwargs: dict[str, Any], ready_file: str
) -> None:
    """Run on the submitit worker: construct args from AsyncEngineArgs and serve.

    Steps:
        1) Build a parser with default values (FrontendArgs + AsyncEngineArgs).
        2) Overlay provided engine/front-end keys.
        3) Pre-bind the socket (setup_server) to get a real port (0 => ephemeral).
        4) Start a single uvicorn worker with run_server_worker.

    Args:
        engine_kwargs: Dict of fields from AsyncEngineArgs.
        frontend_kwargs: Minimal server options like host/port/uds.
        ready_file: Path to write JSON {"host": "...", "port": int} once bound.
    """
    try:
        import uvloop  # type: ignore

        run_coroutine = uvloop.run
    except ModuleNotFoundError:
        run_coroutine = asyncio.run

    if importlib.util.find_spec("vllm") is None:
        uv_bin = shutil.which("uv")
        if uv_bin:
            try:
                subprocess.run([uv_bin, "sync"], check=True, env=os.environ.copy())
            except subprocess.CalledProcessError as sync_err:
                env_root = os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")
                print(
                    "[vllm_launcher] `uv sync` failed on worker; "
                    f"environment {env_root} may be incomplete. "
                    f"Return code: {sync_err.returncode}",
                    flush=True,
                )
        else:
            env_root = os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")
            print(
                "[vllm_launcher] `uv` binary not found on worker. "
                "Install uv or pre-provision the project environment at "
                f"{env_root}.",
                flush=True,
            )
        # Re-check after attempting sync
        if importlib.util.find_spec("vllm") is None:
            import sys

            env_root = os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")
            uv_hint = (
                "run `uv sync` on the worker or adjust UV_VENVS_BASE "
                "to a shared filesystem"
            )
            print(
                "[vllm_launcher] Failed to import vLLM modules after sync attempt. "
                f"Ensure the worker environment at {env_root} has vLLM installed "
                f"({uv_hint}). sys.path={sys.path}",
                flush=True,
            )
            raise ModuleNotFoundError("vllm") from None  # Preserve original failure signature

    from vllm.entrypoints.openai.api_server import run_server_worker, setup_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    # 1) Build a parser with all defaults populated (same defaults as `vllm serve`).
    parser = make_arg_parser(FlexibleArgumentParser(prog="vllm-serve-programmatic"))
    args = parser.parse_args([])

    # 2) Overlay engine args and frontend args into the Namespace (only known keys).
    for key, value in engine_kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in frontend_kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # 3) Pre-bind the socket and emit readiness information.
    listen_address, sock = setup_server(args)
    host_to_report = socket.getfqdn()
    actual_port = sock.getsockname()[1]
    Path(ready_file).write_text(
        json.dumps({"host": host_to_report, "port": int(actual_port)}),
        encoding="utf-8",
    )

    # 4) Block inside the uvicorn worker.
    run_coroutine(run_server_worker(listen_address, sock, args))


def _worker_vllm_serve_engine(
    engine_kwargs: dict[str, Any], frontend_kwargs: dict[str, Any], ready_file: str
) -> None:
    if os.environ.get("VLLM_LAUNCHER_INSIDE_SINGULARITY") == "1":
        _run_vllm_engine(engine_kwargs, frontend_kwargs, ready_file)
        return
    _invoke_in_singularity(engine_kwargs, frontend_kwargs, ready_file)


class VllmJobHandle:
    """Convenience wrapper over submitit.Job with a coroutine stop().

    Use .job to access the raw submitit.Job (id, logs, stdout/stderr).
    """

    def __init__(self, job: submitit.Job):
        self._job = job

    @property
    def job(self) -> submitit.Job:
        """Underlying submitit Job object."""
        return self._job

    async def stop(self, wait: float = 20.0) -> None:
        """Request cancellation and wait briefly for the job to finish.

        Args:
            wait: Maximum seconds to wait after requesting cancel.
        """
        with suppress(Exception):
            self._job.cancel()

        deadline = time.time() + wait
        while time.time() < deadline and not self._job.done():
            await asyncio.sleep(0.25)

        if self._job.done():
            with suppress(Exception):
                _ = self._job.result()


async def launch_vllm_with_submitit(
    config: ServeConfig,
) -> tuple[Awaitable[tuple[str, int]], VllmJobHandle]:
    """Launch `vllm serve` under submitit using typed AsyncEngineArgs.

    Returns a tuple whose first element awaits `(hostname, port)` once the
    worker binds, while the second element exposes lifecycle control via
    ``VllmJobHandle``. Port ``0`` chooses a free port; readiness metadata is
    written to a JSON file and awaited before resolving. The worker reuses the
    same ``setup_server`` and ``run_server_worker`` codepaths as ``vllm serve``.
    """
    log_dir = Path(config.log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    ready_file = log_dir / f"vllm_ready_{os.getpid()}_{uuid.uuid4().hex[:6]}.json"

    engine_kwargs = _to_engine_kwargs(config.engine_args)
    frontend_kwargs: dict[str, Any] = {"host": config.host, "port": int(config.port)}

    executor = submitit.AutoExecutor(folder=str(log_dir), cluster=config.cluster)
    executor.update_parameters()
    executor.update_parameters(**config.slurm_params)

    job = executor.submit(
        _worker_vllm_serve_engine, engine_kwargs, frontend_kwargs, str(ready_file)
    )
    handle = VllmJobHandle(job)

    async def _wait_ready() -> tuple[str, int]:
        while True:
            if ready_file.exists():
                try:
                    data = json.loads(ready_file.read_text(encoding="utf-8"))
                    return str(data["host"]), int(data["port"])
                except Exception:
                    pass  # File could be mid-write.
            if job.done():
                try:
                    _ = job.result()
                except Exception as e:
                    raise RuntimeError("vLLM server job failed early") from e
                raise RuntimeError("vLLM server exited before writing readiness info")
            await asyncio.sleep(0.25)

    return asyncio.create_task(_wait_ready()), handle


async def main():
    """Test full setup."""
    coro, job_handle = await launch_vllm_with_submitit(
        ServeConfig(
            engine_args=VllmAsyncEngineArgs(
                model="/model-weights/Qwen3-0.6B",
                served_model_name="model",
            ),
            slurm_params={
                "timeout_min": 180,
                "slurm_partition": "a40",
                "slurm_qos": "scavenger",
                # "slurm_account": "aip-x-ksupport",
                # "slurm_account": "aip-rrabba",
                "cpus_per_task": 8,
                "slurm_gres": "gpu:1",
                "mem_gb": 60,
            },
        )
    )
    try:
        host, port = await coro
        print(f"http://{host}:{port}/v1")

        while True:
            await asyncio.sleep(3600)

    finally:
        await job_handle.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container-worker",
        type=Path,
        default=None,
        help="Internal flag used when re-invoking inside the Singularity container.",
    )
    parsed_args, _ = parser.parse_known_args()
    if parsed_args.container_worker:
        payload_location = parsed_args.container_worker
        payload = json.loads(payload_location.read_text(encoding="utf-8"))
        ready_file_value = payload.get("ready_file")
        if not ready_file_value:
            raise ValueError("Container payload missing `ready_file` entry.")
        _run_vllm_engine(
            dict(payload.get("engine_kwargs", {})),
            dict(payload.get("frontend_kwargs", {})),
            str(ready_file_value),
        )
    else:
        asyncio.run(main())
