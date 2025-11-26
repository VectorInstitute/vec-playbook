"""Test submitit local executor."""

import io
import os
import pathlib
import signal
import subprocess
import sys
import time

import submitit
import submitit.core.utils


class _CommandFunction(submitit.helpers.CommandFunction):
    """Submitit _CommandFunction, but handles Ctrl-C properly for LocalExecutor."""

    def __call__(self, *args, **kwargs) -> str:
        full_command = (
            self.command
            + [str(x) for x in args]
            + [f"--{x}={y}" for x, y in kwargs.items()]
        )
        if self.verbose:
            print(f'The following command is sent: "{" ".join(full_command)}"')
        with subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            cwd=self.cwd,
            env=self.env,
        ) as process:
            try:
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()

                submitit.core.utils.copy_process_streams(
                    process, stdout_buffer, stderr_buffer, self.verbose
                )
                return stdout_buffer.getvalue().strip()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGKILL)
                raise


# Use a local executor
log_folder = "submitit_local_logs"
os.makedirs(log_folder, exist_ok=True)

executor = submitit.LocalExecutor(folder=log_folder)
# Long timeout so we don't hit submitit's own timeout
executor.update_parameters(timeout_min=60)

# This command:
#   1. Writes its own PID to child_pid.txt
#   2. Sleeps for 600 seconds
#
# Importantly, this process is a *child of the submitit job process*,
# not the controller.
cmd = _CommandFunction(
    # cmd = submitit.helpers.CommandFunction(
    [
        sys.executable,
        "-u",
        "-c",
        (
            "import os, time, pathlib; "
            "path = pathlib.Path('child_pid.txt'); "
            "path.write_text(str(os.getpid())); "
            "time.sleep(600)"
        ),
    ]
)

print("Submitting CommandFunction job...")
job = executor.submit(cmd)
print(f"Submitted job with submitit job_id={job.job_id}")

# Give the child command time to start and write child_pid.txt
time.sleep(3)

pid_file = pathlib.Path("child_pid.txt")
if pid_file.exists():
    child_pid = pid_file.read_text().strip()
    print(f"child_pid.txt found. Child PID recorded as: {child_pid}")
else:
    print("child_pid.txt not found yet (command may not have started).")

print("Cancelling job via job.cancel(check=True)...")
job.cancel(check=True)

# job.cancel() is asynchronous for LocalExecutor; give it a moment
time.sleep(3)

print(f"After cancel: job.done() -> {job.done()}, job.state -> {job.state}")

if pid_file.exists():
    child_pid = pid_file.read_text().strip()
    print()
    print("===================================================")
    print("Repro instructions (run these in a *separate* shell)")
    print("===================================================")
    print("1) Inspect the child process:")
    print(f"   ps -p {child_pid} -o pid,ppid,cmd")
    print()
    print("You should see the python -c '...time.sleep(600)...' process")
    print("still running, even though submitit thinks the job is done and")
    print("we already called job.cancel().")
    print("===================================================")
else:
    print("Could not read child_pid.txt; the child may have exited early.")
