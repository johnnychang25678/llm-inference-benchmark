"""Ollama process manager for benchmarks."""

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass

import httpx
import psutil


class OllamaUnresponsiveError(Exception):
    """Raised when Ollama becomes unresponsive (likely OOM)."""

    pass


@dataclass
class OllamaManager:
    """Manages Ollama server lifecycle for benchmarks.

    Handles starting/stopping Ollama with different OLLAMA_NUM_PARALLEL
    values and monitors memory usage.
    """

    ollama_url: str = "http://localhost:11434"
    startup_timeout: float = 60.0
    health_check_interval: float = 1.0

    async def restart_with_parallel(
        self,
        num_parallel: int,
        num_ctx: int = 32768,
    ) -> bool:
        """Stop Ollama, restart with OLLAMA_NUM_PARALLEL, wait for ready.

        Args:
            num_parallel: Value for OLLAMA_NUM_PARALLEL env var.
            num_ctx: Value for OLLAMA_NUM_CTX env var (context window size).

        Returns:
            True if Ollama started successfully.
        """
        await self.stop()
        await asyncio.sleep(2)  # Give time for cleanup

        # Start Ollama with new settings
        env = os.environ.copy()
        env["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
        env["OLLAMA_NUM_CTX"] = str(num_ctx)

        # Start ollama serve in background
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return await self.wait_for_ready()

    async def stop(self) -> None:
        """Stop all Ollama processes."""
        pids = self._find_ollama_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Wait for processes to stop
        for _ in range(10):
            if not self._find_ollama_pids():
                return
            await asyncio.sleep(0.5)

        # Force kill if still running
        for pid in self._find_ollama_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def wait_for_ready(self) -> bool:
        """Poll health endpoint until Ollama is ready.

        Returns:
            True if Ollama became ready within timeout.
        """
        async with httpx.AsyncClient() as client:
            elapsed = 0.0
            while elapsed < self.startup_timeout:
                try:
                    response = await client.get(
                        f"{self.ollama_url}/api/tags",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(self.health_check_interval)
                elapsed += self.health_check_interval

        return False

    async def check_health(self) -> bool:
        """Check if Ollama is responding.

        Returns:
            True if Ollama is healthy.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=5.0,
                )
                return response.status_code == 200
            except (httpx.ConnectError, httpx.TimeoutException):
                return False

    def get_ollama_memory_mb(self) -> float | None:
        """Get Ollama process memory usage in MB.

        Returns:
            Memory usage in MB, or None if Ollama not found.
        """
        total_memory = 0.0
        found = False

        for proc in psutil.process_iter(["name", "memory_info"]):
            try:
                # Match both "ollama" and "Ollama" process names
                if proc.info["name"] and "ollama" in proc.info["name"].lower():
                    found = True
                    total_memory += proc.info["memory_info"].rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total_memory if found else None

    def _find_ollama_pids(self) -> list[int]:
        """Find all running Ollama process PIDs.

        Returns:
            List of PIDs for Ollama processes.
        """
        pids = []
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if proc.info["name"] and "ollama" in proc.info["name"].lower():
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids
