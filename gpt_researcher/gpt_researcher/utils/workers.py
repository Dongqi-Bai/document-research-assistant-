import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from .rate_limiter import get_global_rate_limiter


class WorkerPool:
    def __init__(self, max_workers: int, rate_limit_delay: float = 0.0):
        """
        Initialize WorkerPool with concurrency and rate limiting.

        Args:
            max_workers: Maximum number of concurrent workers
            rate_limit_delay: Minimum seconds between requests GLOBALLY (0 = no limit)
                             This delay is enforced across ALL WorkerPools to prevent
                             overwhelming rate-limited APIs.
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        # Configure the global rate limiter
        # All WorkerPools share the same rate limiter instance
        global_limiter = get_global_rate_limiter()
        global_limiter.configure(rate_limit_delay)

    @asynccontextmanager
    async def throttle(self):
        """
        Throttle requests with both concurrency limiting and GLOBAL rate limiting.
        """
        async with self.semaphore:
            # Use global rate limiter (shared across all WorkerPools)
            global_limiter = get_global_rate_limiter()
            await global_limiter.wait_if_needed()
            yield
