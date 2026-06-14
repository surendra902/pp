from __future__ import annotations

import asyncio
import sys

from polybot.config import Config
from polybot.logging_setup import get_logger


log = get_logger("main")


async def async_main(cfg: Config) -> None:
    # This commit focuses on *structure* and critical bugfixes.
    # Wiring the full bot loop is the next step.
    log.info("Boot OK (refactor skeleton). dry_run=%s", cfg.dry_run)
    await asyncio.sleep(0.01)


def main() -> None:
    cfg = Config.from_env()
    errs = cfg.validate()
    if errs:
        for e in errs:
            print("ERROR:", e)
        sys.exit(1)

    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
