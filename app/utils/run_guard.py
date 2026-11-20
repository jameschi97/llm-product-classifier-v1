from __future__ import annotations

from typing import Callable

from config.exceptions import PipelineError
from utils.console import console
from utils.logging import get_logger

logger = get_logger(__name__)


def run_guarded(run: Callable[[], int]) -> int:
    """Run a pipeline and turn any failure into an exit code and a console message.

    Returns run()'s exit code; 130 on Ctrl+C; 1 on any error.
    """
    try:
        return run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        console.interrupted()
        return 130
    except PipelineError as e:
        logger.error("Pipeline error: %s", e)
        console.error("Pipeline Error", str(e))
        console.pipeline_finished(success=False)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        console.error("Unexpected Error", str(e))
        console.pipeline_finished(success=False)
        return 1
