from __future__ import annotations

from config.exceptions import PipelineError
from config.loader import PipelineConfig
from utils.logging import get_logger

from .azure_client import AzureClient
from .classification_orchestrator import Batcher, Parser
from .prompt_builder import PromptBuilder

logger = get_logger(__name__)


def build_batcher(cfg: PipelineConfig) -> Batcher:
    """Wire the Azure client, prompt builder and parser from config.

    Shared by production (app/main.py) and test mode (tests/test_runner.py).
    """
    client = AzureClient.from_env(
        system_message=cfg.prompt.system_message,
        max_rpm=cfg.max_rpm,
    )
    if client is None:
        raise PipelineError("Azure OpenAI not configured (missing env vars)")

    logger.info(
        "Azure client initialized: deployment=%s, rate_limit=%d RPM",
        client.deployment,
        cfg.max_rpm,
    )

    builder = PromptBuilder(
        context_columns=cfg.context_columns,
        instructions_template=cfg.prompt.instructions_template,
        response_key=cfg.prompt.response_key,
        fallback_label=cfg.fallback_label,
    )
    parser = Parser(response_key=cfg.prompt.response_key)
    return Batcher(client=client, parser=parser, builder=builder)
