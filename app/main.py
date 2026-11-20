"""
Production Mode - Database-to-Database Classification Pipeline.

This is the production pipeline that:
- Fetches unclassified rows directly from the database
- Classifies products in batches
- Writes results back to the database immediately
- NEVER stores data locally

Safety Features:
- Only processes rows where target column IS NULL or empty
- Only updates rows where target column IS NULL or empty (double-check)
- Idempotent: safe to re-run multiple times

Usage:
    python app/main.py

Configuration:
    config.yaml  — table, columns, labels, prompts, batch size
    .env         — credentials and infrastructure settings
"""
from __future__ import annotations

import time

import pandas as pd
from dotenv import load_dotenv

from config.loader import PipelineConfig, load_config
from db.db_connector import DBConnector
from services import Batcher, build_batcher, run_classification
from utils.logging import get_logger, init_logging
from utils.console import console
from utils.run_guard import run_guarded


# -------------------- Setup -------------------- #
init_logging("main")
logger = get_logger(__name__)


def _collect_updates(df: pd.DataFrame, cfg: PipelineConfig) -> list[dict]:
    """Rows that now have a label, as [{primary_key: value, "label": value}]."""
    pk = cfg.source.primary_key
    updates = []
    for _, row in df.iterrows():
        label_value = row.get(cfg.target.column)
        pk_value = row.get(pk)
        if pd.notna(label_value) and label_value != "" and pk_value is not None:
            updates.append({pk: pk_value, "label": label_value})
    return updates


def classify_batch_and_update(
    connector: DBConnector,
    df: pd.DataFrame,
    batcher: Batcher,
    cfg: PipelineConfig,
    batch_num: int,
    total_batches: int,
    row_offset: int,
) -> int:
    """Classify a batch and write results back to DB.

    Returns number of rows successfully updated.
    """
    if df.empty:
        return 0

    target_col = cfg.target.column
    if target_col not in df.columns:
        df[target_col] = None

    # Run classification (modifies df in place)
    run_classification(
        batcher,
        df,
        context=cfg.prompt.classification_context,
        label_names=cfg.label_names,
        label_descriptions=cfg.label_descriptions,
        target_col=target_col,
        batch_size=cfg.batch_size,
        display_column=cfg.display_column,
        partial_output_json=None,  # No local JSON in production
        show_console_start=False,
        progress_batch_offset=batch_num - 1,
        progress_total_batches=total_batches,
        display_row_offset=row_offset,
    )

    updates = _collect_updates(df, cfg)
    if not updates:
        logger.warning("No classifications to update")
        return 0

    return connector.update_classifications(
        updates,
        table=cfg.source.table,
        schema=cfg.source.schema,
        target_col=target_col,
        primary_key=cfg.source.primary_key,
    )


def _process_batches(
    connector: DBConnector,
    batcher: Batcher,
    cfg: PipelineConfig,
    total_unclassified: int,
) -> int:
    """Fetch, classify and write back batches until none are left. Returns rows updated."""
    total_updated = 0
    batch_num = 0
    rows_processed = 0
    total_batches = (total_unclassified + cfg.batch_size - 1) // cfg.batch_size
    console.classification_start(total_unclassified, cfg.batch_size, total_unclassified)

    while True:
        batch_num += 1
        batch_start = time.time()

        # Always offset=0: updated rows drop out of the "unclassified" query
        df = connector.fetch_unclassified_batch(
            batch_size=cfg.batch_size,
            offset=0,
            table=cfg.source.table,
            schema=cfg.source.schema,
            target_col=cfg.target.column,
            primary_key=cfg.source.primary_key,
        )
        if df.empty:
            logger.info("No more unclassified rows. Stopping.")
            break

        logger.info("Processing batch %d: %d rows", batch_num, len(df))
        updated = classify_batch_and_update(
            connector,
            df,
            batcher,
            cfg,
            batch_num=batch_num,
            total_batches=total_batches,
            row_offset=rows_processed,
        )
        total_updated += updated
        rows_processed += len(df)

        logger.info(
            "Batch %d complete: %d updated in %.1fs. Total: %d/%d. Remaining: ~%d",
            batch_num,
            updated,
            time.time() - batch_start,
            total_updated,
            total_unclassified,
            total_unclassified - total_updated,
        )

        # Safety: rows fetched but none updated would loop forever
        if updated == 0 and len(df) > 0:
            logger.warning(
                "Batch had %d rows but 0 updates. Breaking to avoid infinite loop.",
                len(df),
            )
            break

    return total_updated


def _run() -> int:
    pipeline_start = time.time()
    load_dotenv()
    cfg = load_config()

    logger.info("[PRODUCTION MODE] Pipeline starting")
    logger.info(
        "Config: schema=%s, table=%s, target=%s, batch_size=%d, pk=%s",
        cfg.source.schema,
        cfg.source.table,
        cfg.target.column,
        cfg.batch_size,
        cfg.source.primary_key,
    )
    console.start(
        "Production Pipeline",
        f"Connecting to [{cfg.source.schema}].[{cfg.source.table}]...",
    )

    connector = DBConnector()
    connector.connect_and_verify(schema=cfg.source.schema, table=cfg.source.table)

    total_unclassified = connector.count_unclassified_rows(
        table=cfg.source.table,
        schema=cfg.source.schema,
        target_col=cfg.target.column,
    )
    if total_unclassified == 0:
        logger.info("No unclassified rows found. Nothing to do.")
        console.info("Complete", "No unclassified rows found.")
        console.pipeline_finished(success=True)
        return 0

    logger.info("Found %d unclassified rows to process", total_unclassified)
    console.info("Unclassified Rows", f"{total_unclassified} rows to process")

    batcher = build_batcher(cfg)
    total_updated = _process_batches(connector, batcher, cfg, total_unclassified)

    total_elapsed = time.time() - pipeline_start
    logger.info("[PRODUCTION MODE] Pipeline complete in %.1fs", total_elapsed)
    logger.info("Total rows updated: %d", total_updated)
    logger.info("Rate limiter stats: %s", batcher.client.rate_limiter_stats)

    console.info("Complete", f"Updated {total_updated} rows in {total_elapsed:.1f}s")
    console.pipeline_finished(success=True)
    return 0


def main() -> int:
    """Run the production pipeline. Returns the exit code."""
    return run_guarded(_run)


if __name__ == "__main__":
    exit(main())
