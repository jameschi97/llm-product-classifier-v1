from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .azure_client import AzureClient
from .prompt_builder import PromptBuilder
from helpers.data_operations import JsonManager
from utils.logging import get_logger
from utils.console import console

logger = get_logger(__name__)


class Batcher:
    def __init__(self, client: AzureClient, parser: "Parser", builder: PromptBuilder):
        self.client = client
        self.parser = parser
        self.builder = builder

    def iterate_unclassified_batches(
        self,
        df: pd.DataFrame,
        target_col: str,
        batch_size: int,
    ) -> Iterable[pd.DataFrame]:
        """Yield batches of rows whose target column is still null."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        mask = df[target_col].isna()
        indices = df[mask].index.tolist()
        for i in range(0, len(indices), batch_size):
            chunk_idx = indices[i : i + batch_size]
            batch_df = df.loc[chunk_idx].copy()
            batch_df.insert(0, "ROW_ID", batch_df.index)
            yield batch_df


class Parser:
    def __init__(self, response_key: str = "department"):
        self.response_key = response_key

    @staticmethod
    def extract_first_json_array(text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("[")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def parse_classification_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response expecting classification results."""
        if not response_text:
            logger.warning("Empty response text; nothing to parse.")
            return []
        json_part = self.extract_first_json_array(response_text)
        if not json_part:
            logger.warning("No JSON array detected in response.")
            return []
        try:
            data = json.loads(json_part)
        except Exception as e:
            logger.warning("JSON decode failed: %s", e)
            return []
        if not isinstance(data, list):
            logger.warning(
                "Top-level JSON is not a list (type=%s); skipping.", type(data).__name__
            )
            return []
        rk = self.response_key
        results: List[Dict[str, Any]] = []
        for idx, obj in enumerate(data):
            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping non-dict entry at index %d (type=%s).",
                    idx,
                    type(obj).__name__,
                )
                continue
            if "row_id" not in obj:
                logger.warning("Missing row_id in entry at index %d: %s", idx, obj)
                continue
            if rk not in obj:
                logger.warning(
                    "Missing '%s' key in entry at index %d: %s", rk, idx, obj
                )
                continue
            try:
                row_id_val = int(obj["row_id"])
                label_val = str(obj[rk]).strip()
                
                if label_val == "":
                    logger.warning("Empty '%s' at index %d; skipped.", rk, idx)
                    continue
                
                results.append({"row_id": row_id_val, "label": label_val})
            except Exception as e:
                logger.warning(
                    "Failed coercion in entry at index %d: %s (err=%s)", idx, obj, e
                )
                continue
        return results


def _display_names(batch_df: pd.DataFrame, display_column: str) -> List[str]:
    """Human-readable names for console output (falls back to 'Row <id>')."""
    if display_column in batch_df.columns:
        return batch_df[display_column].fillna("(unknown)").tolist()
    return [f"Row {rid}" for rid in batch_df["ROW_ID"].tolist()]


def _apply_labels(
    df: pd.DataFrame,
    parsed: List[Dict[str, Any]],
    target_col: str,
    batch_num: int,
) -> tuple[int, int, Counter]:
    """Write parsed labels into rows that are still empty.

    Returns (applied, rejected, label_counts). A label is rejected when its row
    is unknown or already classified.
    """
    applied = 0
    rejected = 0
    label_counts: Counter = Counter()
    for obj in parsed:
        row_id = obj.get("row_id")
        label = obj.get("label")
        if not (isinstance(row_id, int) and isinstance(label, str)):
            continue
        if row_id in df.index and pd.isna(df.at[row_id, target_col]):
            df.at[row_id, target_col] = label
            label_counts[label] += 1
            applied += 1
        else:
            rejected += 1
            logger.warning(
                "Batch %d: row_id %d not found or already classified",
                batch_num,
                row_id,
            )
    return applied, rejected, label_counts


def _product_results(
    row_ids: List[int], names: List[str], parsed: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Per-product (name, assigned label) pairs shown in the console."""
    assigned = {
        obj["row_id"]: obj.get("label", "")
        for obj in parsed
        if isinstance(obj.get("row_id"), int)
    }
    return [
        {
            "product": str(names[i] if i < len(names) else f"Row {rid}"),
            "segment": assigned.get(rid, "(unclassified)"),
        }
        for i, rid in enumerate(row_ids)
    ]


def _save_partial(jm: JsonManager, df: pd.DataFrame, path: Optional[str]) -> None:
    """Write df to `path` after every batch, so a crash loses at most one batch."""
    if not path:
        return
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jm.write(output_path, df)
        logger.debug("Partial JSON written to %s", path)
    except Exception as e:
        logger.warning("Failed to write partial JSON '%s': %s", path, e)


def _ask_model(
    batcher: Batcher,
    batch_df: pd.DataFrame,
    context: str,
    label_names: List[str],
    label_descriptions: Dict[str, str],
    batch_num: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Prompt the model for one batch. Returns (parsed results, tokens used)."""
    prompt = batcher.builder.build_classification_prompt(
        batch_df,
        context=context,
        label_names=label_names,
        label_descriptions=label_descriptions,
    )
    logger.debug("Batch %d: prompt length=%d chars", batch_num, len(prompt))

    response_text, usage = batcher.client.send(prompt)
    tokens_used = usage.get("total_tokens", 0) if usage else 0
    logger.debug(
        "Batch %d: response length=%d chars, tokens=%d",
        batch_num,
        len(response_text) if response_text else 0,
        tokens_used,
    )

    parsed = batcher.parser.parse_classification_response(response_text or "")
    logger.debug("Batch %d: parsed %d items", batch_num, len(parsed))
    return parsed, tokens_used


def run_classification(
    batcher: Batcher,
    df: pd.DataFrame,
    context: str,
    label_names: List[str],
    label_descriptions: Dict[str, str],
    target_col: str,
    batch_size: int,
    display_column: str = "product_name",
    partial_output_json: Optional[str] = None,
    show_console_start: bool = True,
    progress_batch_offset: int = 0,
    progress_total_batches: Optional[int] = None,
    display_row_offset: Optional[int] = None,
) -> pd.DataFrame:
    """Classify products into the target column (modifies df in place)."""
    if target_col not in df.columns:
        df[target_col] = None

    total_rows = len(df)
    initial_unclassified = df[target_col].isna().sum()
    total_batches = (initial_unclassified + batch_size - 1) // batch_size
    effective_total_batches = progress_total_batches or total_batches

    logger.info(
        "Starting classification: %d total rows, %d unclassified, "
        "batch_size=%d, total_batches=%d",
        total_rows,
        initial_unclassified,
        batch_size,
        total_batches,
    )
    if show_console_start:
        console.classification_start(total_rows, batch_size, initial_unclassified)

    jm = JsonManager()
    batch_counter = 0
    rows_seen_for_display = 0

    for batch_df in batcher.iterate_unclassified_batches(df, target_col, batch_size):
        batch_start_time = time.time()
        batch_num = batch_counter + 1
        global_batch_num = progress_batch_offset + batch_num
        row_ids = batch_df["ROW_ID"].tolist()
        names = _display_names(batch_df, display_column)

        # Console row numbers continue from display_row_offset when it is given
        if display_row_offset is not None:
            row_start = display_row_offset + rows_seen_for_display
            console_ids = list(range(row_start, row_start + len(batch_df)))
            rows_seen_for_display += len(batch_df)
        else:
            console_ids = row_ids
        console.batch_start(global_batch_num, effective_total_batches, console_ids, names)
        logger.debug("Batch %d: row_ids=%s", batch_num, row_ids)
        logger.debug("Batch %d: display=%s", batch_num, names[:3])

        parsed, tokens_used = _ask_model(
            batcher, batch_df, context, label_names, label_descriptions, batch_num
        )
        applied, rejected, label_counts = _apply_labels(
            df, parsed, target_col, batch_num
        )
        failed = rejected + (len(batch_df) - len(parsed))  # unparsed rows count too

        batch_elapsed = time.time() - batch_start_time
        remaining_unclassified = df[target_col].isna().sum()

        console.batch_result(
            classified=applied,
            requested=len(batch_df),
            elapsed=batch_elapsed,
            category_counts=dict(label_counts),
            failed=failed if failed > 0 else 0,
            tokens=tokens_used,
            product_results=_product_results(row_ids, names, parsed),
        )
        logger.info(
            "Batch %d/%d: %d/%d classified in %.1fs. Labels: %s. Remaining: %d/%d",
            global_batch_num,
            effective_total_batches,
            applied,
            len(batch_df),
            batch_elapsed,
            dict(label_counts),
            remaining_unclassified,
            total_rows,
        )

        _save_partial(jm, df, partial_output_json)
        batch_counter += 1

    logger.info("Classification loop complete: %d batches processed", batch_counter)
    return df


__all__ = ["Batcher", "Parser", "run_classification"]
