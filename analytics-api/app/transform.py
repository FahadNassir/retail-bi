"""Clean, classify, and audit the UCI Online Retail dataset."""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def add_flag(flags: list, mask: pd.Series, flag_name: str) -> None:
    """Append a quality flag to every row where the mask is True."""
    for index in mask.fillna(False).loc[lambda x: x].index:
        flags[index].append(flag_name)


def _number(value: Any) -> float | None:
    """Convert a value to a number, returning None when conversion fails."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def transform_dataframe(
    source_df: pd.DataFrame,
    filename: str,
    output_dir: Path,
) -> dict:
    """
    Transform raw retail transactions into an analytical dataset.

    Exact duplicates are removed from the cleaned dataset but remain
    represented in the source totals and audit report.

    Records with critical missing or invalid fields are quarantined.
    Unusual business events are retained and classified for review.
    """

    # ---------------------------------------------------------
    # 1. Validate the input structure
    # ---------------------------------------------------------
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = len(source_df)

    # Preserve the original values for duplicate detection and auditing.
    original = source_df[EXPECTED_COLUMNS].copy()

    # Source row numbers are Excel row numbers, including the header row.
    original["source_excel_row"] = range(2, source_rows + 2)
    original["source_row_id"] = [
        f"{filename}:row:{row_number}"
        for row_number in original["source_excel_row"]
    ]

    # ---------------------------------------------------------
    # 2. Calculate source financial totals
    # ---------------------------------------------------------
    source_quantity = pd.to_numeric(
        source_df["Quantity"], errors="coerce"
    )
    source_unit_price = pd.to_numeric(
        source_df["UnitPrice"], errors="coerce"
    )

    source_line_value = source_quantity * source_unit_price
    calculable_source_values = source_line_value.dropna()

    source_line_value_total = round(
        float(calculable_source_values.sum()), 2
    )

    # ---------------------------------------------------------
    # 3. Identify and remove exact duplicate records
    # ---------------------------------------------------------
    duplicate_mask = source_df[EXPECTED_COLUMNS].duplicated(
        keep="first"
    )
    exact_duplicate_rows_removed = int(duplicate_mask.sum())

    # Retain the first occurrence and its original source row ID.
    working = original.loc[~duplicate_mask].copy()

    # ---------------------------------------------------------
    # 4. Standardise text fields and data types
    # ---------------------------------------------------------
    text_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Country",
    ]

    for column in text_columns:
        working[column] = (
            working[column]
            .astype("string")
            .str.strip()
        )
        working[column] = working[column].replace("", pd.NA)

    # Convert numeric fields safely.
    working["Quantity"] = pd.to_numeric(
        working["Quantity"], errors="coerce"
    )
    working["UnitPrice"] = pd.to_numeric(
        working["UnitPrice"], errors="coerce"
    )

    # Preserve the original date values for validation.
    original_dates = working["InvoiceDate"].copy()

    working["InvoiceDate"] = pd.to_datetime(
        working["InvoiceDate"], errors="coerce"
    )

    # Customer IDs should be whole-number identifiers.
    customer_numeric = pd.to_numeric(
        working["CustomerID"], errors="coerce"
    )

    non_missing_customer_ids = customer_numeric.dropna()
    customer_ids_are_whole_numbers = (
        non_missing_customer_ids
        .mod(1)
        .eq(0)
        .all()
    )

    if customer_ids_are_whole_numbers:
        working["CustomerID"] = customer_numeric.astype("Int64")
    else:
        working["CustomerID"] = customer_numeric

    # ---------------------------------------------------------
    # 5. Identify records that must be quarantined
    # ---------------------------------------------------------
    quarantine_masks = {
        "missing_invoice_number": working["InvoiceNo"].isna(),
        "missing_invoice_date": original_dates.isna(),
        "invalid_invoice_date": (
            original_dates.notna()
            & working["InvoiceDate"].isna()
        ),
        "missing_quantity": working["Quantity"].isna(),
        "invalid_quantity": (
            original["Quantity"].notna()
            & working["Quantity"].isna()
        ),
        "missing_unit_price": working["UnitPrice"].isna(),
        "invalid_unit_price": (
            original["UnitPrice"].notna()
            & working["UnitPrice"].isna()
        ),
    }

    quarantine_mask = pd.Series(
        False, index=working.index, dtype=bool
    )

    for mask in quarantine_masks.values():
        quarantine_mask |= mask.fillna(False)

    quarantined_records = working.loc[quarantine_mask].copy()
    cleaned = working.loc[~quarantine_mask].copy()

    # Record the specific reasons for quarantine.
    quarantined_records["quarantine_reasons"] = ""

    for reason, mask in quarantine_masks.items():
        affected = mask.fillna(False) & quarantine_mask

        for index in affected.loc[lambda x: x].index:
            current = quarantined_records.at[
                index, "quarantine_reasons"
            ]
            quarantined_records.at[index, "quarantine_reasons"] = (
                f"{current};{reason}" if current else reason
            )

    # ---------------------------------------------------------
    # 6. Classify business events using multiple signals
    # ---------------------------------------------------------
    invoice_text = (
        cleaned["InvoiceNo"]
        .astype("string")
        .str.upper()
    )

    description_text = (
        cleaned["Description"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    stock_code = (
        cleaned["StockCode"]
        .astype("string")
        .str.upper()
    )

    is_bad_debt = (
        stock_code.eq("B")
        & description_text.eq("ADJUST BAD DEBT")
    ).fillna(False)

    is_cancellation_invoice = invoice_text.str.startswith(
        "C", na=False
    )

    is_negative_quantity = cleaned["Quantity"] < 0
    is_negative_price = cleaned["UnitPrice"] < 0
    is_zero_price = cleaned["UnitPrice"] == 0
    is_positive_quantity = cleaned["Quantity"] > 0
    is_positive_price = cleaned["UnitPrice"] > 0

    # These records look like credits or returns, but lack enough
    # information to confidently interpret them as normal returns.
    is_unclassified_zero_value_credit = (
        is_negative_quantity
        & is_zero_price
        & cleaned["Description"].isna()
        & cleaned["CustomerID"].isna()
    )

    cleaned["transaction_type"] = "other_review"

    cleaned.loc[
        is_positive_quantity & is_positive_price,
        "transaction_type",
    ] = "sale_candidate"

    cleaned.loc[
        is_zero_price & ~is_negative_quantity,
        "transaction_type",
    ] = "zero_price_review"

    cleaned.loc[
        is_negative_price,
        "transaction_type",
    ] = "negative_price_review"

    cleaned.loc[
        is_negative_quantity,
        "transaction_type",
    ] = "return_credit_candidate"

    cleaned.loc[
        is_unclassified_zero_value_credit,
        "transaction_type",
    ] = "unclassified_zero_value_credit"

    # Stronger invoice/accounting signals override general categories.
    cleaned.loc[
        is_cancellation_invoice,
        "transaction_type",
    ] = "cancellation_invoice"

    cleaned.loc[
        is_bad_debt,
        "transaction_type",
    ] = "bad_debt_adjustment"

    # ---------------------------------------------------------
    # 7. Calculate line values and attach quality flags
    # ---------------------------------------------------------
    cleaned["LineValue"] = (
        cleaned["Quantity"] * cleaned["UnitPrice"]
    ).round(2)

    cleaned["quality_flags"] = [
        [] for _ in range(len(cleaned))
    ]

    flags = dict(zip(cleaned.index, cleaned["quality_flags"]))

    add_flag(
        flags,
        cleaned["Description"].isna(),
        "missing_description",
    )
    add_flag(
        flags,
        cleaned["CustomerID"].isna(),
        "missing_customer_id",
    )
    add_flag(
        flags,
        cleaned["Quantity"] < 0,
        "negative_quantity",
    )
    add_flag(
        flags,
        cleaned["UnitPrice"] == 0,
        "zero_unit_price",
    )
    add_flag(
        flags,
        cleaned["UnitPrice"] < 0,
        "negative_unit_price",
    )

    cleaned["quality_flags"] = [
        ";".join(row_flags)
        for row_flags in cleaned["quality_flags"]
    ]

    # ---------------------------------------------------------
    # 8. Build the audit report
    # ---------------------------------------------------------
    rows_after_deduplication = len(working)
    quarantined_rows = len(quarantined_records)
    cleaned_rows = len(cleaned)

    deduplicated_line_value_total = round(
        float(
            (
                pd.to_numeric(working["Quantity"], errors="coerce")
                * pd.to_numeric(
                    working["UnitPrice"], errors="coerce"
                )
            ).sum()
        ),
        2,
    )

    cleaned_line_value_total = round(
        float(cleaned["LineValue"].sum()), 2
    )

    quarantined_line_value_total = round(
        float(
            (
                pd.to_numeric(
                    quarantined_records["Quantity"],
                    errors="coerce",
                )
                * pd.to_numeric(
                    quarantined_records["UnitPrice"],
                    errors="coerce",
                )
            ).sum()
        ),
        2,
    )

    transaction_summary = (
        cleaned.groupby("transaction_type", dropna=False)
        .agg(
            rows=("InvoiceNo", "size"),
            line_value_total=("LineValue", "sum"),
        )
        .reset_index()
    )

    transaction_summary["line_value_total"] = (
        transaction_summary["line_value_total"].round(2)
    )

    audit = {
        "source_rows": source_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "rows_after_deduplication": rows_after_deduplication,
        "quarantined_rows": quarantined_rows,
        "cleaned_rows": cleaned_rows,
        "row_reconciliation_passed": (
            rows_after_deduplication
            == quarantined_rows + cleaned_rows
        ),
        "financial_reconciliation": {
            "source_rows_with_calculable_line_value": int(
                calculable_source_values.size
            ),
            "source_line_value_total": source_line_value_total,
            "deduplicated_line_value_total": (
                deduplicated_line_value_total
            ),
            "cleaned_line_value_total": cleaned_line_value_total,
            "quarantined_line_value_total": (
                quarantined_line_value_total
            ),
        },
        "quarantine_reasons": {
            reason: int(mask.fillna(False).sum())
            for reason, mask in quarantine_masks.items()
        },
        "transaction_summary": transaction_summary.to_dict(
            orient="records"
        ),
        "missing_description_rows_retained": int(
            cleaned["Description"].isna().sum()
        ),
        "missing_customer_id_rows_retained": int(
            cleaned["CustomerID"].isna().sum()
        ),
        "negative_quantity_rows_retained": int(
            (cleaned["Quantity"] < 0).sum()
        ),
        "zero_price_rows_retained": int(
            (cleaned["UnitPrice"] == 0).sum()
        ),
        "negative_price_rows_retained": int(
            (cleaned["UnitPrice"] < 0).sum()
        ),
    }

    # ---------------------------------------------------------
    # 9. Export the analytical dataset and audit artefacts
    # ---------------------------------------------------------
    cleaned_filename = "online_retail_cleaned_transactions.csv"
    quarantined_filename = "online_retail_quarantined_records.csv"
    audit_filename = "online_retail_audit_report.json"

    # Standardize column names before exporting the datasets.
    # Keep the source column names unchanged until all transformations finish.
    column_renames = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
        "LineValue": "line_value",
    }

    cleaned = cleaned.rename(columns=column_renames)
    quarantined_records = quarantined_records.rename(
        columns=column_renames
    )

    cleaned.to_csv(
        output_dir / cleaned_filename,
        index=False,
    )

    quarantined_records.to_csv(
        output_dir / quarantined_filename,
        index=False,
    )

    with open(
        output_dir / audit_filename,
        "w",
        encoding="utf-8",
    ) as audit_file:
        json.dump(audit, audit_file, indent=2, default=str)

    # ---------------------------------------------------------
    # 10. Return a compact response for the API and n8n
    # ---------------------------------------------------------
    return {
        "status": "completed",
        "dataset": filename,
        "audit": audit,
        "outputs": {
            "cleaned_transactions": cleaned_filename,
            "quarantined_records": quarantined_filename,
            "audit_report": audit_filename,
        },
    }
