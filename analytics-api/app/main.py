import json
from typing import Any
from pathlib import Path

import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg.types.json import Jsonb

from app.config import ALLOWED_EXTENSIONS, DATA_DIR, OUTPUT_DIR, RETAIL_COLUMNS
from app.database import get_db_connection
from app.transform import transform_dataframe

app = FastAPI(
    title="Retail BI Analytics API",
    description="Dataset profiling and data quality validation.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=[],
)

def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible Python values."""

    if value is None:
        return None

    if not pd.api.types.is_scalar(value):
        return str(value)

    if pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def resolve_dataset(filename: str) -> Path:
    """Resolve a dataset filename without allowing path traversal."""

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Provide a filename, not a filesystem path.",
        )

    path = DATA_DIR / filename

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported formats: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {filename}",
        )

    return path


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a supported dataset using pandas."""

    path = resolve_dataset(filename)

    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)

        return pd.read_excel(path, sheet_name=0)

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read dataset: {exc}",
        ) from exc


def profile_column(series: pd.Series) -> dict[str, Any]:
    """Summarize one column without returning the entire column."""

    missing_count = int(series.isna().sum())
    total = len(series)

    # Limit sample values to keep API responses manageable.
    sample_values = [
        json_safe(value)
        for value in series.dropna().drop_duplicates().head(5).tolist()
    ]

    return {
        "name": str(series.name),
        "inferred_type": str(series.dtype),
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_count / total * 100, 2
        ) if total else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": sample_values,
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    severity: str,
    count: int,
    message: str,
    sample_rows: list[int] | None = None,
) -> None:
    """Add a consistent validation finding to the report."""

    checks.append({
        "check": name,
        "severity": severity,
        "count": int(count),
        "message": message,
        "sample_excel_rows": sample_rows or [],
    })


def excel_row_numbers(mask: pd.Series, limit: int = 5) -> list[int]:
    """Return approximate Excel row numbers, assuming row 1 is the header."""

    positions = mask.fillna(False).to_numpy().nonzero()[0][:limit]
    return [int(position + 2) for position in positions]


@app.get("/health")
def health() -> dict[str, str]:
    """Health check for Docker and service monitoring."""

    return {"status": "ok"}


@app.get("/datasets")
def list_datasets() -> dict[str, list[dict[str, Any]]]:
    """List supported files available in the mounted data directory."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(DATA_DIR.iterdir())
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    return {"datasets": datasets}


@app.get("/profile")
def profile_dataset(
    filename: str = Query(..., description="Dataset filename"),
) -> dict[str, Any]:
    """Generate a structural and statistical profile of a dataset."""

    df = load_dataset(filename)
    row_count, column_count = df.shape

    columns = [
        profile_column(df[column])
        for column in df.columns
    ]

    missing_cells = int(df.isna().sum().sum())
    total_cells = row_count * column_count

    return {
        "dataset": filename,
        "dimensions": {
            "rows": row_count,
            "columns": column_count,
        },
        "quality": {
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": round(
                df.duplicated().mean() * 100, 2
            ) if row_count else 0.0,
            "columns_with_missing_values": int(
                (df.isna().sum() > 0).sum()
            ),
            "missing_cells": missing_cells,
            "missing_cell_percentage": round(
                missing_cells / total_cells * 100, 2
            ) if total_cells else 0.0,
        },
        "columns": columns,
    }


@app.get("/validate")
def validate_dataset(
    filename: str = Query(..., description="Dataset filename"),
) -> dict[str, Any]:

    print(f"Starting validation: {filename}", flush=True)

    df = load_dataset(filename)

    print(f"Excel loaded: {len(df)} rows", flush=True)

    checks: list[dict[str, Any]] = []

    # Required columns determine whether retail-specific checks are possible.
    missing_columns = [
        column for column in RETAIL_COLUMNS
        if column not in df.columns
    ]

    add_check(
        checks,
        name="expected_columns",
        severity="error" if missing_columns else "info",
        count=len(missing_columns),
        message=(
            f"Missing expected columns: {missing_columns}"
            if missing_columns
            else "All expected retail columns are present."
        ),
    )

    if missing_columns:
        return {
            "dataset": filename,
            "rows": len(df),
            "passed": False,
            "checks": checks,
            "summary": {
                "errors": sum(c["severity"] == "error" for c in checks),
                "warnings": sum(c["severity"] == "warning" for c in checks),
                "info": sum(c["severity"] == "info" for c in checks),
            },
            "note": "Retail-specific checks stopped because columns are missing.",
        }

    # Exact duplicate rows are flagged, not automatically removed.
    print("Starting duplicate check", flush=True)
    duplicate_mask = df.duplicated(keep=False)
    print("Duplicate check complete", flush=True)
    duplicate_count = int(df.duplicated().sum())

    add_check(
        checks,
        name="exact_duplicate_rows",
        severity="warning" if duplicate_count else "info",
        count=duplicate_count,
        message=(
            "Exact duplicate rows exist; investigate before deduplicating."
            if duplicate_count
            else "No exact duplicate rows found."
        ),
        sample_rows=excel_row_numbers(duplicate_mask),
    )

    # Missing values in fields important for transaction analysis.
    for column in ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]:
        missing_mask = df[column].isna()
        missing_count = int(missing_mask.sum())

        add_check(
            checks,
            name=f"missing_{column}",
            severity="error" if missing_count else "info",
            count=missing_count,
            message=f"{missing_count} rows have no value for {column}.",
            sample_rows=excel_row_numbers(missing_mask),
        )

    # Description and CustomerID may be missing in real source records.
    for column in ["Description", "CustomerID"]:
        missing_mask = df[column].isna()
        missing_count = int(missing_mask.sum())

        add_check(
            checks,
            name=f"missing_{column}",
            severity="warning" if missing_count else "info",
            count=missing_count,
            message=(
                f"{missing_count} rows lack {column}; "
                "retain for now and assess analytical impact."
            ),
            sample_rows=excel_row_numbers(missing_mask),
        )

    # Convert values for validation only; the original DataFrame is unchanged.
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(df["UnitPrice"], errors="coerce")

    invalid_quantity = df["Quantity"].notna() & quantity.isna()
    invalid_price = df["UnitPrice"].notna() & unit_price.isna()

    for name, mask in [
        ("non_numeric_quantity", invalid_quantity),
        ("non_numeric_unit_price", invalid_price),
    ]:
        count = int(mask.sum())

        add_check(
            checks,
            name=name,
            severity="error" if count else "info",
            count=count,
            message=f"{count} rows contain values that cannot be parsed numerically.",
            sample_rows=excel_row_numbers(mask),
        )

    # Zero and negative quantities/prices are findings, not automatic failures.
    quantity_checks = [
        (
            "zero_quantity",
            quantity.eq(0),
            "warning",
            "Zero-quantity rows need investigation.",
        ),
        (
            "negative_quantity",
            quantity.lt(0),
            "info",
            "Negative quantities may represent returns or credit notes.",
        ),
        (
            "zero_unit_price",
            unit_price.eq(0),
            "warning",
            "Zero-price rows need investigation; they may have a valid business reason.",
        ),
        (
            "negative_unit_price",
            unit_price.lt(0),
            "warning",
            "Negative prices need investigation.",
        ),
    ]

    for name, mask, severity, message in quantity_checks:
        count = int(mask.fillna(False).sum())

        add_check(
            checks,
            name=name,
            severity=severity if count else "info",
            count=count,
            message=f"{message} Found: {count}.",
            sample_rows=excel_row_numbers(mask),
        )

    # Cancellation invoices are identified by an InvoiceNo starting with C.
    invoice_text = df["InvoiceNo"].astype("string").str.strip()
    cancellation_mask = invoice_text.str.upper().str.startswith(
        "C", na=False
    )

    add_check(
        checks,
        name="cancellation_invoice_rows",
        severity="info",
        count=int(cancellation_mask.sum()),
        message=(
            "Rows associated with cancellation invoice numbers. "
            "Do not treat these automatically as ordinary sales."
        ),
        sample_rows=excel_row_numbers(cancellation_mask),
    )

    # Validate invoice dates without overwriting the source column.
    parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invalid_dates = df["InvoiceDate"].notna() & parsed_dates.isna()

    add_check(
        checks,
        name="unparseable_invoice_dates",
        severity="error" if invalid_dates.any() else "info",
        count=int(invalid_dates.sum()),
        message="Non-empty InvoiceDate values that pandas could not parse.",
        sample_rows=excel_row_numbers(invalid_dates),
    )

    # Calculate indicative transaction-value totals for reconciliation.
    line_value = quantity * unit_price
    valid_line_value = line_value.dropna()

    value_summary = {
        "rows_with_calculable_line_value": int(valid_line_value.size),
        "positive_line_value_total": round(
            float(valid_line_value[valid_line_value > 0].sum()), 2
        ),
        "negative_line_value_total": round(
            float(valid_line_value[valid_line_value < 0].sum()), 2
        ),
        "net_line_value_total": round(float(valid_line_value.sum()), 2),
    }

    errors = sum(check["severity"] == "error" for check in checks)
    warnings = sum(check["severity"] == "warning" for check in checks)
    info = sum(check["severity"] == "info" for check in checks)

    return {
        "dataset": filename,
        "rows": len(df),
        "passed": errors == 0,
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "info": info,
        },
        "checks": checks,
        "value_summary": value_summary,
        "note": (
            "Validation reports findings only. No records have been "
            "deleted, corrected, or transformed."
        ),
    }


@app.get("/anomalies")
def inspect_anomalies(
    filename: str = Query(..., description="Dataset filename"),
) -> dict[str, Any]:
    """Return representative records for investigating data anomalies."""

    df = load_dataset(filename)

    # Include original column values and approximate Excel row numbers.
    def records_with_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        records = []

        for index, row in frame.iterrows():
            record = {
                column: json_safe(value)
                for column, value in row.items()
            }
            record["source_excel_row"] = int(index + 2)
            records.append(record)

        return records

    # Find the two negative-price records.
    prices = pd.to_numeric(df["UnitPrice"], errors="coerce")
    negative_prices = df.loc[prices < 0]

    # Inspect cancellation invoices, including their quantities and prices.
    invoice_text = df["InvoiceNo"].astype("string").str.strip()
    cancellations = df.loc[
        invoice_text.str.upper().str.startswith("C", na=False)
    ].head(5)

    # Select up to five distinct exact-duplicate groups.
    duplicate_rows = df.loc[df.duplicated(keep=False)]
    duplicate_groups = []

    if not duplicate_rows.empty:
        grouped = duplicate_rows.groupby(
            list(df.columns),
            dropna=False,
            sort=False,
        )

        for _, group in grouped:
            duplicate_groups.append({
                "group_size": len(group),
                "records": records_with_rows(group),
            })

            if len(duplicate_groups) == 5:
                break
            # Investigate negative quantities outside cancellation invoices.
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    invoice_text = df["InvoiceNo"].astype("string").str.strip()

    non_cancellation_returns = df.loc[
        (quantity < 0)
        & ~invoice_text.str.upper().str.startswith("C", na=False)
    ].head(10)

    zero_price_rows = df.loc[
        pd.to_numeric(df["UnitPrice"], errors="coerce") == 0
    ].head(5)
    return {
        "dataset": filename,
        "negative_unit_price_rows": records_with_rows(negative_prices),
        "cancellation_rows": records_with_rows(cancellations),
        "duplicate_groups": duplicate_groups,
        "negative_quantity_non_cancellation_rows": records_with_rows(
            non_cancellation_returns
        ),
        "zero_price_rows": records_with_rows(zero_price_rows),
        "note": (
            "Inspection only. No source records have been modified."
        ),
    }

@app.post("/transform")
def transform_dataset(
    filename: str = Query(..., description="Dataset filename"),
) -> dict[str, Any]:
    """Run the pandas transformation and save its output files."""

    df = load_dataset(filename)

    try:
        return transform_dataframe(
            source_df=df,
            filename=filename,
            output_dir=OUTPUT_DIR,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


def resolve_output_file(filename: str) -> Path:
    """Resolve a generated output filename safely."""

    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Provide a filename, not a filesystem path.",
        )

    path = OUTPUT_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Output file not found: {filename}",
        )

    return path


@app.post("/load")
def load_transformed_data(
    filename: str = Query(..., description="Original dataset filename"),
    cleaned_filename: str = Query(...),
    quarantine_filename: str = Query(...),
    audit_filename: str = Query(...),
) -> dict[str, Any]:
    """Load transformed CSVs and record their audit report atomically."""

    cleaned_path = resolve_output_file(cleaned_filename)
    quarantine_path = resolve_output_file(quarantine_filename)
    audit_path = resolve_output_file(audit_filename)

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read audit report: {exc}",
        ) from exc

    if not audit.get("row_reconciliation_passed", False):
        raise HTTPException(
            status_code=422,
            detail="Audit reconciliation failed; data was not loaded.",
        )

    sales_columns = (
        "invoice_no, stock_code, description, quantity, invoice_date, "
        "unit_price, customer_id, country, source_excel_row, source_row_id, "
        "transaction_type, line_value, quality_flags"
    )

    quarantine_columns = (
        "invoice_no, stock_code, description, quantity, invoice_date, "
        "unit_price, customer_id, country, source_excel_row, source_row_id, "
        "quarantine_reasons"
    )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TEMP TABLE stage_sales (
                        LIKE sales_transactions INCLUDING DEFAULTS
                    ) ON COMMIT DROP
                """)

                cur.execute("""
                    CREATE TEMP TABLE stage_quarantine (
                        LIKE quarantined_records INCLUDING DEFAULTS
                    ) ON COMMIT DROP
                """)

                # Stream the cleaned CSV into a temporary staging table.
                with cleaned_path.open("rb") as file:
                    with cur.copy(
                        f"COPY stage_sales ({sales_columns}) "
                        "FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
                    ) as copy:
                        while chunk := file.read(1024 * 1024):
                            copy.write(chunk)

                # A header-only quarantine file is valid.
                with quarantine_path.open("rb") as file:
                    with cur.copy(
                        f"COPY stage_quarantine ({quarantine_columns}) "
                        "FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
                    ) as copy:
                        while chunk := file.read(1024 * 1024):
                            copy.write(chunk)

                cur.execute("SELECT COUNT(*) FROM stage_sales")
                cleaned_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM stage_quarantine")
                quarantine_count = cur.fetchone()[0]

                # Replace the current dataset snapshot in this transaction.
                cur.execute("TRUNCATE sales_transactions, quarantined_records")

                cur.execute(f"""
                    INSERT INTO sales_transactions ({sales_columns})
                    SELECT {sales_columns}
                    FROM stage_sales
                """)

                cur.execute(f"""
                    INSERT INTO quarantined_records ({quarantine_columns})
                    SELECT {quarantine_columns}
                    FROM stage_quarantine
                """)

                # Record the same audit report that describes this load.
                cur.execute("""
                    INSERT INTO pipeline_runs (
                        dataset,
                        status,
                        source_rows,
                        cleaned_rows,
                        quarantined_rows,
                        duplicate_rows_removed,
                        audit_report
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING run_id
                """, (
                    filename,
                    "completed",
                    audit["source_rows"],
                    cleaned_count,
                    quarantine_count,
                    audit["exact_duplicate_rows_removed"],
                    Jsonb(audit),
                ))

                run_id = cur.fetchone()[0]
        # The transaction has committed successfully by this point.
        return {
            "status": "completed",
            "run_id": run_id,
            "dataset": filename,
            "cleaned_rows_loaded": cleaned_count,
            "quarantined_rows_loaded": quarantine_count,
        }

    except (psycopg.Error, OSError, KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Data loading failed; transaction rolled back: {exc}",
        ) from exc


@app.get("/dashboard/summary")
def dashboard_summary():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)::int AS cleaned_rows,
                        COUNT(DISTINCT invoice_no)::int AS invoice_count,
                        COUNT(DISTINCT customer_id)::int
                            AS identified_customers,
                        COUNT(DISTINCT country)::int AS countries,
                        ROUND(
                            COALESCE(
                                SUM(line_value) FILTER (
                                    WHERE transaction_type = 'sale_candidate'
                                ),
                                0
                            )::numeric,
                            2
                        ) AS candidate_sales_value
                    FROM sales_transactions
                """)
                metrics = cur.fetchone()

                cur.execute("""
                    SELECT
                        run_id,
                        dataset,
                        status,
                        source_rows,
                        cleaned_rows
                    FROM pipeline_runs
                    ORDER BY run_id DESC
                    LIMIT 1
                """)
                latest_run = cur.fetchone()

        return {
            "metrics": metrics,
            "latest_pipeline_run": latest_run,
        }

    except (psycopg.Error, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dashboard error: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/dashboard/sales-over-time")
def dashboard_sales_over_time():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        TO_CHAR(
                            DATE_TRUNC('month', invoice_date),
                            'YYYY-MM'
                        ) AS month,
                        ROUND(
                            SUM(line_value) FILTER (
                                WHERE transaction_type = 'sale_candidate'
                            )::numeric,
                            2
                        ) AS candidate_sales_value
                    FROM sales_transactions
                    WHERE invoice_date IS NOT NULL
                    GROUP BY DATE_TRUNC('month', invoice_date)
                    ORDER BY DATE_TRUNC('month', invoice_date)
                """)
                rows = cur.fetchall()

        return {"sales_over_time": rows}

    except (psycopg.Error, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Sales-over-time data is temporarily unavailable.",
        ) from exc

@app.get("/dashboard/product-performance")
def dashboard_product_performance():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        stock_code,
                        COALESCE(
                            MAX(NULLIF(BTRIM(description), '')),
                            stock_code
                        ) AS product,
                        ROUND(SUM(line_value)::numeric, 2)
                            AS candidate_sales_value
                    FROM sales_transactions
                    WHERE transaction_type = 'sale_candidate'
                      AND NULLIF(BTRIM(stock_code), '') IS NOT NULL
                    GROUP BY stock_code
                    ORDER BY SUM(line_value) DESC
                    LIMIT 10
                """)
                rows = cur.fetchall()

        return {"product_performance": rows}

    except (psycopg.Error, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Product performance data is temporarily unavailable.",
        ) from exc

@app.get("/dashboard/geographic-breakdown")
def dashboard_geographic_breakdown():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COALESCE(
                            NULLIF(BTRIM(country), ''),
                            'Unknown'
                        ) AS country,
                        ROUND(
                            SUM(line_value)::numeric,
                            2
                        ) AS candidate_sales_value
                    FROM sales_transactions
                    WHERE transaction_type = 'sale_candidate'
                    GROUP BY COALESCE(
                        NULLIF(BTRIM(country), ''),
                        'Unknown'
                    )
                    ORDER BY SUM(line_value) DESC
                    LIMIT 10
                """)
                rows = cur.fetchall()

        return {"geographic_breakdown": rows}

    except (psycopg.Error, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Geographic breakdown data is temporarily unavailable.",
        ) from exc

@app.get("/dashboard/transaction-composition")
def dashboard_transaction_composition():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        transaction_type,
                        COUNT(*)::int AS record_count
                    FROM sales_transactions
                    GROUP BY transaction_type
                    ORDER BY record_count DESC
                """)
                rows = cur.fetchall()

        total_records = sum(row["record_count"] for row in rows)

        return {
            "total_records": total_records,
            "transaction_composition": rows,
        }

    except (psycopg.Error, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Transaction composition data is temporarily unavailable.",
        ) from exc