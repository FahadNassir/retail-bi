"""Application configuration sourced from Docker Compose environment."""

import os
from pathlib import Path

DATA_DIR = Path("/analytics-api/app/data")
OUTPUT_DIR = Path("/app/output")
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# These names match compose.yml; no DATABASE_URL variable is required.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]

# Columns required for retail-specific profiling and validation.
RETAIL_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]