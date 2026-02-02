import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# ============================================================
# CONFIG
# ============================================================

@dataclass
class CleaningConfig:
    input_csv: str = "raw_data.csv"
    output_dir: str = "outputs"

    require_email: bool = True  
    require_name: bool = False          
    drop_invalid_email: bool = True     

    conflict_resolution: str = "MOST_FREQUENT"  
    
    fill_numeric: str = "MEDIAN"        
    fill_categorical: str = "MODE"    

    keep_best_record_per_email: bool = True

    log_level: str = "INFO"

# ============================================================
# LOGGER
# ============================================================

def setup_logger(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("SMART_CLEANING")


# ============================================================
# HELPERS
# ============================================================

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common placeholder strings into np.nan + strip text columns."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    placeholders = ["", "nan", "NaN", "None", "NULL", "null",
                    "unknown", "Unknown", "not_known", "not known", "N/A", "na"]

    df.replace(placeholders, np.nan, inplace=True)
    return df


def clean_name(name):
    if pd.isna(name):
        return np.nan
    name = str(name).strip()
    if name == "":
        return np.nan
    return name.title()


def clean_city(city):
    if pd.isna(city):
        return np.nan
    city = str(city).strip()
    if city == "":
        return np.nan
    return city.title()


def clean_gender(g):
    if pd.isna(g):
        return np.nan
    g = str(g).strip().lower()
    mapping = {"m": "male", "male": "male", "f": "female", "female": "female"}
    return mapping.get(g, np.nan)


def validate_email(email):
    if pd.isna(email):
        return np.nan
    email = str(email).strip().lower()
    if email == "":
        return np.nan
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return email if re.match(pattern, email) else np.nan


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_signup_date(df: pd.DataFrame, col="signup_date") -> pd.DataFrame:
    """Parse signup_date and create signup_year/month/day without breaking pipeline."""
    if col not in df.columns:
        return df

    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    df["signup_year"] = df[col].dt.year
    df["signup_month"] = df[col].dt.month
    df["signup_day"] = df[col].dt.day
    df.drop(columns=[col], inplace=True)
    return df


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """Higher score => more complete row."""
    cols = ["age", "gender", "income", "city", "signup_year", "signup_month", "signup_day"]
    score = pd.Series(0, index=df.index, dtype=int)
    for c in cols:
        if c in df.columns:
            score += df[c].notna().astype(int)
    return score


def signup_sort_value(df: pd.DataFrame) -> pd.Series:
    """Convert signup date parts into sortable YYYYMMDD number."""
    if not set(["signup_year", "signup_month", "signup_day"]).issubset(df.columns):
        return pd.Series(0, index=df.index)

    return (
        df["signup_year"].fillna(0).astype(int) * 10000 +
        df["signup_month"].fillna(0).astype(int) * 100 +
        df["signup_day"].fillna(0).astype(int)
    )


# ============================================================
# SMART CLEANING STEPS
# ============================================================

def standardize_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Standard cleaning of common columns."""
    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_name)

    if "city" in df.columns:
        df["city"] = df["city"].apply(clean_city)

    if "gender" in df.columns:
        df["gender"] = df["gender"].apply(clean_gender)

    if "email" in df.columns:
        df["email_raw"] = df["email"]  # keep raw for audit
        df["email"] = df["email"].apply(validate_email)

    if "age" in df.columns:
        df["age"] = parse_numeric(df["age"])

    if "income" in df.columns:
        df["income"] = parse_numeric(df["income"])

    logger.info("Standardization completed (name/city/gender/email/age/income).")
    return df


def resolve_email_name_conflicts(df: pd.DataFrame, cfg: CleaningConfig, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    SMART CONFLICT RESOLUTION:
    Instead of dropping conflicts (strict), we resolve a canonical name per email.
    We also save a conflict report.
    """
    if "email" not in df.columns or "name" not in df.columns:
        logger.warning("Skipping email-name conflict resolution (email/name missing).")
        return df, pd.DataFrame()

    temp = df[df["email"].notna() & df["name"].notna()].copy()
    if temp.empty:
        return df, pd.DataFrame()

    counts = (
        temp.groupby(["email", "name"])
        .size()
        .reset_index(name="count")
        .sort_values(["email", "count"], ascending=[True, False])
    )

    if cfg.conflict_resolution == "MOST_FREQUENT":
        canonical = (
            counts.sort_values(["email", "count"], ascending=[True, False])
            .drop_duplicates(subset=["email"], keep="first")
            .set_index("email")["name"]
            .to_dict()
        )
    else:
        # MOST_RECENT uses signup_sort as tie-breaker if available
        temp["_signup_sort"] = signup_sort_value(temp)
        temp["_quality"] = compute_quality_score(temp)

        temp = temp.sort_values(
            ["email", "_quality", "_signup_sort"],
            ascending=[True, False, False]
        )
        canonical = (
            temp.drop_duplicates("email", keep="first")
            .set_index("email")["name"]
            .to_dict()
        )
        temp.drop(columns=["_signup_sort", "_quality"], inplace=True)

    df["resolved_name"] = df["email"].map(canonical)

    df["email_name_conflict"] = False
    mask = df["email"].notna() & df["name"].notna() & df["resolved_name"].notna()
    df.loc[mask, "email_name_conflict"] = (df.loc[mask, "name"] != df.loc[mask, "resolved_name"])

    conflicts = df[df["email_name_conflict"]].copy()

    df.loc[df["resolved_name"].notna(), "name"] = df.loc[df["resolved_name"].notna(), "resolved_name"]

    logger.info(f"Email-name conflicts found: {len(conflicts)} (resolved instead of dropping).")

    df.drop(columns=["resolved_name"], inplace=True)
    return df, conflicts


def handle_identity_rules(df: pd.DataFrame, cfg: CleaningConfig, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identity rules: email is primary identifier.
    Smart approach: keep name-missing rows (optional), but always enforce email if required.
    """
    dropped = pd.DataFrame()

    if cfg.require_email:
        drop_mask = df["email"].isna() if "email" in df.columns else pd.Series(True, index=df.index)
        dropped = df[drop_mask].copy()
        df = df[~drop_mask].copy()
        logger.info(f"Dropped rows due to missing/invalid email: {len(dropped)}")

    if cfg.require_name and "name" in df.columns:
        drop_mask = df["name"].isna()
        dropped2 = df[drop_mask].copy()
        df = df[~drop_mask].copy()
        dropped = pd.concat([dropped, dropped2], ignore_index=True)
        logger.info(f"Dropped rows due to missing name (require_name=True): {len(dropped2)}")

    return df, dropped


def remove_exact_duplicates(df: pd.DataFrame, logger) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Exact duplicates removed: {before - len(df)}")
    return df


def deduplicate_people_by_email(df: pd.DataFrame, cfg: CleaningConfig, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If same email appears multiple times, keep the BEST row:
    - highest quality_score
    - newest signup date
    """
    if not cfg.keep_best_record_per_email:
        return df, pd.DataFrame()

    if "email" not in df.columns:
        logger.warning("Skipping email deduplication (email column missing).")
        return df, pd.DataFrame()

    df["_quality_score"] = compute_quality_score(df)
    df["_signup_sort"] = signup_sort_value(df)

    df = df.sort_values(
        ["email", "_quality_score", "_signup_sort"],
        ascending=[True, False, False]
    )

    duplicated_people = df[df.duplicated(subset=["email"], keep="first")].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["email"], keep="first")
    logger.info(f"Duplicate people removed (same email): {before - len(df)}")

    df.drop(columns=["_quality_score", "_signup_sort"], inplace=True, errors="ignore")
    return df, duplicated_people


def fill_missing_values(df: pd.DataFrame, cfg: CleaningConfig, logger) -> pd.DataFrame:
    
    numeric_cols = ["age", "income", "signup_year", "signup_month", "signup_day"]
    if cfg.fill_numeric == "MEDIAN":
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        logger.info("Filled numeric missing values using MEDIAN.")

    categorical_cols = ["gender", "city"]
    if cfg.fill_categorical == "MODE":
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True)[0])
        logger.info("Filled categorical missing values using MODE.")

    return df


def build_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, artifacts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create simple KPI report."""
    report = {
        "raw_rows": len(raw_df),
        "raw_cols": raw_df.shape[1],
        "clean_rows": len(clean_df),
        "clean_cols": clean_df.shape[1],
        "rows_removed": len(raw_df) - len(clean_df),
        "pct_removed": round((len(raw_df) - len(clean_df)) / max(len(raw_df), 1) * 100, 2),
    }

    for name, df in artifacts.items():
        report[f"{name}_rows"] = len(df) if df is not None else 0

    if not clean_df.empty:
        report["clean_total_missing_cells"] = int(clean_df.isna().sum().sum())

    return pd.DataFrame([report])


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(cfg: CleaningConfig):
    logger = setup_logger(cfg.log_level)

    input_path = Path(cfg.input_csv)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input CSV: {input_path}")
    raw_df = pd.read_csv(input_path)

    logger.info(f"RAW shape: {raw_df.shape}")

    df = raw_df.copy()
    
    df = normalize_missing_values(df)

    df = standardize_columns(df, logger)

    df = parse_signup_date(df, col="signup_date")

    df = remove_exact_duplicates(df, logger)

    df, email_name_conflicts = resolve_email_name_conflicts(df, cfg, logger)

    df, dropped_identity = handle_identity_rules(df, cfg, logger)

    df, dropped_email_dupes = deduplicate_people_by_email(df, cfg, logger)

    df = fill_missing_values(df, cfg, logger)

    duplicate_emails_left = df["email"].duplicated().sum() if "email" in df.columns else None
    logger.info(f"Duplicate emails left: {duplicate_emails_left}")

    conflicts_left = 0
    if "email" in df.columns and "name" in df.columns:
        mapping_check = df.groupby("email")["name"].nunique()
        conflicts_left = int((mapping_check > 1).sum())
    logger.info(f"Email->multiple names conflicts left: {conflicts_left}")

# outputs
    clean_path = out_dir / "final_cleaned_dataset_smart.csv"
    conflicts_path = out_dir / "resolved_email_name_conflicts.csv"
    dropped_identity_path = out_dir / "dropped_identity.csv"
    dropped_email_dupes_path = out_dir / "dropped_duplicate_email_rows.csv"
    report_path = out_dir / "cleaning_report.csv"

    df.to_csv(clean_path, index=False)
    email_name_conflicts.to_csv(conflicts_path, index=False)
    dropped_identity.to_csv(dropped_identity_path, index=False)
    dropped_email_dupes.to_csv(dropped_email_dupes_path, index=False)

    artifacts = {
        "resolved_email_name_conflicts": email_name_conflicts,
        "dropped_identity": dropped_identity,
        "dropped_duplicate_email_rows": dropped_email_dupes,
    }

    report_df = build_report(raw_df, df, artifacts)
    report_df.to_csv(report_path, index=False)

    logger.info("Saved outputs:")
    logger.info(f"- {clean_path}")
    logger.info(f"- {conflicts_path}")
    logger.info(f"- {dropped_identity_path}")
    logger.info(f"- {dropped_email_dupes_path}")
    logger.info(f"- {report_path}")

    return df, report_df


if __name__ == "__main__":
    cfg = CleaningConfig(
        input_csv="raw_data.csv",
        output_dir="outputs",
        require_email=True,
        require_name=False,
        drop_invalid_email=True,
        conflict_resolution="MOST_FREQUENT",
        fill_numeric="MEDIAN",
        fill_categorical="MODE",
        keep_best_record_per_email=True,
        log_level="INFO"
    )

    run_pipeline(cfg)
