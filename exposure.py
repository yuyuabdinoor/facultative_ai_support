from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os
import re
import pandas as pd

def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip()).lower()

def _find_country_dirs(gem_repo_path: Path, country_name: str) -> List[Path]:
    """Return candidate directories matching the country name (case-insensitive, underscores/spaces tolerant)."""
    norm = _normalize_name(country_name)
    candidates = []
    for root, dirs, files in os.walk(gem_repo_path):
        for d in dirs:
            dnorm = d.replace("-", "_").lower()
            if dnorm == norm or norm in dnorm or dnorm in norm:
                candidates.append(Path(root) / d)
    # also check top-level region folders that directly contain the name
    return list(dict.fromkeys(candidates))  # preserve order, deduplicate

def _score_and_load_csvs(country_dir: Path) -> List[Tuple[Path, pd.DataFrame]]:
    """
    Load CSV-like files that look promising (summary/exposure/counts).
    Returns list of (path, dataframe).
    """
    hits = []
    csv_patterns = re.compile(r".*(summary|exposure|counts|adm1|national).*\.csv$", re.I)
    for p in country_dir.glob("**/*"):
        if p.is_file() and csv_patterns.match(p.name):
            try:
                df = pd.read_csv(p, low_memory=False)
                hits.append((p, df))
            except Exception:
                # try with alternative separators or encoding
                try:
                    df = pd.read_csv(p, sep=";", low_memory=False, encoding="utf-8")
                    hits.append((p, df))
                except Exception:
                    continue
    return hits

def _extract_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Given a dataframe, attempt to extract total replacement cost and building counts and group breakdowns."""
    cols = [c.upper() for c in df.columns]
    col_map = {c.upper(): c for c in df.columns}

    metrics = {
        "total_replacement_cost_usd": None,
        "total_building_count": None,
        "total_area_sqm": None,
        "occupancy_breakdown": None,
        "taxonomy_breakdown": None,
    }

    # Replacement cost column candidates
    repl_candidates = [k for k in cols if "REPL" in k or "REPLACE" in k or "REPL_COST" in k or "TOTAL_REPL" in k]
    if repl_candidates:
        c = col_map[repl_candidates[0]]
        try:
            metrics["total_replacement_cost_usd"] = float(pd.to_numeric(df[c], errors="coerce").sum(skipna=True))
        except Exception:
            metrics["total_replacement_cost_usd"] = None

    # Building count candidates
    count_candidates = [k for k in cols if "BUILD" in k or "COUNT" in k and ("BUILD" in k or "ASSET" in k or "ESTABLISH" in k)]
    # fallback: any column that looks like COUNT or ASSET
    if not count_candidates:
        count_candidates = [k for k in cols if re.search(r"(^|_)count($|_)", k.lower()) or "ASSET" in k or "BUILDING" in k]
    if count_candidates:
        c = col_map[count_candidates[0]]
        try:
            metrics["total_building_count"] = int(pd.to_numeric(df[c], errors="coerce").sum(skipna=True))
        except Exception:
            metrics["total_building_count"] = None

    # Total area
    area_candidates = [k for k in cols if "AREA" in k and ("SQM" in k or "SQM" in k.upper() or "SQUARE" in k)]
    if area_candidates:
        c = col_map[area_candidates[0]]
        try:
            metrics["total_area_sqm"] = float(pd.to_numeric(df[c], errors="coerce").sum(skipna=True))
        except Exception:
            metrics["total_area_sqm"] = None

    # Occupancy/taxonomy breakdown if there are category columns
    # If columns include OCCUPANCY or TAXONOMY or "OCCUPANCY_TYPE", try grouping
    category_cols = [col_map[c] for c in cols if "OCCUPANCY" in c or "TAXONOMY" in c or "OCCUPANT" in c or "TYPE" == c]
    # More flexible: find columns likely categorical (string dtype, few unique values)
    if not category_cols:
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique(dropna=True) <= 50 and df[c].nunique(dropna=True) > 1:
                category_cols.append(c)
    # If we have at least one category column plus a numeric column to aggregate:
    numeric_col_for_agg = None
    if repl_candidates:
        numeric_col_for_agg = col_map[repl_candidates[0]]
    else:
        # try a generic numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_col_for_agg = c
                break

    if category_cols and numeric_col_for_agg:
        cat = category_cols[0]
        try:
            grp = df.groupby(cat)[numeric_col_for_agg].sum(min_count=1).dropna()
            metrics["occupancy_breakdown"] = grp.to_dict()
        except Exception:
            metrics["occupancy_breakdown"] = None

    # taxonomy breakdown by summing 'BUILDING' counts per taxonomy column if present
    tax_col = None
    for c in df.columns:
        if "TAXONOMY" in c.upper() or "TAXON" in c.upper():
            tax_col = c
            break
    if tax_col:
        # find a count-like column
        cnt_col = None
        for c in df.columns:
            if re.search(r"count|asset|establishment|number", c, re.I):
                cnt_col = c
                break
        if cnt_col:
            try:
                metrics["taxonomy_breakdown"] = df.groupby(tax_col)[cnt_col].sum(min_count=1).dropna().to_dict()
            except Exception:
                metrics["taxonomy_breakdown"] = None

    return metrics

def gem_exposure_summary(country_code_or_name: str, gem_repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Return a best-effort exposure summary for the requested country.
    - country_code_or_name: name like "Kenya" or "United Kingdom" (underscores/spaces tolerated)
    - gem_repo_path: path to the extracted 'global_exposure_model' repo folder (must exist)
    Returns a dict with:
      { 'country_dir': Path or None,
        'files_searched': [...],
        'files_used': [...],
        'summary': { total_replacement_cost_usd, total_building_count, ... },
        'note': explanatory message }
    """
    if gem_repo_path is None:
        raise ValueError("gem_repo_path must be provided and point to the extracted global_exposure_model repository root.")

    gem_repo_path = Path(gem_repo_path)
    if not gem_repo_path.exists():
        raise FileNotFoundError(f"gem_repo_path not found: {gem_repo_path}")

    candidates = _find_country_dirs(gem_repo_path, country_code_or_name)
    if not candidates:
        # try a looser search over filenames (maybe the folder name differs)
        loose_matches = []
        norm = _normalize_name(country_code_or_name)
        for p in gem_repo_path.glob("**/*"):
            if p.is_dir() and norm in p.name.lower().replace(" ", "_"):
                loose_matches.append(p)
        candidates = loose_matches

    if not candidates:
        return {
            "country_dir": None,
            "files_searched": [],
            "files_used": [],
            "summary": {},
            "note": f"No folder found for '{country_code_or_name}' under {str(gem_repo_path)}. Try using the official country folder name (check repository)."
        }

    # Prefer the first candidate (usually exact match)
    country_dir = candidates[0]

    csv_hits = _score_and_load_csvs(country_dir)
    files_used = []
    aggregate_metrics = {
        "total_replacement_cost_usd": None,
        "total_building_count": None,
        "total_area_sqm": None,
        "occupancy_breakdown": {},
        "taxonomy_breakdown": {},
    }

    for p, df in csv_hits:
        metrics = _extract_metrics_from_df(df)
        # accumulate totals if present
        if metrics.get("total_replacement_cost_usd") is not None:
            if aggregate_metrics["total_replacement_cost_usd"] is None:
                aggregate_metrics["total_replacement_cost_usd"] = 0.0
            aggregate_metrics["total_replacement_cost_usd"] += metrics["total_replacement_cost_usd"]
        if metrics.get("total_building_count") is not None:
            if aggregate_metrics["total_building_count"] is None:
                aggregate_metrics["total_building_count"] = 0
            aggregate_metrics["total_building_count"] += metrics["total_building_count"]
        if metrics.get("total_area_sqm") is not None:
            if aggregate_metrics["total_area_sqm"] is None:
                aggregate_metrics["total_area_sqm"] = 0.0
            aggregate_metrics["total_area_sqm"] += metrics["total_area_sqm"]

        # merge breakdowns (add numeric values)
        for k in ("occupancy_breakdown", "taxonomy_breakdown"):
            b = metrics.get(k)
            if b:
                agg = aggregate_metrics[k] or {}
                for cat, val in (b.items() if isinstance(b, dict) else []):
                    try:
                        agg[cat] = agg.get(cat, 0) + float(val)
                    except Exception:
                        agg[cat] = agg.get(cat, 0)
                aggregate_metrics[k] = agg

        files_used.append(str(p))

    note = "Summary assembled from available CSVs in the country folder."
    if not files_used:
        note = "No summary/exposure CSVs found in country folder; try downloading the release zip for your required version or check folder contents."

    return {
        "country_dir": str(country_dir),
        "files_searched": [str(p) for p, _ in csv_hits],
        "files_used": files_used,
        "summary": aggregate_metrics,
        "note": note,
    }
for i in ['Kenya', 'Indonesia', 'Tanzania', 'Ghana', 'Nigeria']:
    gem_root = 'global_exposure_model'
    res = gem_exposure_summary(i, gem_root)
    print(res["note"])
    print("Country folder:", res["country_dir"])
    print("Files used:", res["files_used"])
    print("Summary:", res["summary"])
