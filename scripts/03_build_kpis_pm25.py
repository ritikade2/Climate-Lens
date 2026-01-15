from __future__ import annotations

import sys
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# repo imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from trust_bi.common.settings import Settings  # noqa: E402
from trust_bi.warehouse.duckdb_io import DuckDBClient  # noqa: E402

# -----------------------------
# Config / constants
# -----------------------------
KPI_ID = "pm25_monthly_mean"
SOURCE_ID = "cams"
EXPECTED_VAR = "particulate_matter_2.5um"
DEFAULT_UNIT = "ug/m3"


# -------------------------
# Helpers: choose raw file
# -------------------------
def pick_latest_cams_raw(cams_dir: Path) -> Path:
    """
    Picks latest CAMS raw artifact from data/raw/cams.
    Prefer *.nc.zip; fallback to *.zip then *.nc.
    """
    if not cams_dir.exists():
        raise RuntimeError(f"Missing folder: {cams_dir}")

    candidates = []
    for pat in ["*.nc.zip", "*.zip", "*.nc"]:
        candidates.extend([p for p in cams_dir.glob(pat) if p.is_file()])

    candidates = [p for p in candidates if "unzipped" not in str(p)]

    if not candidates:
        raise RuntimeError(
            f"No CAMS raw files found in {cams_dir}. Expected *.nc.zip or *.nc.\n"
            "Run: python3 scripts/02_ingest_cams.py"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


# -------------------------
# helpers: prep file to open
# -------------------------
def prepare_cams_file(raw_path: Path, out_dir: Path) -> Tuple[Path, str]:
    """
    Never mutates raw_path.

    Returns (path_to_open, kind) where kind is one of:
      - 'netcdf'
      - 'grib'
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: true zip container
    if zipfile.is_zipfile(raw_path):
        with zipfile.ZipFile(raw_path, "r") as z:
            ncs = [n for n in z.namelist() if n.endswith(".nc")]
            if ncs:
                member = ncs[0]
                out_path = out_dir / Path(member).name
                z.extract(member, path=out_dir)
                extracted = out_dir / member
                if extracted != out_path:
                    extracted.replace(out_path)
                return out_path, "netcdf"

            gribs = [n for n in z.namelist() if n.lower().endswith((".grib", ".grb", ".grib2"))]
            if gribs:
                member = gribs[0]
                out_path = out_dir / Path(member).name
                z.extract(member, path=out_dir)
                extracted = out_dir / member
                if extracted != out_path:
                    extracted.replace(out_path)
                return out_path, "grib"

            raise RuntimeError(f"Zip contains neither .nc nor .grib: {raw_path}")

    # Case 2: not a zip container, sniff magic bytes
    with raw_path.open("rb") as f:
        head = f.read(16)

    # GRIB
    if head[:4] == b"GRIB":
        out_path = out_dir / (raw_path.stem + ".grib")
        shutil.copy2(raw_path, out_path)
        return out_path, "grib"

    # NetCDF classic or NetCDF4/HDF5
    if head[:3] == b"CDF" or head[:8] == b"\x89HDF\r\n\x1a\n":
        out_path = out_dir / (raw_path.stem + ".nc")
        shutil.copy2(raw_path, out_path)
        return out_path, "netcdf"

    # Plain .nc
    if raw_path.suffix == ".nc":
        return raw_path, "netcdf"

    raise RuntimeError(f"Unknown CAMS file type: {raw_path} header={head!r}")


# -------------------------
# helpers: xarray coords/vars
# -------------------------
def infer_coord_names(ds: xr.Dataset) -> tuple[str, str, str]:
    lat = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    t = "time" if "time" in ds.coords else ("valid_time" if "valid_time" in ds.coords else None)
    if lat is None or lon is None or t is None:
        raise RuntimeError(f"Could not infer coords. coords={list(ds.coords)} dims={list(ds.dims)}")
    return lat, lon, t


def normalize_lon_to_180(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    """
    Ensures longitudes are in [-180, 180] to match country polygons CRS.
    """
    lon = da[lon_name].values
    if np.nanmax(lon) > 180:
        da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)}).sortby(lon_name)
    return da


def load_world_countries(repo_root: Path):
    """
    Loads Natural Earth Admin-0 country polygons (110m) from local disk.
    Auto-downloads the zip once if missing.
    """
    import geopandas as gpd
    import subprocess

    ref_dir = repo_root / "data" / "reference" / "natural_earth"
    ref_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ref_dir / "ne_110m_admin_0_countries.zip"
    shp_inside_zip = "ne_110m_admin_0_countries.shp"

    if not zip_path.exists():
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(zip_path), url],
            check=True,
        )

    world = gpd.read_file(f"zip://{zip_path}!{shp_inside_zip}")
    world = world.rename(columns={"ISO_A3": "country_iso3", "NAME": "geo_name"})
    world = world[world["country_iso3"].notna() & (world["country_iso3"] != "-99")].copy()
    world = world[["country_iso3", "geo_name", "geometry"]].to_crs(4326)
    return world


def build_dim_time_from_month_starts(dates: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dates)
    out = pd.DataFrame(
        {
            "date": dt.dt.date,
            "year": dt.dt.year.astype(int),
            "month": dt.dt.month.astype(int),
            "day": dt.dt.day.astype(int),
            "month_start": pd.to_datetime(dict(year=dt.dt.year, month=dt.dt.month, day=1)).dt.date,
            "is_complete_period": True,
        }
    )
    return out.drop_duplicates(subset=["date"])


# -------------------------
# main
# -------------------------
def main() -> None:
    s = Settings.load(".env")
    db = DuckDBClient(s.duckdb_path)

    cams_dir = s.raw_dir / "cams"
    raw_path = pick_latest_cams_raw(cams_dir)
    print(f"Using CAMS raw: {raw_path}")

    data_path, kind = prepare_cams_file(raw_path, cams_dir / "unzipped")
    print(f"Using CAMS file to open: {data_path}")
    print(f"Detected file format: {kind}")

    if kind == "grib":
        ds = xr.open_dataset(data_path, engine="cfgrib")
    else:
        ds = xr.open_dataset(data_path, engine="netcdf4")

    # pick variable
    if EXPECTED_VAR in ds.data_vars:
        var = EXPECTED_VAR
    else:
        var = list(ds.data_vars)[0]
        print(f"Expected '{EXPECTED_VAR}' not found. Using '{var}' instead.")

    lat_name, lon_name, t_name = infer_coord_names(ds)
    da = ds[var]
    da = normalize_lon_to_180(da, lon_name)

    # Ensure dims are (time, lat, lon)
    if da.dims[0] != t_name:
        da = da.transpose(t_name, lat_name, lon_name)

    # unit best-effort
    unit = str(da.attrs.get("units") or DEFAULT_UNIT)
    raw_unit = str(da.attrs.get("units") or "").strip().lower()

    # CAMS pm2p5 is usually kg/m3 -> convert to ug/m3
    if raw_unit in {"kg m-3", "kg/m3", "kg m**-3", "kg m^-3"} or raw_unit == "":
        scale = 1e9
        unit = "ug/m3"
    else:
        scale = 1.0

    print(f"PM2.5 raw units='{raw_unit or 'UNKNOWN'}' -> storing as '{unit}' (scale={scale})")

    # Build lat/lon grid
    lats = da[lat_name].values
    lons = da[lon_name].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    pts = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel()})

    # geopandas stack required
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError(
            "Missing geopandas stack required for country mapping.\n"
            "Install: python3 -m pip install geopandas shapely pyproj fiona rtree\n"
            f"Original error: {repr(e)}"
        )

    world = load_world_countries(REPO_ROOT)
    gpts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs=4326)

    joined = gpd.sjoin(gpts, world, how="left", predicate="within")
    iso3_flat = joined["country_iso3"].to_numpy()
    iso3_grid = iso3_flat.reshape(lat2d.shape)

    # Computing monthly mean PM2.5 by country (simple mean across grid points)
    times = pd.to_datetime(da[t_name].values)
    values = da.values  # time x lat x lon

    rows: list[dict[str, Any]] = []
    countries = np.unique(iso3_grid[~pd.isna(iso3_grid)])

    created_at = datetime.utcnow()

    for ti, ts in enumerate(times):
        grid = values[ti, :, :]
        for c in countries:
            mask = (iso3_grid == c)
            v = np.nanmean(grid[mask]) * scale
            if np.isnan(v):
                continue
            rows.append(
                {
                    "kpi_id": KPI_ID,
                    "geo_id": f"ISO3:{c}",
                    "date": ts.date().replace(day=1),
                    "source_id": SOURCE_ID,
                    "value": float(v),
                    "value_se": None,
                    "unit": unit,
                    "n_obs": int(np.sum(mask)),
                    "coverage_pct": None,
                    "data_quality_flag": "pass",
                    "created_at": created_at,
                }
            )

    fact = pd.DataFrame(rows)
    if fact.empty:
        raise RuntimeError("No KPI rows produced. Country mapping failed or values are NaN.")

    # Upsert KPI facts
    db.upsert_df(
        fact,
        "fact_kpi_timeseries",
        pk_cols=["kpi_id", "geo_id", "date", "source_id"],
    )

    # Upsert dim_time for these months (helps later scripts)
    dim_time = build_dim_time_from_month_starts(pd.Series(fact["date"]))
    db.upsert_df(dim_time, "dim_time", pk_cols=["date"], created_at_col=None)

    # ---- dim_geo upsert without overwriting lat/lon ----
    # Building candidate dim_geo rows
    dim_geo_new = (
        fact[["geo_id"]]
        .drop_duplicates()
        .assign(
            geo_level="country",
            country_iso3=lambda d: d["geo_id"].str.replace("ISO3:", "", regex=False),
            geo_name=lambda d: d["geo_id"].str.replace("ISO3:", "", regex=False),
            lat=None,
            lon=None,
        )
    )

    # Pull existing lat/lon and preserve them
    existing = db.query_df("SELECT geo_id, lat, lon FROM dim_geo")
    merged = dim_geo_new.merge(existing, on="geo_id", how="left", suffixes=("", "_old"))

    merged["lat"] = merged["lat_old"].combine_first(merged["lat"])
    merged["lon"] = merged["lon_old"].combine_first(merged["lon"])
    merged = merged.drop(columns=[c for c in ["lat_old", "lon_old"] if c in merged.columns])

    db.upsert_df(merged, "dim_geo", pk_cols=["geo_id"], created_at_col=None)
    # -----------------------------------------------------------

    print(f"Wrote {len(fact):,} country-month PM2.5 rows to fact_kpi_timeseries (kpi_id={KPI_ID}).")


if __name__ == "__main__":
    main()
