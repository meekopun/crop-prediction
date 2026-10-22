"""Export monthly ERA5-Land weather for the harvest quarter sections.

ERA5-Land cells are about 11 km across, so neighbouring quarters share nearly
the same values: this is mainly a season-level signal. For each pilot-eligible
``(pid, year)`` and each month April-September it records total precipitation
(mm), mean air temperature and mean daily maximum (°C), growing degree days above
5 °C from daily (max + min) / 2, and mean topsoil water (m³/m³).

Earth Engine is imported lazily; only the export step needs it and a login.
"""

from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path

from .harvest import DEFAULT_OUTPUT_DIR
from .harvest_features import DEFAULT_BOUNDARIES_PATH, DEFAULT_MATCHES_PATH, DEFAULT_YEARS, load_feature_units

DEFAULT_WEATHER_DIR = DEFAULT_OUTPUT_DIR / "weather"
WEATHER_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
WEATHER_MONTHS = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep"}
WEATHER_VARIABLES = ("precip", "tmean", "tmax", "gdd5", "soilwater")
KELVIN = 273.15
GDD_BASE_C = 5.0
# Fine enough that every quarter contains sample points of its ERA5-Land cell.
SAMPLE_SCALE_M = 100


def weather_columns() -> list[str]:
    return [f"w_{variable}_{month}" for month in WEATHER_MONTHS.values() for variable in WEATHER_VARIABLES]


def _weather_image(ee, aoi, year: int):
    image = None
    for month, name in WEATHER_MONTHS.items():
        start = ee.Date.fromYMD(year, month, 1)
        daily = ee.ImageCollection(WEATHER_COLLECTION).filterBounds(aoi).filterDate(start, start.advance(1, "month"))
        gdd = daily.map(lambda day: day.select("temperature_2m_max").add(day.select("temperature_2m_min"))
                        .divide(2).subtract(KELVIN + GDD_BASE_C).max(0)).sum()
        monthly = ee.Image.cat([
            daily.select("total_precipitation_sum").sum().multiply(1000).rename(f"w_precip_{name}"),
            daily.select("temperature_2m").mean().subtract(KELVIN).rename(f"w_tmean_{name}"),
            daily.select("temperature_2m_max").mean().subtract(KELVIN).rename(f"w_tmax_{name}"),
            gdd.rename(f"w_gdd5_{name}"),
            daily.select("volumetric_soil_water_layer_1").mean().rename(f"w_soilwater_{name}"),
        ]).toFloat()
        image = monthly if image is None else image.addBands(monthly)
    return image


def extract_weather_year(units, year: int) -> list[dict]:
    """Reduce one year's monthly weather over its units and return one row per unit."""
    import ee

    year_units = [unit for unit in units if unit.year == year]
    if not year_units:
        return []
    collection = ee.FeatureCollection([ee.Feature(unit.geometry, {"pid": unit.pid}) for unit in year_units])
    image = _weather_image(ee, collection.geometry().bounds(), year)
    reduced = image.reduceRegions(collection=collection, reducer=ee.Reducer.mean(), scale=SAMPLE_SCALE_M)
    values = {feature["properties"]["pid"]: feature["properties"] for feature in reduced.getInfo()["features"]}
    return [{"pid": unit.pid, "year": unit.year,
             **{column: values[unit.pid].get(column) for column in weather_columns()}}
            for unit in year_units]


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "year"] + weather_columns(), restval="")
        writer.writeheader()
        writer.writerows(rows)


def export_harvest_weather(matches_path: Path | str = DEFAULT_MATCHES_PATH,
                           boundaries_path: Path | str = DEFAULT_BOUNDARIES_PATH,
                           output_dir: Path | str = DEFAULT_WEATHER_DIR,
                           *, years: tuple[int, ...] = DEFAULT_YEARS, project: str | None = None,
                           overwrite: bool = False, progress=print) -> dict:
    """Export each year to its own CSV (skipping finished years), then combine them."""
    import ee

    units = load_feature_units(matches_path, boundaries_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=project)
    combined = []
    for year in years:
        path = destination / f"harvest_weather_{year}.csv"
        if path.exists() and not overwrite:
            progress(f"{year}: kept existing {path.name}")
        else:
            rows = extract_weather_year(units, year)
            _write_rows(path, rows)
            progress(f"{year}: wrote {len(rows)} quarter sections to {path.name}")
        with path.open(encoding="utf-8", newline="") as handle:
            combined.extend(csv.DictReader(handle))
    _write_rows(destination / "harvest_weather.csv", combined)
    manifest = {
        "schema_version": 1,
        "created_on": date.today().isoformat(),
        "collection": WEATHER_COLLECTION,
        "years": list(years),
        "unit_count": len(combined),
        "months": list(WEATHER_MONTHS.values()),
        "sample_scale_m": SAMPLE_SCALE_M,
        "columns": {
            "w_precip_<month>": "total precipitation, mm (sum of daily total_precipitation_sum x 1000)",
            "w_tmean_<month>": "mean of daily mean 2 m air temperature, °C",
            "w_tmax_<month>": "mean of daily maximum 2 m air temperature, °C",
            "w_gdd5_<month>": f"growing degree days, sum of max(0, (tmax + tmin) / 2 - {GDD_BASE_C} °C)",
            "w_soilwater_<month>": "mean volumetric soil water, layer 1 (0-7 cm), m³/m³",
        },
    }
    (destination / "harvest_weather_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                              encoding="utf-8")
    return manifest
