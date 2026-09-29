from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

ATS_LAYER_URL = (
    "https://geospatial.alberta.ca/titan/rest/services/base/"
    "alberta_township_system/MapServer/19/query"
)
DEFAULT_PAGE_SIZE = 1000


def build_query_url(
    *,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = True,
    result_offset: int = 0,
    result_record_count: int = DEFAULT_PAGE_SIZE,
    output_format: str = "geojson",
) -> str:
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": str(return_geometry).lower(),
        "f": output_format,
        "resultOffset": result_offset,
        "resultRecordCount": result_record_count,
    }
    return f"{ATS_LAYER_URL}?{urlencode(params)}"


def fetch_geojson_page(
    *,
    where: str = "1=1",
    result_offset: int = 0,
    result_record_count: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    url = build_query_url(
        where=where,
        result_offset=result_offset,
        result_record_count=result_record_count,
    )
    with urlopen(url) as response:
        return json.load(response)


def download_quarter_sections(
    *,
    where: str = "1=1",
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    all_features: list[dict[str, Any]] = []
    offset = 0

    while True:
        payload = fetch_geojson_page(
            where=where,
            result_offset=offset,
            result_record_count=page_size,
        )
        features = payload.get("features", [])
        all_features.extend(features)

        if len(features) < page_size:
            break
        offset += page_size

    return {
        "type": "FeatureCollection",
        "features": all_features,
    }


def save_geojson(collection: dict[str, Any], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


def save_geopackage_if_available(
    collection: dict[str, Any],
    output_path: Path | str,
) -> Path | None:
    try:
        import geopandas as gpd
    except ImportError:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame.from_features(collection["features"], crs="EPSG:3400")
    frame.to_file(path, driver="GPKG")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Alberta ATS quarter sections from the provincial ArcGIS layer."
    )
    parser.add_argument(
        "--where",
        default="1=1",
        help="ArcGIS SQL where clause, e.g. \"M=4 AND RGE=25 AND TWP=50\"",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Number of records to request per page",
    )
    parser.add_argument(
        "--geojson-out",
        type=Path,
        default=Path("data/raw/alberta_quarter_sections.geojson"),
        help="Output path for the merged GeoJSON",
    )
    parser.add_argument(
        "--gpkg-out",
        type=Path,
        default=Path("data/raw/alberta_quarter_sections.gpkg"),
        help="Optional output path for a GeoPackage if geopandas is installed",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    collection = download_quarter_sections(
        where=args.where,
        page_size=args.page_size,
    )
    geojson_path = save_geojson(collection, args.geojson_out)
    print(f"Saved GeoJSON: {geojson_path}")
    print(f"Features: {len(collection['features'])}")

    gpkg_path = save_geopackage_if_available(collection, args.gpkg_out)
    if gpkg_path is None:
        print("GeoPackage not written: install geopandas to enable GPKG output.")
    else:
        print(f"Saved GeoPackage: {gpkg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
