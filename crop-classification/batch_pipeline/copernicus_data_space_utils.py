from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class SceneItem:
    scene_id: str
    datetime: str
    cloud_cover: float | None
    collection: str
    bbox: list[float]
    geometry: dict[str, Any] | None
    product_name: str


def iso_date_range(year: int, start_month: int, start_day: int, end_month: int, end_day: int) -> tuple[date, date]:
    return date(year, start_month, start_day), date(year, end_month, end_day)


def weekly_ranges(start: date, end: date) -> list[tuple[int, date, date]]:
    ranges: list[tuple[int, date, date]] = []
    current = start
    index = 1
    while current < end:
        next_date = min(current + timedelta(days=7), end)
        ranges.append((index, current, next_date))
        current = next_date
        index += 1
    return ranges


def bbox_union(features: list[dict[str, Any]]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        geometry = feature["geometry"]
        coords = geometry["coordinates"]
        polygons = [coords] if geometry["type"] == "Polygon" else coords
        for polygon in polygons:
            for ring in polygon:
                for lon, lat in ring:
                    xs.append(float(lon))
                    ys.append(float(lat))
    return [min(xs), min(ys), max(xs), max(ys)]


class CDSEClient:
    def __init__(
        self,
        *,
        username: str,
        password: str,
        token_url: str,
        client_id: str,
        stac_search_endpoint: str,
        odata_products_endpoint: str,
        odata_download_base: str,
    ) -> None:
        self.username = username
        self.password = password
        self.token_url = token_url
        self.client_id = client_id
        self.stac_search_endpoint = stac_search_endpoint
        self.odata_products_endpoint = odata_products_endpoint
        self.odata_download_base = odata_download_base
        self._access_token: str | None = None

    def access_token(self) -> str:
        if self._access_token is None:
            response = requests.post(
                self.token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                    "client_id": self.client_id,
                },
                timeout=60,
            )
            response.raise_for_status()
            self._access_token = response.json()["access_token"]
        return self._access_token

    def search_sentinel2_l2a(
        self,
        *,
        bbox: list[float],
        start_iso: str,
        end_iso: str,
        max_cloud_coverage: int,
        limit: int = 100,
    ) -> list[SceneItem]:
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_iso}T00:00:00Z/{end_iso}T00:00:00Z",
            "limit": limit,
            "filter-lang": "cql2-json",
            "filter": {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, max_cloud_coverage],
            },
        }
        response = requests.post(self.stac_search_endpoint, json=payload, timeout=120)
        response.raise_for_status()
        items = response.json().get("features", [])
        scene_items: list[SceneItem] = []
        for item in items:
            props = item.get("properties", {}) or {}
            product_name = str(
                props.get("productIdentifier")
                or item.get("id")
                or ""
            ).split("/")[-1]
            scene_items.append(
                SceneItem(
                    scene_id=str(item.get("id")),
                    datetime=str(props.get("datetime")),
                    cloud_cover=float(props["eo:cloud_cover"]) if props.get("eo:cloud_cover") is not None else None,
                    collection=str(item.get("collection")),
                    bbox=[float(x) for x in item.get("bbox", [])],
                    geometry=item.get("geometry"),
                    product_name=product_name,
                )
            )
        return scene_items

    def resolve_product_uuid(self, product_name: str) -> str | None:
        params = {
            "$filter": f"Name eq '{product_name}'",
            "$top": "1",
            "$select": "Id,Name",
        }
        response = requests.get(self.odata_products_endpoint, params=params, timeout=120)
        response.raise_for_status()
        values = response.json().get("value", [])
        if not values:
            return None
        return str(values[0]["Id"])

    def product_download_url(self, product_uuid: str) -> str:
        return f"{self.odata_download_base}({product_uuid})/$value"


def cdse_client_from_env(config: dict[str, Any]) -> CDSEClient:
    import os

    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD.")
    source_cfg = config["sentinel2_source"]
    return CDSEClient(
        username=username,
        password=password,
        token_url=source_cfg["oauth_token_url"],
        client_id=source_cfg["oauth_client_id"],
        stac_search_endpoint=source_cfg["stac_search_endpoint"],
        odata_products_endpoint=source_cfg["odata_products_endpoint"],
        odata_download_base=source_cfg["odata_download_base"],
    )

