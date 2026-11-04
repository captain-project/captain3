"""Unit tests for the KML/KMZ export of protection maps."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from captain.utils.export import KmlClass, _kml_color, geotiff_to_kml, raster_to_kml

NS = {"k": "http://www.opengis.net/kml/2.2"}
CLASSES = [KmlClass(1, "Existing", "#52514e"), KmlClass(2, "New", "#1baf7a", opacity=0.5)]
# 0.01 degree cells with the top-left corner at 43W, 22S
TRANSFORM = from_origin(-43.0, -22.0, 0.01, 0.01)


def _grid():
    g = np.zeros((6, 6), dtype=np.float32)
    g[0:2, 0:2] = 1  # one existing block
    g[3:6, 3:6] = 2  # a 3x3 new block with a hole
    g[4, 4] = 0
    g[0, 5] = 2  # an isolated new cell
    g[5, 0] = np.nan  # outside the AOI
    return g


def _folders(path):
    root = ET.parse(path).getroot()
    return {f.find("k:name", NS).text: f for f in root.iter(f"{{{NS['k']}}}Folder")}


def test_polygons_per_class(tmp_path):
    out = raster_to_kml(_grid(), TRANSFORM, "EPSG:4326", tmp_path / "plan.kml", CLASSES)
    folders = _folders(out)
    assert len(folders["Existing"].findall("k:Placemark", NS)) == 1
    assert len(folders["New"].findall("k:Placemark", NS)) == 2


def test_hole_and_coordinates(tmp_path):
    out = raster_to_kml(_grid(), TRANSFORM, "EPSG:4326", tmp_path / "plan.kml", CLASSES)
    new = _folders(out)["New"]
    assert len(new.findall(".//k:innerBoundaryIs", NS)) == 1
    coords = [
        tuple(map(float, xy.split(",")))
        for c in new.iter(f"{{{NS['k']}}}coordinates")
        for xy in c.text.split()
    ]
    lon, lat = np.array(coords).T
    assert lon.min() == pytest.approx(-42.97) and lon.max() == pytest.approx(-42.94)
    assert lat.min() == pytest.approx(-22.06) and lat.max() == pytest.approx(-22.0)


def test_kmz_is_zipped_kml(tmp_path):
    out = raster_to_kml(_grid(), TRANSFORM, "EPSG:4326", tmp_path / "plan.kmz", CLASSES)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["doc.kml"]
        assert b"<Folder><name>New</name>" in z.read("doc.kml")


def test_reprojects_to_wgs84(tmp_path):
    """A UTM grid ends up in longitude/latitude."""
    g = np.full((2, 2), 2.0, dtype=np.float32)
    utm = from_origin(700_000, 7_560_000, 300, 300)  # near Rio de Janeiro, UTM 23S
    out = raster_to_kml(g, utm, "EPSG:32723", tmp_path / "utm.kml", CLASSES)
    text = _folders(out)["New"].find(".//k:coordinates", NS).text
    lon, lat = map(float, text.split()[0].split(","))
    assert -44 < lon < -42 and -23 < lat < -21


def test_geotiff_roundtrip(tmp_path):
    tif = tmp_path / "plan.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=TRANSFORM,
        nodata=np.nan,
    ) as dst:
        dst.write(_grid(), 1)
    out = geotiff_to_kml(tif, tmp_path / "plan.kml", CLASSES, name="Run E")
    assert ET.parse(out).getroot().find("k:Document/k:name", NS).text == "Run E"
    assert len(_folders(out)["New"].findall("k:Placemark", NS)) == 2


def test_kml_color():
    assert _kml_color("#1baf7a", 0.5) == "807aaf1b"


@pytest.mark.parametrize("kwargs", [{"color": "green"}, {"color": "#1baf7a", "opacity": 2}])
def test_invalid_class(kwargs):
    with pytest.raises(ValueError):
        KmlClass(2, "New", **kwargs)


def test_duplicate_class_values(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        raster_to_kml(
            _grid(),
            TRANSFORM,
            "EPSG:4326",
            tmp_path / "x.kml",
            [CLASSES[0], KmlClass(1, "Again", "#000000")],
        )
