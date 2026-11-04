"""Export protection maps as KML/KMZ for Google Earth and other GIS viewers.

Cells of each class (e.g. existing vs. new protected areas) are merged into polygons
(rasterio.features.shapes), reprojected to WGS84 longitude/latitude and written as KML
with one folder and style per class. A ``.kmz`` output is the same KML, zipped.

Example:
    >>> from captain.utils.export import KmlClass, geotiff_to_kml
    >>> geotiff_to_kml("new_protection.tif", "new_protection.kmz", [
    ...     KmlClass(1, "Existing protected areas", "#52514e"),
    ...     KmlClass(2, "New protection", "#1baf7a"),
    ... ])
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.warp import transform_geom

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affine import Affine

logger = logging.getLogger(__name__)

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class KmlClass:
    """One map class exported as a KML folder.

    Attributes:
        value: Grid value of the class (e.g. 2 = new protection).
        name: Folder and polygon name shown in Google Earth.
        color: Fill and outline colour as "#rrggbb".
        opacity: Fill opacity in [0, 1] (outlines are opaque).
    """

    value: float
    name: str
    color: str
    opacity: float = 0.6

    def __post_init__(self) -> None:
        if not _HEX_COLOR.match(self.color):
            raise ValueError(f"color must be '#rrggbb', got {self.color!r}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {self.opacity}")


def _kml_color(color: str, opacity: float) -> str:
    """'#rrggbb' + opacity -> KML 'aabbggrr'."""
    r, g, b = color[1:3], color[3:5], color[5:7]
    return f"{round(opacity * 255):02x}{b}{g}{r}".lower()


def _ring(coords: Sequence[Sequence[float]], decimals: int) -> str:
    return " ".join(f"{x:.{decimals}f},{y:.{decimals}f}" for x, y in coords)


def _polygon_kml(geom: dict, decimals: int) -> str:
    rings = geom["coordinates"]
    inner = "".join(
        f"<innerBoundaryIs><LinearRing><coordinates>{_ring(r, decimals)}</coordinates>"
        "</LinearRing></innerBoundaryIs>"
        for r in rings[1:]
    )
    return (
        "<Polygon><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>"
        f"<outerBoundaryIs><LinearRing><coordinates>{_ring(rings[0], decimals)}</coordinates>"
        f"</LinearRing></outerBoundaryIs>{inner}</Polygon>"
    )


def raster_to_kml(
    grid: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    outfile: str | Path,
    classes: Sequence[KmlClass],
    name: str = "CAPTAIN protection plan",
    description: str | None = None,
    decimals: int = 6,
) -> Path:
    """Write the cells of each class as KML polygons.

    Neighbouring cells of the same class (4-connectivity) are merged into one polygon;
    holes are kept. Coordinates are reprojected from ``crs`` to WGS84.

    Args:
        grid: 2D array of class values (NaN or values not in ``classes`` are skipped).
        transform: Affine transform of the grid (e.g. ``rasterio`` dataset.transform).
        crs: Coordinate reference system of the grid.
        outfile: Output path; ``.kmz`` writes a zipped KML, anything else plain KML.
        classes: The classes to export, one KML folder each, in drawing order.
        name: Document name shown in Google Earth.
        description: Optional document description.
        decimals: Decimal places of the coordinates (6 = ~0.1 m).

    Returns:
        Path of the written file.

    Raises:
        ValueError: If ``grid`` is not 2D or ``classes`` is empty or has duplicate values.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {grid.shape}")
    values = [c.value for c in classes]
    if not values or len(set(values)) != len(values):
        raise ValueError("classes must be non-empty with distinct values")
    outfile = Path(outfile)
    reproject = CRS.from_user_input(crs) != CRS.from_user_input(WGS84)

    styles: list[str] = []
    folders: list[str] = []
    for i, cls in enumerate(classes):
        style_id = f"class{i}"
        styles.append(
            f'<Style id="{style_id}"><LineStyle><color>{_kml_color(cls.color, 1.0)}</color>'
            "<width>1</width></LineStyle>"
            f"<PolyStyle><color>{_kml_color(cls.color, cls.opacity)}</color></PolyStyle></Style>"
        )
        mask = grid == cls.value
        placemarks: list[str] = []
        for geom, _ in shapes(
            mask.astype(np.uint8), mask=mask, connectivity=4, transform=transform
        ):
            if reproject:
                geom = transform_geom(crs, WGS84, geom)
            placemarks.append(
                f"<Placemark><name>{escape(cls.name)} {len(placemarks) + 1}</name>"
                f"<styleUrl>#{style_id}</styleUrl>{_polygon_kml(geom, decimals)}</Placemark>"
            )
        logger.info("%s: %d cells in %d polygons", cls.name, int(mask.sum()), len(placemarks))
        folders.append(f"<Folder><name>{escape(cls.name)}</name>{''.join(placemarks)}</Folder>")

    desc = f"<description>{escape(description)}</description>" if description else ""
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
        f"<name>{escape(name)}</name>{desc}{''.join(styles)}{''.join(folders)}"
        "</Document></kml>\n"
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if outfile.suffix.lower() == ".kmz":
        with zipfile.ZipFile(outfile, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("doc.kml", kml)
    else:
        outfile.write_text(kml, encoding="utf-8")
    return outfile


def geotiff_to_kml(
    tif: str | Path,
    outfile: str | Path,
    classes: Sequence[KmlClass],
    band: int = 1,
    **kwargs: object,
) -> Path:
    """Export a classified GeoTIFF (e.g. CAPTAIN's new_protection.tif) as KML/KMZ.

    Args:
        tif: Input GeoTIFF with class values in ``band``.
        outfile: Output ``.kml`` or ``.kmz`` path.
        classes: The classes to export (see ``raster_to_kml``).
        band: Raster band to read.
        **kwargs: Passed to ``raster_to_kml`` (name, description, decimals).

    Returns:
        Path of the written file.
    """
    with rasterio.open(tif) as src:
        grid = src.read(band)
        return raster_to_kml(grid, src.transform, src.crs, outfile, classes, **kwargs)
