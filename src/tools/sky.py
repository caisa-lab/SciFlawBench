"""
An astropy-backed tool for positions on the sky.

It converts coordinates between celestial frames, measures angular separations, and answers HEALPix pixel
geometry questions (with astropy-healpix). astropy runs here, in harness code: the agent passes numbers and
names from a fixed list and gets JSON back. It never handles astropy objects, maps, files or code, so the code
executor's restrictions are unaffected. Only frames that need no downloaded data are offered, and astropy's
internet access and automatic Earth-orientation downloads are switched off before astropy is first used.

astropy and astropy-healpix are imported when the tool is first used, not at module level: the tool registry
imports this module on every harness start, and importing astropy takes one to two seconds that only runs using
this tool should pay.
"""

import json
import math
from functools import cache
from types import SimpleNamespace

from smolagents import Tool

MAX_POINTS = 1_000  # positions or pixels in one call
MAX_NSIDE = 8192
# Largest disc a search may cover. astropy-healpix's cone search time grows roughly with the square of the pixel
# count (about 1 s at 15,000 pixels and 8 s at 50,000 here), and the harness has no per-call timeout.
MAX_DISC_PIXELS = 50_000
MAX_RETURNED = 1_000  # pixel indices returned by a disc search

FRAMES = ("icrs", "fk5", "galactic", "supergalactic", "ecliptic")
ORDERS = ("ring", "nested")


@cache
def _astropy() -> SimpleNamespace:
    """The astropy pieces this tool uses, imported once on first use with remote data switched off."""
    import astropy.units as u
    from astropy.coordinates import (
        FK5,
        ICRS,
        BarycentricMeanEcliptic,
        Galactic,
        Supergalactic,
        UnitSphericalRepresentation,
    )
    from astropy.utils import iers
    from astropy.utils.data import conf as astropy_data_conf
    from astropy_healpix import HEALPix

    # none of the frames below needs remote data; make sure nothing in astropy tries to fetch any
    astropy_data_conf.allow_internet = False
    iers.conf.auto_download = False
    frames = {
        "icrs": ICRS(),
        "fk5": FK5(equinox="J2000"),
        "galactic": Galactic(),
        "supergalactic": Supergalactic(),
        "ecliptic": BarycentricMeanEcliptic(equinox="J2000"),
    }
    return SimpleNamespace(u=u, UnitSphericalRepresentation=UnitSphericalRepresentation, HEALPix=HEALPix, frames=frames)


class SkyError(ValueError):
    """Invalid input; reported to the agent as an error message."""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:  # a Python int beyond the float range
        raise SkyError(f"{name} is too large to represent") from exc
    if not math.isfinite(number):
        raise SkyError(f"{name} must be a finite number")
    return number


def _numbers(values, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise SkyError(f"{name} must be a non-empty list of numbers")
    if len(values) > MAX_POINTS:
        raise SkyError(f"{name} accepts at most {MAX_POINTS} values")
    return [_number(value, name) for value in values]


def _positions(lon, lat, suffix: str = ""):
    """Validated longitudes and latitudes in degrees, as astropy quantities."""
    lons, lats = _numbers(lon, f"lon{suffix}"), _numbers(lat, f"lat{suffix}")
    if len(lons) != len(lats):
        raise SkyError(f"lon{suffix} and lat{suffix} must have the same length")
    if any(abs(b) > 90 for b in lats):
        raise SkyError(f"lat{suffix} must lie between -90 and 90 degrees")
    u = _astropy().u
    return lons * u.deg, lats * u.deg


def _frame(name, label: str = "frame"):
    if name not in FRAMES:
        raise SkyError(f"{label} must be one of {list(FRAMES)}")
    return _astropy().frames[name]


def _healpix(nside, order):
    if isinstance(nside, bool) or not isinstance(nside, int) or not 1 <= nside <= MAX_NSIDE or nside & (nside - 1):
        raise SkyError(f"nside must be a power of 2 between 1 and {MAX_NSIDE}")
    if order not in ORDERS:
        raise SkyError('order must be "ring" or "nested"')
    return _astropy().HEALPix(nside=nside, order=order)


def _degrees(angles) -> list[float]:
    return [float(a) for a in angles.to_value(_astropy().u.deg)]


def convert(from_frame, to_frame, lon, lat) -> dict:
    source, target = _frame(from_frame, "from_frame"), _frame(to_frame, "to_frame")
    lons, lats = _positions(lon, lat)
    ap = _astropy()
    out = source.realize_frame(ap.UnitSphericalRepresentation(lons, lats)).transform_to(target)
    spherical = out.represent_as(ap.UnitSphericalRepresentation)
    return {"frame": to_frame, "lon": _degrees(spherical.lon.wrap_at(360 * ap.u.deg)), "lat": _degrees(spherical.lat)}


def separation(frame, lon1, lat1, lon2, lat2) -> dict:
    target = _frame(frame)
    first, second = _positions(lon1, lat1, "1"), _positions(lon2, lat2, "2")
    if len(first[0]) != len(second[0]):
        raise SkyError("the two lists of positions must have the same length")
    sphere = _astropy().UnitSphericalRepresentation
    a = target.realize_frame(sphere(*first))
    b = target.realize_frame(sphere(*second))
    return {"separation_deg": _degrees(a.separation(b))}


def healpix_info(nside) -> dict:
    hp = _healpix(nside, "ring")
    u = _astropy().u
    area = hp.pixel_area
    return {
        "npix": int(hp.npix),
        "pixel_area_sr": float(area.to_value(u.sr)),
        "pixel_area_deg2": float(area.to_value(u.deg**2)),
        "pixel_area_arcmin2": float(area.to_value(u.arcmin**2)),
        "resolution_arcmin": float(hp.pixel_resolution.to_value(u.arcmin)),
    }


def healpix_pixels(nside, order, lon, lat) -> dict:
    hp = _healpix(nside, order)
    lons, lats = _positions(lon, lat)
    return {"pixels": [int(p) for p in hp.lonlat_to_healpix(lons, lats)]}


def healpix_centers(nside, order, pixels) -> dict:
    hp = _healpix(nside, order)
    if not isinstance(pixels, (list, tuple)) or not pixels or len(pixels) > MAX_POINTS:
        raise SkyError(f"pixels must be a non-empty list of at most {MAX_POINTS} pixel indices")
    if any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < hp.npix for p in pixels):
        raise SkyError(f"pixels must be integers from 0 to {hp.npix - 1}")
    lon, lat = hp.healpix_to_lonlat(list(pixels))
    return {"lon": _degrees(lon), "lat": _degrees(lat)}


def healpix_disc(nside, order, lon, lat, radius) -> dict:
    hp = _healpix(nside, order)
    lon_deg, lat_deg, radius_deg = _number(lon, "lon"), _number(lat, "lat"), _number(radius, "radius")
    if abs(lat_deg) > 90:
        raise SkyError("lat must lie between -90 and 90 degrees")
    if not 0 < radius_deg <= 180:
        raise SkyError("radius must lie above 0 and at most 180 degrees")
    expected = hp.npix * (1 - math.cos(math.radians(radius_deg))) / 2
    if expected > MAX_DISC_PIXELS:
        raise SkyError(f"the disc would cover about {expected:,.0f} pixels; the limit is {MAX_DISC_PIXELS:,}")
    deg = _astropy().u.deg
    found = hp.cone_search_lonlat(lon_deg * deg, lat_deg * deg, radius_deg * deg)
    result = {"n_pixels": int(len(found))}
    if len(found) <= MAX_RETURNED:
        result["pixels"] = sorted(int(p) for p in found)
    else:
        result["note"] = f"pixel indices are returned only for discs of at most {MAX_RETURNED} pixels"
    return result


OPERATIONS = {
    "convert": convert,
    "separation": separation,
    "healpix_info": healpix_info,
    "healpix_pixels": healpix_pixels,
    "healpix_centers": healpix_centers,
    "healpix_disc": healpix_disc,
}


class SkyTool(Tool):
    name = "sky"

    description = f"""Sky coordinates and HEALPix pixel geometry, computed with astropy and astropy-healpix. Returns a
    JSON string (parse it with json.loads). Angles are in degrees. Positions are given as lists `lon` and `lat` (up to
    {MAX_POINTS}): right ascension
    and declination in icrs and fk5 (J2000), l and b in galactic, SGL and SGB in supergalactic, ecliptic longitude
    and latitude in ecliptic (mean ecliptic, J2000). Operations and their arguments:
    - convert: from_frame, to_frame, lon, lat -> lon and lat in to_frame
    - separation: frame, lon1, lat1, lon2, lat2 (lists of equal length) -> angular separations
    - healpix_info: nside -> number of pixels, pixel area (sr, deg², arcmin²) and resolution (arcmin)
    - healpix_pixels: nside, order ("ring" or "nested"), lon, lat -> the pixel containing each position
    - healpix_centers: nside, order, pixels -> lon and lat of each pixel centre
    - healpix_disc: nside, order, lon, lat, radius (single position) -> number of pixels within radius, and their
      indices when there are at most {MAX_RETURNED}; discs of more than {MAX_DISC_PIXELS:,} pixels are refused
    nside is a power of 2 up to {MAX_NSIDE}. HEALPix positions are in the map's own coordinate frame."""

    inputs = {
        "operation": {"type": "string", "description": f"one of {', '.join(OPERATIONS)}"},
        "arguments": {"type": "object", "description": "the operation's arguments, as a dict of names to values"},
    }
    output_type = "string"

    def forward(self, operation: str, arguments: dict) -> str:
        if operation not in OPERATIONS:
            return f"Error: unknown operation {operation!r}; available: {', '.join(OPERATIONS)}"
        if not isinstance(arguments, dict):
            return "Error: arguments must be a dict of argument names to values"
        try:
            result = OPERATIONS[operation](**arguments)
        except TypeError as exc:  # missing or unexpected argument names
            return f"Error: {exc}"
        except SkyError as exc:
            return f"Error: {exc}"
        try:
            return json.dumps(result, allow_nan=False)
        except ValueError:  # inf or nan in the result
            return "Error: result is not finite; check the inputs"
