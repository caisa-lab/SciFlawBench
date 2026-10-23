import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.sky import FRAMES, MAX_POINTS, MAX_RETURNED, SkyTool

tool = SkyTool()

# a fixed spread of positions, including both poles' neighbourhoods and the 0/360 seam
LONS = [0.0, 45.0, 123.4, 180.0, 266.4, 359.9, 10.0, 300.0]
LATS = [0.0, 30.0, -60.0, 89.0, -28.9, -89.0, 45.0, -10.0]


def run(operation, **arguments):
    out = tool.forward(operation, arguments)
    assert not out.startswith("Error"), out
    return json.loads(out)


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in degrees, written out independently of astropy."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return math.degrees(2 * math.asin(math.sqrt(h)))


# --- frames -----------------------------------------------------------------------------------------


def test_galactic_pole_and_centre_match_their_definitions():
    pole = run("convert", from_frame="icrs", to_frame="galactic", lon=[192.85948], lat=[27.12825])
    assert pole["lat"][0] == pytest.approx(90, abs=1e-3)
    centre = run("convert", from_frame="galactic", to_frame="icrs", lon=[0], lat=[0])
    assert centre["lon"][0] == pytest.approx(266.40499, abs=1e-3)
    assert centre["lat"][0] == pytest.approx(-28.93617, abs=1e-3)


def test_supergalactic_pole_and_origin_match_their_definitions():
    pole = run("convert", from_frame="galactic", to_frame="supergalactic", lon=[47.37], lat=[6.32])
    assert pole["lat"][0] == pytest.approx(90, abs=1e-3)
    origin = run("convert", from_frame="supergalactic", to_frame="galactic", lon=[0], lat=[0])
    assert origin["lon"][0] == pytest.approx(137.37, abs=1e-3)
    assert origin["lat"][0] == pytest.approx(0, abs=1e-3)


def test_ecliptic_pole_is_tilted_by_the_j2000_obliquity():
    pole = run("convert", from_frame="icrs", to_frame="ecliptic", lon=[270.0], lat=[90 - 23.4392911])
    assert pole["lat"][0] == pytest.approx(90, abs=1e-3)


def test_fk5_j2000_differs_from_icrs_by_less_than_a_tenth_of_an_arcsecond():
    out = run("convert", from_frame="icrs", to_frame="fk5", lon=LONS, lat=LATS)
    for lon, lat, lon2, lat2 in zip(LONS, LATS, out["lon"], out["lat"], strict=True):
        assert haversine(lon, lat, lon2, lat2) < 0.1 / 3600


@pytest.mark.parametrize("frame", [name for name in FRAMES if name != "icrs"])
def test_round_trip_through_each_frame_returns_the_input(frame):
    there = run("convert", from_frame="icrs", to_frame=frame, lon=LONS, lat=LATS)
    back = run("convert", from_frame=frame, to_frame="icrs", lon=there["lon"], lat=there["lat"])
    for lon, lat, lon2, lat2 in zip(LONS, LATS, back["lon"], back["lat"], strict=True):
        assert haversine(lon, lat, lon2, lat2) < 1e-8


def test_longitudes_are_returned_between_0_and_360():
    out = run("convert", from_frame="icrs", to_frame="galactic", lon=LONS, lat=LATS)
    assert all(0 <= lon < 360 for lon in out["lon"])


# --- separations -------------------------------------------------------------------------------------


def test_separation_matches_an_independent_formula():
    lon2, lat2 = LONS[1:] + LONS[:1], LATS[1:] + LATS[:1]
    out = run("separation", frame="icrs", lon1=LONS, lat1=LATS, lon2=lon2, lat2=lat2)
    expected = [haversine(*args) for args in zip(LONS, LATS, lon2, lat2, strict=True)]
    assert out["separation_deg"] == pytest.approx(expected, abs=1e-9)


def test_separation_does_not_depend_on_the_frame():
    a, b = (80.894, -69.756), (13.187, -72.829)  # LMC and SMC
    in_icrs = run("separation", frame="icrs", lon1=[a[0]], lat1=[a[1]], lon2=[b[0]], lat2=[b[1]])["separation_deg"][0]
    ga = run("convert", from_frame="icrs", to_frame="galactic", lon=[a[0], b[0]], lat=[a[1], b[1]])
    in_gal = run(
        "separation",
        frame="galactic",
        lon1=[ga["lon"][0]],
        lat1=[ga["lat"][0]],
        lon2=[ga["lon"][1]],
        lat2=[ga["lat"][1]],
    )["separation_deg"][0]
    assert in_icrs == pytest.approx(20.746, abs=1e-3)
    assert in_gal == pytest.approx(in_icrs, abs=1e-9)


# --- HEALPix -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("nside", [1, 64, 2048, 8192])
def test_healpix_info_follows_the_pixel_count_formula(nside):
    out = run("healpix_info", nside=nside)
    npix = 12 * nside**2
    area_sr = 4 * math.pi / npix
    assert out["npix"] == npix
    assert out["pixel_area_sr"] == pytest.approx(area_sr, rel=1e-12)
    assert out["pixel_area_deg2"] == pytest.approx(area_sr * (180 / math.pi) ** 2, rel=1e-12)
    assert out["pixel_area_arcmin2"] == pytest.approx(out["pixel_area_deg2"] * 3600, rel=1e-12)
    assert out["resolution_arcmin"] == pytest.approx(math.sqrt(area_sr) * 180 / math.pi * 60, rel=1e-9)


@pytest.mark.parametrize("order", ["ring", "nested"])
def test_pixel_centres_map_back_to_their_pixels(order):
    pixels = [0, 1, 777, 12_345, 49_151]  # nside 64 has 49,152 pixels
    centres = run("healpix_centers", nside=64, order=order, pixels=pixels)
    back = run("healpix_pixels", nside=64, order=order, lon=centres["lon"], lat=centres["lat"])
    assert back["pixels"] == pixels


def test_ring_and_nested_agree_at_nside_1_and_differ_above():
    pixels = list(range(12))
    assert run("healpix_centers", nside=1, order="ring", pixels=pixels) == run(
        "healpix_centers", nside=1, order="nested", pixels=pixels
    )
    ring = run("healpix_pixels", nside=8, order="ring", lon=LONS, lat=LATS)["pixels"]
    nested = run("healpix_pixels", nside=8, order="nested", lon=LONS, lat=LATS)["pixels"]
    assert ring != nested


def test_north_pole_is_in_one_of_the_four_polar_base_pixels():
    assert run("healpix_pixels", nside=1, order="ring", lon=[0], lat=[90])["pixels"][0] in range(4)


def test_disc_size_matches_its_area_and_contains_its_centre():
    big = run("healpix_disc", nside=2048, order="ring", lon=10, lat=20, radius=1)
    expected = 12 * 2048**2 * (1 - math.cos(math.radians(1))) / 2
    assert big["n_pixels"] == pytest.approx(expected, rel=0.1)
    assert "pixels" not in big and "note" in big

    small = run("healpix_disc", nside=256, order="nested", lon=10, lat=20, radius=0.5)
    centre = run("healpix_pixels", nside=256, order="nested", lon=[10], lat=[20])["pixels"][0]
    assert small["n_pixels"] == len(small["pixels"]) <= MAX_RETURNED
    assert centre in small["pixels"]


# --- bad input ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation, arguments, message",
    [
        ("convert", {"from_frame": "altaz", "to_frame": "icrs", "lon": [0], "lat": [0]}, "from_frame must be one of"),
        ("convert", {"from_frame": "icrs", "to_frame": "itrs", "lon": [0], "lat": [0]}, "to_frame must be one of"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": [0], "lat": [95]}, "between -90 and 90"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": [0, 1], "lat": [0]}, "same length"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": [], "lat": []}, "non-empty list"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": 0, "lat": 0}, "non-empty list"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": ["0"], "lat": [0]}, "finite number"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": [True], "lat": [0]}, "finite number"),
        ("convert", {"from_frame": "icrs", "to_frame": "galactic", "lon": [10**400], "lat": [0]}, "too large"),
        (
            "convert",
            {
                "from_frame": "icrs",
                "to_frame": "galactic",
                "lon": [0] * (MAX_POINTS + 1),
                "lat": [0] * (MAX_POINTS + 1),
            },
            "at most",
        ),
        ("separation", {"frame": "icrs", "lon1": [0], "lat1": [0], "lon2": [0, 1], "lat2": [0, 1]}, "same length"),
        ("healpix_info", {"nside": 3}, "power of 2"),
        ("healpix_info", {"nside": 16384}, "power of 2"),
        ("healpix_info", {"nside": 2048.0}, "power of 2"),
        ("healpix_pixels", {"nside": 64, "order": "rings", "lon": [0], "lat": [0]}, 'order must be "ring" or "nested"'),
        ("healpix_pixels", {"nside": 64, "lon": [0], "lat": [0]}, "missing 1 required positional argument: 'order'"),
        ("healpix_centers", {"nside": 1, "order": "ring", "pixels": [12]}, "integers from 0 to 11"),
        ("healpix_centers", {"nside": 1, "order": "ring", "pixels": [1.5]}, "integers from 0 to 11"),
        ("healpix_centers", {"nside": 1, "order": "ring", "pixels": []}, "non-empty list"),
        ("healpix_disc", {"nside": 64, "order": "ring", "lon": 0, "lat": 0, "radius": 0}, "radius must lie"),
        ("healpix_disc", {"nside": 64, "order": "ring", "lon": 0, "lat": 0, "radius": 200}, "radius must lie"),
        ("healpix_disc", {"nside": 64, "order": "ring", "lon": 0, "lat": 91, "radius": 1}, "between -90 and 90"),
        ("healpix_disc", {"nside": 8192, "order": "ring", "lon": 0, "lat": 0, "radius": 60}, "the limit is"),
        (
            "convert",
            {"from_frame": "icrs", "to_frame": "galactic", "lon": [0], "lat": [0], "x": 1},
            "unexpected keyword",
        ),
        ("altaz", {}, "unknown operation"),
    ],
)
def test_bad_input_returns_an_error_message(operation, arguments, message):
    out = tool.forward(operation, arguments)
    assert out.startswith("Error") and message in out


def test_arguments_must_be_a_dict():
    assert tool.forward("healpix_info", [2048]).startswith("Error: arguments must be a dict")


def test_tool_opens_no_network_connection_and_writes_no_files(tmp_path):
    # a fresh interpreter with an empty home directory and sockets that refuse to connect
    script = """
import json, socket
def refuse(*args, **kwargs):
    raise RuntimeError("network access attempted")
socket.socket.connect = refuse
socket.create_connection = refuse
from tools.sky import FRAMES, SkyTool
tool = SkyTool()
positions = {"lon": [266.4, 10], "lat": [-29.0, 89]}
calls = [("convert", {"from_frame": "icrs", "to_frame": name, **positions}) for name in FRAMES]
calls += [
    ("separation", {"frame": "ecliptic", "lon1": [1], "lat1": [2], "lon2": [3], "lat2": [4]}),
    ("healpix_info", {"nside": 2048}),
    ("healpix_pixels", {"nside": 2048, "order": "nested", "lon": [10], "lat": [20]}),
    ("healpix_centers", {"nside": 2048, "order": "ring", "pixels": [0, 5]}),
    ("healpix_disc", {"nside": 2048, "order": "ring", "lon": 10, "lat": 20, "radius": 1}),
]
for operation, arguments in calls:
    out = tool.forward(operation, arguments)
    assert not out.startswith("Error"), out
    json.loads(out)
print("ok")
"""
    home = tmp_path / "home"
    home.mkdir()
    src = Path(__file__).resolve().parents[3] / "src"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": str(src)}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert list(home.rglob("*")) == []  # nothing was written under the home directory


def test_registered_in_the_tool_registry():
    from core.events import EventWatcher
    from tools.definitions import tool_registry

    sky = tool_registry.create("sky", watcher=EventWatcher("t", sink=lambda e: None))
    assert sky.name == "sky"


def test_description_says_how_to_read_the_output():
    # descriptions are wrapped over several lines, so compare with the whitespace collapsed
    assert "JSON string (parse it with json.loads)" in " ".join(SkyTool.description.split())
