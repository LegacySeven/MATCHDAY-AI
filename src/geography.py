import os
import json
import time
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Tuple

try:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="football_predictor_bot")
except ImportError:
    geolocator = None

COORD_CACHE_FILE = "data/team_coordinates.json"

if os.path.exists(COORD_CACHE_FILE):
    with open(COORD_CACHE_FILE, "r") as f:
        _GLOBAL_CACHE = json.load(f)
else:
    _GLOBAL_CACHE = {}


def load_cache():
    return _GLOBAL_CACHE


def save_cache(cache):
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = cache
    os.makedirs("data", exist_ok=True)
    with open(COORD_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_team_coordinates(team_name: str) -> Optional[Tuple[float, float]]:
    """
    Gets (lat, lon) for a team, or None if the geocode failed.

    FIX: previously returned (0.0, 0.0) on failure, which is
    indistinguishable from a real coordinate near (0,0) and made
    haversine_distance() silently return 0.0 km — identical to a
    genuine local derby. Now failures propagate as None -> NaN,
    so they can be filled deliberately (see feature_engineering.py)
    instead of being confused with real short distances.
    """
    cache = load_cache()
    if team_name in cache:
        val = cache[team_name]
        return tuple(val) if val is not None else None

    if not geolocator:
        cache[team_name] = None
        save_cache(cache)
        return None

    print(f"Geocoding {team_name}...")
    try:
        query = f"{team_name} football club city"
        location = geolocator.geocode(query, timeout=10)
        coords = (location.latitude, location.longitude) if location else None

        cache[team_name] = coords
        save_cache(cache)
        time.sleep(1.1)  # Respect Nominatim's 1 req/sec limit
        return coords
    except Exception as e:
        print(f"Error geocoding {team_name}: {e}")
        cache[team_name] = None
        save_cache(cache)
        return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in kilometers between two coordinates."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def calculate_travel_distance(home_team: str, away_team: str) -> float:
    """
    Returns the distance the away team traveled, in km, or NaN if
    either team's coordinates could not be geocoded.
    """
    home_coords = get_team_coordinates(home_team)
    away_coords = get_team_coordinates(away_team)
    if home_coords is None or away_coords is None:
        return float("nan")
    return haversine_distance(home_coords[0], home_coords[1], away_coords[0], away_coords[1])