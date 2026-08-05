"""
Tool implementations.

Each tool returns a plain JSON-serializable dict (pricing in USD, as
required). Data is mocked but deterministic so the same city pair always
returns the same numbers, and unseen city pairs still resolve to sensible
values instead of raising.
"""

from __future__ import annotations


def _seed_from_string(s: str) -> int:
    """Small deterministic hash so unseen city pairs still get stable numbers."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _round2(n: float) -> float:
    return round(n * 100) / 100


# A few real-world-ish direct routes so common queries look realistic.
_FLIGHT_DB = {
    "lagos->nairobi": {
        "airline": "Kenya Airways",
        "flight_number": "KQ 785",
        "duration_hours": 6.5,
        "price_usd": 450,
    },
    "nairobi->lagos": {
        "airline": "Kenya Airways",
        "flight_number": "KQ 786",
        "duration_hours": 6.5,
        "price_usd": 470,
    },
}

_HOTEL_DB = {
    "nairobi": {"hotel_name": "Nairobi Grand Suites", "price_per_night_usd": 120},
    "lagos": {"hotel_name": "Lagos Business Hotel", "price_per_night_usd": 140},
}

# Fixed mock exchange rates, expressed as "1 USD = X <currency>".
_USD_RATES = {
    "USD": 1,
    "NGN": 1550,
    "KES": 129,
    "EUR": 0.92,
    "GBP": 0.79,
}


def _synthetic_flight(from_key: str, to_key: str) -> dict:
    seed = _seed_from_string(f"{from_key}|{to_key}")
    duration_hours = _round2(2 + (seed % 900) / 100)  # 2.0 - 10.99 hrs
    price_usd = 150 + (seed % 700)  # 150 - 849 USD
    return {
        "airline": "Regional Air Connect",
        "flight_number": f"RA {100 + (seed % 900)}",
        "duration_hours": duration_hours,
        "price_usd": price_usd,
    }


def _synthetic_hotel(city_key: str) -> dict:
    seed = _seed_from_string(city_key)
    price_per_night_usd = 60 + (seed % 140)  # 60 - 199 USD/night
    name = city_key[:1].upper() + city_key[1:] if city_key else "City"
    return {
        "hotel_name": f"{name} City Hotel",
        "price_per_night_usd": price_per_night_usd,
    }


def get_flight_schedule(origin: str, destination: str) -> dict:
    """
    Returns the outbound leg (origin -> destination) and return leg
    (destination -> origin) with duration and price in USD for each leg.
    """
    origin_key = origin.strip().lower()
    dest_key = destination.strip().lower()

    outbound = _FLIGHT_DB.get(
        f"{origin_key}->{dest_key}"
    ) or _synthetic_flight(origin_key, dest_key)
    ret = _FLIGHT_DB.get(
        f"{dest_key}->{origin_key}"
    ) or _synthetic_flight(dest_key, origin_key)

    return {
        "origin": origin,
        "destination": destination,
        "outbound_flight": {
            "airline": outbound["airline"],
            "flight_number": outbound["flight_number"],
            "route": f"{origin} -> {destination}",
            "duration_hours": outbound["duration_hours"],
            "price_usd": outbound["price_usd"],
        },
        "return_flight": {
            "airline": ret["airline"],
            "flight_number": ret["flight_number"],
            "route": f"{destination} -> {origin}",
            "duration_hours": ret["duration_hours"],
            "price_usd": ret["price_usd"],
        },
        "total_flight_duration_hours": _round2(
            outbound["duration_hours"] + ret["duration_hours"]
        ),
        "total_flight_cost_usd": outbound["price_usd"] + ret["price_usd"],
        "currency": "USD",
    }


def get_hotel_schedule(city: str, nights: int) -> dict:
    """
    Returns nightly rate and total cost in USD for the given city and
    number of nights.
    """
    city_key = city.strip().lower()
    hotel = _HOTEL_DB.get(city_key) or _synthetic_hotel(city_key)
    num_nights = int(nights)

    return {
        "city": city,
        "hotel_name": hotel["hotel_name"],
        "nights": num_nights,
        "price_per_night_usd": hotel["price_per_night_usd"],
        "total_price_usd": _round2(hotel["price_per_night_usd"] * num_nights),
        "currency": "USD",
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Converts `amount` from `from_currency` to `to_currency`.
    Falls back to a 1:1 rate for unrecognized currency codes.
    """
    from_code = (from_currency or "USD").strip().upper()
    to_code = (to_currency or "USD").strip().upper()

    from_rate = _USD_RATES.get(from_code, 1)  # units of `from` per 1 USD
    to_rate = _USD_RATES.get(to_code, 1)  # units of `to` per 1 USD

    amount_in_usd = amount / from_rate
    converted = amount_in_usd * to_rate

    return {
        "original_amount": amount,
        "from_currency": from_code,
        "to_currency": to_code,
        "converted_amount": _round2(converted),
    }


# Maps tool name (as declared to the LLM) -> python callable.
TOOL_IMPLEMENTATIONS = {
    "get_flight_schedule": get_flight_schedule,
    "get_hotel_schedule": get_hotel_schedule,
    "convert_currency": convert_currency,
}
