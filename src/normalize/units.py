"""Unit parsing and canonicalisation.

Industrial catalogues mix HP with kW, inches with mm, PSI with bar and GPM with
m3/h in the same spreadsheet. Everything is converted to the canonical unit
declared in the attribute dictionary, and the conversion is recorded so the
original figure stays visible in the audit trail.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# Canonical unit per family, plus multiplicative factors to reach it.
# offset is applied after the factor: canonical = raw * factor + offset
UNIT_FAMILIES: dict[str, dict[str, Any]] = {
    "length": {
        "canonical": "mm",
        "units": {
            "mm": 1.0, "millimeter": 1.0, "millimetre": 1.0, "millimeters": 1.0,
            "cm": 10.0, "centimeter": 10.0, "centimetre": 10.0,
            "m": 1000.0, "meter": 1000.0, "metre": 1000.0, "meters": 1000.0,
            "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4,
            "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8,
            "mil": 0.0254, "thou": 0.0254,
            "um": 0.001, "micron": 0.001,
        },
    },
    # A US building-products catalogue is written in inches, feet and pounds.
    # Converting those to millimetres would be correct and useless: the delivery
    # template carries its own UOM column, so the canonical unit is the one the
    # trade actually publishes.
    "length_in": {
        "canonical": "in",
        "units": {
            "in": 1.0, "inch": 1.0, "inches": 1.0, '"': 1.0, "''": 1.0,
            "ft": 12.0, "foot": 12.0, "feet": 12.0, "'": 12.0,
            "yd": 36.0, "yard": 36.0, "yards": 36.0,
            "mm": 0.0393701, "cm": 0.393701, "m": 39.3701,
        },
    },
    "length_ft": {
        "canonical": "ft",
        "units": {
            "ft": 1.0, "foot": 1.0, "feet": 1.0, "'": 1.0,
            "in": 0.0833333, "inch": 0.0833333, '"': 0.0833333,
            "yd": 3.0, "yard": 3.0, "m": 3.28084, "mm": 0.00328084,
        },
    },
    "mass_lb": {
        "canonical": "lb",
        "units": {
            "lb": 1.0, "lbs": 1.0, "pound": 1.0, "pounds": 1.0, "#": 1.0,
            "oz": 0.0625, "ounce": 0.0625, "ounces": 0.0625,
            "kg": 2.20462, "g": 0.00220462, "ton": 2000.0, "tons": 2000.0,
        },
    },
    "power_w": {
        "canonical": "W",
        "units": {
            "w": 1.0, "watt": 1.0, "watts": 1.0,
            "kw": 1000.0, "kilowatt": 1000.0, "mw": 0.001, "milliwatt": 0.001,
        },
    },
    "luminous_flux": {
        "canonical": "lm",
        "units": {"lm": 1.0, "lumen": 1.0, "lumens": 1.0, "klm": 1000.0},
    },
    "color_temperature": {
        "canonical": "K",
        "units": {"k": 1.0, "kelvin": 1.0},
    },
    "sound_level": {
        "canonical": "dBA",
        "units": {"dba": 1.0, "db": 1.0, "decibel": 1.0, "decibels": 1.0},
    },
    "volume_cuft": {
        "canonical": "cu ft",
        "units": {
            "cu ft": 1.0, "cuft": 1.0, "ft3": 1.0, "cf": 1.0, "cubic feet": 1.0,
            "cu in": 0.000578704, "l": 0.0353147, "liter": 0.0353147,
        },
    },
    "area_sqft": {
        "canonical": "sq ft",
        "units": {
            "sq ft": 1.0, "sqft": 1.0, "ft2": 1.0, "square feet": 1.0,
            "sq in": 0.00694444, "sq": 100.0,   # roofing "square" = 100 sq ft
        },
    },
    "length_large": {
        "canonical": "m",
        "units": {
            "m": 1.0, "meter": 1.0, "metre": 1.0, "meters": 1.0,
            "mm": 0.001, "cm": 0.01, "km": 1000.0,
            "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
            "in": 0.0254, "inch": 0.0254,
        },
    },
    "mass": {
        "canonical": "kg",
        "units": {
            "kg": 1.0, "kilogram": 1.0, "kgs": 1.0, "kilograms": 1.0,
            "g": 0.001, "gram": 0.001, "grams": 0.001,
            "mg": 1e-6,
            "t": 1000.0, "ton": 1000.0, "tonne": 1000.0, "mt": 1000.0,
            "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
            "oz": 0.028349523, "ounce": 0.028349523,
        },
    },
    "power": {
        "canonical": "kW",
        "units": {
            "kw": 1.0, "kilowatt": 1.0, "kilowatts": 1.0,
            "w": 0.001, "watt": 0.001, "watts": 0.001,
            "mw": 1000.0, "megawatt": 1000.0,
            "hp": 0.7457, "horsepower": 0.7457, "bhp": 0.7457,
            "ps": 0.735499, "cv": 0.735499,
        },
    },
    "voltage": {
        "canonical": "V",
        "units": {
            "v": 1.0, "volt": 1.0, "volts": 1.0, "vac": 1.0, "vdc": 1.0,
            "kv": 1000.0, "kilovolt": 1000.0,
            "mv": 0.001, "millivolt": 0.001,
        },
    },
    "current": {
        "canonical": "A",
        "units": {
            "a": 1.0, "amp": 1.0, "amps": 1.0, "ampere": 1.0, "amperes": 1.0,
            "ma": 0.001, "milliamp": 0.001, "milliampere": 0.001,
            "ka": 1000.0, "kiloamp": 1000.0,
        },
    },
    "current_large": {
        "canonical": "kA",
        "units": {"ka": 1.0, "kiloamp": 1.0, "a": 0.001, "amp": 0.001, "amps": 0.001},
    },
    "pressure": {
        "canonical": "bar",
        "units": {
            "bar": 1.0, "bars": 1.0,
            "mbar": 0.001, "millibar": 0.001,
            "pa": 1e-5, "pascal": 1e-5,
            "kpa": 0.01, "kilopascal": 0.01,
            "mpa": 10.0, "megapascal": 10.0,
            "psi": 0.0689476, "psig": 0.0689476, "lbf/in2": 0.0689476,
            "atm": 1.01325, "kgf/cm2": 0.980665, "kg/cm2": 0.980665,
            "torr": 0.00133322, "mmhg": 0.00133322,
        },
    },
    "temperature": {
        "canonical": "degC",
        "units": {
            "c": (1.0, 0.0), "degc": (1.0, 0.0), "celsius": (1.0, 0.0),
            "centigrade": (1.0, 0.0), "°c": (1.0, 0.0),
            "f": (5.0 / 9.0, -32.0 * 5.0 / 9.0), "degf": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
            "fahrenheit": (5.0 / 9.0, -32.0 * 5.0 / 9.0), "°f": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
            "k": (1.0, -273.15), "kelvin": (1.0, -273.15),
        },
    },
    "speed": {
        "canonical": "rpm",
        "units": {
            "rpm": 1.0, "r/min": 1.0, "min-1": 1.0, "1/min": 1.0, "revs/min": 1.0,
            "rps": 60.0, "hz": 60.0,
        },
    },
    "frequency": {
        "canonical": "Hz",
        "units": {
            "hz": 1.0, "hertz": 1.0,
            "khz": 1000.0, "mhz": 1e6, "ghz": 1e9,
        },
    },
    "flow": {
        "canonical": "m3/h",
        "units": {
            "m3/h": 1.0, "m3/hr": 1.0, "cbm/h": 1.0, "m^3/h": 1.0, "cum/hr": 1.0,
            "l/min": 0.06, "lpm": 0.06, "lit/min": 0.06,
            "l/s": 3.6, "lps": 3.6,
            "l/h": 0.001, "lph": 0.001,
            "gpm": 0.2271247, "usgpm": 0.2271247, "gal/min": 0.2271247,
            "cfm": 1.699011, "ft3/min": 1.699011,
        },
    },
    "force": {
        "canonical": "kN",
        "units": {
            "kn": 1.0, "kilonewton": 1.0,
            "n": 0.001, "newton": 0.001,
            "mn": 1000.0,
            "kgf": 0.00980665, "kg": 0.00980665,
            "lbf": 0.004448222,
        },
    },
    "area": {
        "canonical": "mm2",
        "units": {
            "mm2": 1.0, "mm^2": 1.0, "sqmm": 1.0, "sq mm": 1.0, "sq.mm": 1.0,
            "cm2": 100.0, "cm^2": 100.0,
            "m2": 1e6, "m^2": 1e6,
            "in2": 645.16, "sqin": 645.16,
        },
    },
    "time": {
        "canonical": "month",
        "units": {
            "month": 1.0, "months": 1.0, "mo": 1.0,
            "year": 12.0, "years": 12.0, "yr": 12.0, "yrs": 12.0,
            "day": 1.0 / 30.0, "days": 1.0 / 30.0,
        },
    },
}

# AWG -> mm2, needed because cable catalogues switch between the two freely.
AWG_TO_MM2 = {
    "0000": 107.2, "000": 85.0, "00": 67.4, "0": 53.5,
    "1": 42.4, "2": 33.6, "3": 26.7, "4": 21.2, "5": 16.8, "6": 13.3,
    "7": 10.5, "8": 8.37, "9": 6.63, "10": 5.26, "11": 4.17, "12": 3.31,
    "13": 2.62, "14": 2.08, "15": 1.65, "16": 1.31, "17": 1.04, "18": 0.823,
    "20": 0.518, "22": 0.326, "24": 0.205, "26": 0.129, "28": 0.081, "30": 0.051,
}

# The comma-grouped branch must require an actual comma group: with `*` it
# matched the first three digits of "2700" and left "0" behind as the unit, so
# every four-digit figure (2700 K, 1465 rpm, 1100 lm) parsed an order of
# magnitude low. Plain numbers are matched whole by the second branch.
_NUMBER = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?|[-+]?\.\d+"
_VALUE_UNIT_RE = re.compile(rf"({_NUMBER})\s*([^\s\d,;]*(?:/[^\s\d,;]+)?)", re.IGNORECASE)
_RANGE_RE = re.compile(rf"({_NUMBER})\s*(?:-|to|\.\.\.|–|…)\s*({_NUMBER})", re.IGNORECASE)
_AWG_RE = re.compile(r"\b(\d{1,2}|0{1,4})\s*awg\b|\bawg\s*(\d{1,2}|0{1,4})\b", re.IGNORECASE)


@dataclass
class ParsedQuantity:
    """Result of parsing a raw quantity string."""

    value: float | None
    unit: str | None
    raw: str
    is_range: bool = False
    range_low: float | None = None
    range_high: float | None = None
    note: str | None = None


def _clean_unit(token: str) -> str:
    return token.strip().strip(".,;:()[]").lower().replace("µ", "u")


_MIXED_FRACTION_RE = re.compile(r"(?<![\d/.])(\d+)[\s-](\d{1,2})/(\d{1,2})(?![\d/])")
_SIMPLE_FRACTION_RE = re.compile(r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?![\d/])")


def expand_fractions(text: str) -> str:
    """Rewrite imperial fractions as decimals: 2-3/4 -> 2.75, 1/2 -> 0.5.

    Mixed fractions are handled first: "50-1/4" is fifty and a quarter, not a
    range from 50 to 1, and every downstream parser would read it as the latter.
    """
    def _mixed(match: re.Match[str]) -> str:
        whole, num, den = (int(g) for g in match.groups())
        return f"{whole + num / den:g}" if den else match.group(0)

    def _simple(match: re.Match[str]) -> str:
        num, den = (int(g) for g in match.groups())
        return f"{num / den:g}" if den else match.group(0)

    return _SIMPLE_FRACTION_RE.sub(_simple, _MIXED_FRACTION_RE.sub(_mixed, text))


def parse_quantity(raw: Any) -> ParsedQuantity:
    """Pull a number and (optionally) a unit token out of a messy string."""
    if raw is None:
        return ParsedQuantity(None, None, "")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return ParsedQuantity(float(raw), None, str(raw))

    text = str(raw).strip()
    if not text:
        return ParsedQuantity(None, None, "")

    # US building products are sized in fractions - 1/2", 2-3/4 in, 50-1/4IN.
    # Everything downstream expects a decimal, and a range parser would read
    # "50-1/4" as "50 to 1" without this.
    text = expand_fractions(text)

    # "16 - 24 V" style ranges: keep both ends, use the midpoint as the value.
    range_match = _RANGE_RE.search(text)
    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        tail = text[range_match.end():].strip()
        unit = _clean_unit(tail.split()[0]) if tail else None
        return ParsedQuantity(
            value=(low + high) / 2.0,
            unit=unit,
            raw=text,
            is_range=True,
            range_low=low,
            range_high=high,
            note="range midpoint used",
        )

    match = _VALUE_UNIT_RE.search(text)
    if not match:
        return ParsedQuantity(None, None, text)

    value = float(match.group(1).replace(",", ""))
    unit_token = _clean_unit(match.group(2) or "")
    # A trailing word can be the unit when it did not stick to the number.
    if not unit_token:
        tail = text[match.end():].strip()
        if tail:
            unit_token = _clean_unit(tail.split()[0])
    return ParsedQuantity(value, unit_token or None, text)


def convert(value: float, from_unit: str | None, family: str) -> tuple[float | None, str]:
    """Convert `value` from `from_unit` into the canonical unit of `family`.

    Returns (converted_value, canonical_unit). If the unit is unknown the value
    is passed through unchanged and the caller decides whether to trust it.
    """
    spec = UNIT_FAMILIES.get(family)
    if spec is None:
        return value, from_unit or ""

    canonical = spec["canonical"]
    if from_unit is None:
        # No unit given: assume the source already used the canonical unit.
        return value, canonical

    key = _clean_unit(from_unit)
    factor = spec["units"].get(key)
    if factor is None:
        return None, canonical

    if isinstance(factor, tuple):                     # affine (temperature)
        scale, offset = factor
        return round(value * scale + offset, 4), canonical
    return round(value * factor, 6), canonical


def normalize_quantity(
    raw: Any, family: str | None, canonical_unit: str | None = None
) -> tuple[float | None, str | None, dict[str, Any]]:
    """Full parse-and-convert for one attribute value.

    Returns (value, unit, meta) where meta explains what happened, so the
    conversion can be shown in the audit trail.
    """
    meta: dict[str, Any] = {"converted": False}
    parsed = parse_quantity(raw)
    meta["parsed_raw"] = parsed.raw
    meta["detected_unit"] = parsed.unit
    if parsed.is_range:
        meta["range"] = [parsed.range_low, parsed.range_high]
        meta["note"] = parsed.note

    if parsed.value is None:
        meta["error"] = "no numeric value found"
        return None, canonical_unit, meta

    if not family:
        return parsed.value, canonical_unit, meta

    converted, unit = convert(parsed.value, parsed.unit, family)
    if converted is None:
        meta["error"] = f"unknown unit '{parsed.unit}' for family '{family}'"
        return parsed.value, unit, meta

    if parsed.unit and not math.isclose(converted, parsed.value, rel_tol=1e-9):
        meta["converted"] = True
        meta["from"] = f"{parsed.value} {parsed.unit}"
        meta["to"] = f"{converted} {unit}"
    return converted, unit, meta


def awg_to_mm2(text: str) -> float | None:
    """Convert an AWG designation embedded in text to mm2, if present."""
    match = _AWG_RE.search(text or "")
    if not match:
        return None
    gauge = (match.group(1) or match.group(2) or "").lstrip("#")
    return AWG_TO_MM2.get(gauge)


def canonical_unit_for(family: str | None) -> str | None:
    """The canonical unit string for a family, or None if the family is unknown."""
    if not family:
        return None
    spec = UNIT_FAMILIES.get(family)
    return spec["canonical"] if spec else None
