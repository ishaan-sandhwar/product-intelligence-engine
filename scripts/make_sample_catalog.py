"""Generate a deliberately messy demo catalogue at data/raw/sample_catalog.csv.

The point of the mess is that it mirrors what real supplier exports look like:
inconsistent headers, mixed unit systems, values buried in free text, blank
cells, duplicated SKUs and a description column that is sometimes the only place
a specification appears. Deterministic — same file every run.

Usage:  python scripts/make_sample_catalog.py [--rows 40] [--out PATH]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RANDOM_SEED, RAW_DIR  # noqa: E402  (path bootstrap must run first)

# Headers are deliberately non-canonical: this is what the alias resolver is for.
COLUMNS = [
    "Part No",
    "Description",
    "Manufacturer",
    "Prod Category",
    "Rating",
    "Volt",
    "Dimensions",
    "Wt",
    "Temp Range",
    "Certs",
    "Notes",
    "Datasheet URL",
]

MOTORS = [
    ("1LA7-090-4AA10", "Siemens", "1.5 kW 4-pole TEFC induction motor, 400V 50Hz",
     "1.5 kW", "400 V", "IE3, B3 mount, frame 90L"),
    ("M2BAX-112-M-4", "ABB", "3 HP squirrel cage motor 415V 3ph",
     "3 HP", "415V", "IE2 efficiency, foot mounted"),
    ("WEG-W22-5.5", "WEG", "W22 cast iron motor 5.5kW 1465 rpm",
     "5.5kw", "380-415 V", "IP55, insulation class F"),
    ("SEW-DRN90L4", "SEW", "DRN.. AC motor 1.5kW 4 pole", "", "400V", "IE3 premium"),
]

BEARINGS = [
    ("6205-2RS", "SKF", "Deep groove ball bearing 25x52x15mm sealed both sides",
     "14 kN", "", "double rubber seal, C3 clearance"),
    ("22212-EK", "FAG", "Spherical roller bearing tapered bore", "122 kN", "",
     "brass cage"),
    ("NU-2210-E", "NSK", "Cylindrical roller bearing 50 x 90 x 23", "", "",
     "steel cage, C0"),
    ("6205ZZ", "NBC", "ball bearing 25mm bore metal shielded", "13.5kN", "",
     "max 14000 rpm"),
]

VALVES = [
    ("BV-DN50-316", "Kitz", "2 inch ball valve stainless 316, flanged PN16",
     "16 bar", "", "lever operated, PTFE seat"),
    ("GV-DN80-CI", "Zoloto", "Gate valve DN80 cast iron", "10 bar", "",
     "handwheel, rising stem"),
    ("BFV-200-EPDM", "Advance", 'butterfly valve 8" wafer type EPDM seat', "PN10", "",
     "pneumatic actuator option"),
]

SENSORS = [
    ("IME12-04BPSZW2S", "Sick", "Inductive proximity sensor M12, 4mm range, PNP NO",
     "", "10-30 VDC", "IP67, 2m cable"),
    ("E2E-X10ME1", "Omron", "proximity switch 10 mm sensing NPN", "", "12-24VDC",
     "M18 threaded barrel"),
    ("XS612B1PAL2", "Telemecanique", "Inductive sensor 12mm flush", "", "24 V DC",
     "M12 connector, 500 Hz"),
]

CABLES = [
    ("YY-4C-1.5", "Lapp", "4 core 1.5 sq mm flexible control cable PVC",
     "", "300/500V", "unarmoured, grey sheath"),
    ("XLPE-3C-25", "Polycab", "3 core 25 sqmm XLPE armoured aluminium cable",
     "", "1.1 kV", "steel wire armour, FRLS"),
    ("AWG-14-THHN", "Southwire", "THHN single core 14 AWG copper", "", "600 V",
     "nylon jacket, 90C dry"),
]

PUMPS = [
    ("CRI5-10", "Grundfos", "Vertical multistage pump 5 m3/h, 60 m head, 1.5kW",
     "1.5 kW", "415V", "SS304 impeller"),
    ("KDS-32-160", "Kirloskar", "End suction centrifugal pump 32x160", "3 HP", "",
     "cast iron casing, mech seal"),
]

BREAKERS = [
    ("5SL6-C32", "Siemens", "MCB 32A C curve 3 pole 6kA", "32 A", "415V",
     "DIN rail, 3 module"),
    ("NSX250N", "Schneider", "MCCB 250A 4P 50kA breaking", "250A", "690 V",
     "thermal magnetic trip"),
    ("RCCB-63-30", "Havells", "RCCB 63A 30mA 2 pole", "63 A", "240V", "type AC"),
]

FAMILIES = {
    "electric motor": MOTORS,
    "Bearings": BEARINGS,
    "valve": VALVES,
    "sensors": SENSORS,
    "CABLE": CABLES,
    "pump": PUMPS,
    "circuit protection": BREAKERS,
}

DIMENSIONS = ["", "120 x 80 x 60 mm", '4" x 3" x 2"', "L200 W150 H90", "OD 52mm"]
WEIGHTS = ["", "12.5", "3 kg", "0.85kg", "45 lbs", "2,300 g"]
TEMPS = ["", "-20 to 60 C", "-10..+70°C", "0-55 deg C", "-40 to 85"]
CERTS = ["", "CE", "CE, RoHS", "UL listed; CE", "IS 12615, CE", "ATEX, IECEx"]
NOTES = [
    "",
    "see attached datasheet for full specs",
    "MOQ 5 pcs. lead time 3 weeks",
    "supersedes previous part number",
    "ingress protection IP65",
    "warranty 24 months from despatch",
]


def build_rows(target_rows: int, rng: random.Random) -> list[dict[str, str]]:
    """Cycle the seed products, degrading fields at random, until we have enough rows."""
    rows: list[dict[str, str]] = []
    pool = [(cat, item) for cat, items in FAMILIES.items() for item in items]

    while len(rows) < target_rows:
        category, (sku, brand, description, rating, volt, note) = pool[len(rows) % len(pool)]
        suffix = "" if len(rows) < len(pool) else f"-{len(rows) // len(pool) + 1}"

        row = {
            "Part No": f"{sku}{suffix}",
            "Description": description,
            "Manufacturer": brand if rng.random() > 0.12 else "",
            "Prod Category": category if rng.random() > 0.25 else "",
            "Rating": rating,
            "Volt": volt,
            "Dimensions": rng.choice(DIMENSIONS),
            "Wt": rng.choice(WEIGHTS),
            "Temp Range": rng.choice(TEMPS),
            "Certs": rng.choice(CERTS),
            "Notes": f"{note}. {rng.choice(NOTES)}".strip(". "),
            "Datasheet URL": "",
        }
        # A fifth of the catalogue arrives as little more than a part number:
        # the hard case the engine exists for.
        if rng.random() < 0.2:
            for key in ("Rating", "Volt", "Dimensions", "Wt", "Temp Range", "Certs"):
                row[key] = ""
            row["Description"] = description.split(",")[0]
        rows.append(row)

    # One duplicated SKU, because real exports always have one.
    duplicate = dict(rows[3])
    duplicate["Notes"] = "duplicate line from a second supplier sheet"
    rows.append(duplicate)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40, help="how many rows to generate")
    parser.add_argument("--out", type=Path, default=RAW_DIR / "sample_catalog.csv")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    frame = pd.DataFrame(build_rows(args.rows, rng), columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"Wrote {len(frame)} rows to {args.out}")
    print(f"Columns: {', '.join(COLUMNS)}")
    blank_pct = 100.0 * (frame == "").sum().sum() / frame.size
    print(f"Blank cells: {blank_pct:.1f}% — that gap is what the pipeline has to close.")


if __name__ == "__main__":
    main()
