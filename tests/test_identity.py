"""Identity resolution: supplier accounts, brands and manufacturers.

These cases are taken from rows that scored badly before the resolver existed,
so a regression here shows up as a distributor's name on a manufacturer line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.loader import row_to_record  # noqa: E402
from src.normalize.identity import (  # noqa: E402
    BrandVocabulary,
    brand_from_account,
    is_distributor_account,
    resolve_batch,
)
from src.schema import load_schema  # noqa: E402


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _record(schema, part_num: str, desc: str, supplier: str, dib: str = "-- No DIB Brand --"):
    """Seed one record from a distributor row, exactly as the loader would."""
    return row_to_record(
        {
            "Mfg_Part_Num": part_num,
            "Part_Desc": desc,
            "E1_Brand": "-- Unbranded --",
            "Unilog_Brand": "-- No Unilog Brand --",
            "DIB_Brand": dib,
            "Part_Manuf": supplier,
        },
        schema,
        source_name="test",
    )


# ------------------------------------------------------------------ accounts
@pytest.mark.parametrize(
    "account, expected",
    [
        ("Jam Industrial Supply LLC", True),
        ("Appliance Dealers Cooperative", True),
        ("Boise Cascade Building Materials", True),
        ("Westwood Lumber Sales", True),
        ("Freud Inc", False),
        ("Mirka Abrasives Inc", False),
        ("Kichler Lighting", False),
        ("Velux America Inc", False),
    ],
)
def test_distributor_accounts_are_recognised(account, expected):
    assert is_distributor_account(account) is expected


@pytest.mark.parametrize(
    "account, expected",
    [
        ("Freud Inc", "Freud"),
        ("Mirka Abrasives Inc", "Mirka"),
        ("Leviton Mfg Co", "Leviton"),
        ("Velux America Inc", "Velux"),
        ("Kichler Lighting", "Kichler"),
        ("Milwaukee Accessory", "Milwaukee"),
    ],
)
def test_account_reduces_to_trade_name(account, expected):
    assert brand_from_account(account) == expected


# -------------------------------------------------------------- vocabulary
@pytest.mark.parametrize(
    "description, expected",
    [
        ('DCB518ASTS06G Diablo 1/2"x18" - Sanding Belt 6pc', "Diablo"),
        ("3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box", "3M"),
        ('5B-332-080 HIOLIT 5" P80', "Mirka"),          # sub-brand maps to its owner
        ("TREX 1x6x16 SELECT PEBBLE GREY GROOVED", "Trex"),
    ],
)
def test_brand_found_in_description(description, expected):
    vocab = BrandVocabulary.build()
    hit = vocab.find(description)
    assert hit is not None and hit[0] == expected


def test_single_character_tokens_never_become_brands():
    """'U S Lumber' must not register 'u' or 's' as a brand alias."""
    vocab = BrandVocabulary.build()
    assert all(len(alias) > 1 for alias in vocab.aliases)


# ----------------------------------------------------------------- resolver
def test_distributor_account_never_reaches_manufacturer(schema):
    """A 3M abrasive bought through a distributor is manufactured by 3M."""
    record = _record(
        schema, "3MABR-7100075678",
        "3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box",
        "Jam Industrial Supply LLC (JAMIN)",
    )
    resolve_batch([record], schema)

    assert record.value_of("brand") == "3M"
    assert record.value_of("manufacturer") == "3M"
    assert "Jam Industrial" not in str(record.value_of("manufacturer"))
    # The account itself is still published, just on the right line.
    assert record.value_of("supplier") == "Jam Industrial Supply LLC"


def test_manufacturer_account_reaches_manufacturer(schema):
    """Freud Inc is a maker, so it belongs on the manufacturer line as supplied."""
    record = _record(
        schema, "DCB518ASTS06G",
        'DCB518ASTS06G Diablo 1/2"x18" - Sanding Belt 6pc',
        "Freud Inc (2435)",
    )
    resolve_batch([record], schema)

    assert record.value_of("brand") == "Diablo"
    assert record.value_of("manufacturer") == "Freud Inc"


def test_brand_column_outranks_description(schema):
    record = _record(
        schema, "X-1", "3M 775L Stikit Film P150", "Freud Inc", dib="Diablo"
    )
    resolve_batch([record], schema)
    assert record.value_of("brand") == "Diablo"


def test_resolution_never_overwrites_a_published_value(schema):
    record = _record(schema, "X-2", "TREX 1x6x16 GROOVED", "Parksite", dib="TREX")
    before = record.value_of("brand")
    resolve_batch([record], schema)
    assert record.value_of("brand") == before
