"""Shorthand decoding: which attribute each number in a dimension set belongs to.

The distributor writes '12x1/8' for a cut-off wheel and '1/2"x18"' for a sanding
belt. Same syntax, different attributes - the product's form decides, and
getting it wrong publishes a 12-inch width for a 12-inch disc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.extract import domain_extract  # noqa: E402
from src.models import SourceDocument  # noqa: E402
from src.schema import load_schema  # noqa: E402


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _extract(schema, text: str, category: str) -> dict[str, float]:
    source = SourceDocument(kind="text", name="test_desc", text=text)
    found = domain_extract(text, source, schema, category)
    return {key: av.value for key, av in found.items() if not av.is_empty}


def test_round_abrasive_pair_is_diameter_and_thickness(schema):
    """'12x1/8 Cut-Off Wheel' is a 12in disc 1/8in thick, not 12in wide."""
    values = _extract(schema, '12x1/8 Metal Cut-Off Wheel', "abrasive_product")
    assert values.get("abrasive_diameter_in") == pytest.approx(12.0)
    assert "abrasive_width_in" not in values


def test_flat_abrasive_pair_stays_width_by_length(schema):
    """A belt is flat stock, so the same syntax means width by length."""
    values = _extract(schema, 'Diablo 1/2"x18" - Sanding Belt 6pc', "abrasive_product")
    assert values.get("abrasive_width_in") == pytest.approx(0.5)
    assert values.get("abrasive_length_in") == pytest.approx(18.0)
    assert "abrasive_diameter_in" not in values


def test_triple_still_wins_over_pair(schema):
    """'5"x.045"x7/8"' is diameter x thickness x arbor, one dimension set."""
    values = _extract(schema, '5"x.045"x7/8" Cut-Off Wheel', "abrasive_product")
    assert values.get("abrasive_diameter_in") == pytest.approx(5.0)
    assert values.get("disc_thickness_in") == pytest.approx(0.045)
    assert values.get("arbor_size_in") == pytest.approx(0.875)


def test_lone_inch_is_the_working_diameter(schema):
    values = _extract(schema, '5B-332-080 HIOLIT 5" P80', "abrasive_product")
    assert values.get("abrasive_diameter_in") == pytest.approx(5.0)


def test_foot_mark_is_converted_not_copied(schema):
    """A decking board quoted '1x6-16' carries its length in feet."""
    values = _extract(schema, "TREX 1x6x16 SELECT PEBBLE GREY GROOVED", "decking_board")
    assert values.get("board_width_in") == pytest.approx(6.0)


def test_board_stock_reads_thickness_width_length(schema):
    """"1x6-16'" is a one-inch board, six inches wide, sixteen feet long."""
    values = _extract(schema, "1x6-16' Coastline Sq Edge - Vintage Azek PVC Decking",
                      "decking_board")
    assert values.get("board_thickness_in") == pytest.approx(1.0)
    assert values.get("board_width_in") == pytest.approx(6.0)
    assert values.get("board_length_ft") == pytest.approx(16.0)


def test_board_stock_without_a_length_still_reads_the_profile(schema):
    values = _extract(schema, "5/4x6 Square Edge Composite Decking", "decking_board")
    assert values.get("board_width_in") == pytest.approx(6.0)
