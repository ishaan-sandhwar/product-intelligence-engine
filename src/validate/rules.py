"""Cross-field validation rules grounded in engineering relationships.

These checks are deterministic and cost nothing, and they catch the failure mode
that language models are worst at: a value that is individually plausible but
inconsistent with the rest of the record. A motor whose nameplate current does
not agree with its rated power is wrong even though both figures look fine on
their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from src.models import ProductRecord, Severity, ValidationIssue
from src.schema import AttributeSchema

SQRT_3 = math.sqrt(3)


@dataclass(frozen=True)
class Rule:
    """One named consistency check over a record."""

    code: str
    description: str
    applies_to: tuple[str, ...]      # category keys, or ("*",) for all
    check: Callable[[ProductRecord], list[ValidationIssue]]

    def matches(self, category: str | None) -> bool:
        return "*" in self.applies_to or (category or "") in self.applies_to


def _num(record: ProductRecord, key: str) -> float | None:
    value = record.value_of(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _issue(
    code: str,
    message: str,
    severity: Severity = Severity.WARNING,
    field_key: str | None = None,
    suggestion: Any = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field_key=field_key,
        suggestion=suggestion,
        raised_by="rule_engine",
    )


# ------------------------------------------------------------- generic rules
def _temperature_order(record: ProductRecord) -> list[ValidationIssue]:
    low = _num(record, "operating_temp_min_c")
    high = _num(record, "operating_temp_max_c")
    if low is None or high is None or low <= high:
        return []
    return [
        _issue(
            "TEMP_RANGE_INVERTED",
            f"Minimum operating temperature ({low} degC) exceeds the maximum ({high} degC); the two are likely swapped",
            Severity.ERROR,
            "operating_temp_min_c",
            suggestion={"operating_temp_min_c": high, "operating_temp_max_c": low},
        )
    ]


def _dimension_sanity(record: ProductRecord) -> list[ValidationIssue]:
    dims = {k: _num(record, k) for k in ("length_mm", "width_mm", "height_mm")}
    present = {k: v for k, v in dims.items() if v is not None}
    if len(present) < 2:
        return []

    issues: list[ValidationIssue] = []
    largest, smallest = max(present.values()), min(present.values())
    if smallest > 0 and largest / smallest > 500:
        issues.append(
            _issue(
                "DIMENSION_RATIO_EXTREME",
                f"Dimensions span a factor of {largest / smallest:.0f} ({present}); one is probably in the wrong unit",
                Severity.WARNING,
                max(present, key=lambda k: present[k]),
            )
        )

    # A rough density check against declared weight catches unit slips.
    weight = _num(record, "weight_kg")
    if weight and len(present) == 3:
        volume_m3 = (dims["length_mm"] * dims["width_mm"] * dims["height_mm"]) / 1e9
        if volume_m3 > 0:
            density = weight / volume_m3
            if density > 20000:      # denser than tungsten
                issues.append(
                    _issue(
                        "IMPLAUSIBLE_DENSITY",
                        f"Implied density is {density:.0f} kg/m3, above any engineering material; check weight or dimensions",
                        Severity.WARNING,
                        "weight_kg",
                    )
                )
            elif density < 20:       # lighter than foam packaging
                issues.append(
                    _issue(
                        "IMPLAUSIBLE_DENSITY",
                        f"Implied density is only {density:.1f} kg/m3; the weight or dimensions are likely mis-scaled",
                        Severity.INFO,
                        "weight_kg",
                    )
                )
    return issues


def _required_present(schema: AttributeSchema) -> Callable[[ProductRecord], list[ValidationIssue]]:
    def check(record: ProductRecord) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for key in schema.required_keys(record.category):
            av = record.get(key)
            if av is None or av.is_empty:
                spec = schema.spec(key, record.category)
                label = spec.label if spec else key
                issues.append(
                    _issue(
                        "MISSING_REQUIRED",
                        f"{label} is required for this category but is missing",
                        Severity.ERROR,
                        key,
                    )
                )
        return issues

    return check


# --------------------------------------------------------------- motor rules
def _motor_electrical_balance(record: ProductRecord) -> list[ValidationIssue]:
    """P = sqrt(3) . V . I . cos(phi) . eta for three-phase; P = V . I . cos(phi) . eta otherwise."""
    power_kw = _num(record, "power_kw")
    voltage = _num(record, "voltage_v")
    current = _num(record, "current_a")
    if not (power_kw and voltage and current):
        return []

    power_factor = _num(record, "power_factor") or 0.85
    efficiency = 0.90                       # IE2-class assumption when unstated
    phase = str(record.value_of("phase", "Three Phase"))

    if "single" in phase.lower():
        implied_kw = voltage * current * power_factor * efficiency / 1000.0
    elif phase.strip().upper() == "DC":
        implied_kw = voltage * current * efficiency / 1000.0
    else:
        implied_kw = SQRT_3 * voltage * current * power_factor * efficiency / 1000.0

    if implied_kw <= 0:
        return []
    ratio = power_kw / implied_kw
    # Nameplate figures legitimately vary; only flag a real mismatch.
    if 0.65 <= ratio <= 1.5:
        return []

    return [
        _issue(
            "MOTOR_POWER_MISMATCH",
            (
                f"Rated power {power_kw} kW does not agree with {voltage} V x {current} A "
                f"at pf {power_factor} ({phase}), which implies about {implied_kw:.2f} kW"
            ),
            Severity.WARNING if 0.4 <= ratio <= 2.5 else Severity.ERROR,
            "power_kw",
        )
    ]


def _motor_speed_vs_frequency(record: ProductRecord) -> list[ValidationIssue]:
    """Synchronous speed = 120 f / p, so speed must sit near a valid pole count."""
    speed = _num(record, "speed_rpm")
    frequency = _num(record, "frequency_hz") or 50.0
    if not speed or speed <= 0:
        return []

    poles = 120.0 * frequency / speed
    nearest_even = max(2, round(poles / 2) * 2)
    synchronous = 120.0 * frequency / nearest_even
    slip = (synchronous - speed) / synchronous

    if -0.02 <= slip <= 0.12:            # induction slip, or a synchronous machine
        return []
    return [
        _issue(
            "SPEED_FREQUENCY_MISMATCH",
            (
                f"{speed} rpm at {frequency:.0f} Hz implies {poles:.1f} poles; "
                f"the nearest valid pole count gives {synchronous:.0f} rpm "
                f"({slip * 100:+.0f}% slip, outside the normal range)"
            ),
            Severity.WARNING,
            "speed_rpm",
        )
    ]


def _motor_efficiency_plausible(record: ProductRecord) -> list[ValidationIssue]:
    power_factor = _num(record, "power_factor")
    if power_factor is None:
        return []
    if 0.5 <= power_factor <= 1.0:
        return []
    if 50 <= power_factor <= 100:
        return [
            _issue(
                "POWER_FACTOR_AS_PERCENT",
                f"Power factor {power_factor} looks like a percentage; it should be a fraction",
                Severity.WARNING,
                "power_factor",
                suggestion=round(power_factor / 100.0, 3),
            )
        ]
    return [
        _issue(
            "POWER_FACTOR_INVALID",
            f"Power factor {power_factor} is outside the physically possible range 0-1",
            Severity.ERROR,
            "power_factor",
        )
    ]


# ------------------------------------------------------------- bearing rules
def _bearing_geometry(record: ProductRecord) -> list[ValidationIssue]:
    bore = _num(record, "bore_diameter_mm")
    outer = _num(record, "outer_diameter_mm")
    width = _num(record, "bearing_width_mm")
    issues: list[ValidationIssue] = []

    if bore and outer:
        if bore >= outer:
            issues.append(
                _issue(
                    "BEARING_BORE_TOO_LARGE",
                    f"Bore diameter ({bore} mm) must be smaller than the outer diameter ({outer} mm); the two look swapped",
                    Severity.ERROR,
                    "bore_diameter_mm",
                    suggestion={"bore_diameter_mm": outer, "outer_diameter_mm": bore},
                )
            )
        else:
            wall = (outer - bore) / 2.0
            if wall < 1.5:
                issues.append(
                    _issue(
                        "BEARING_WALL_TOO_THIN",
                        f"Radial section is only {wall:.1f} mm; no standard bearing is built that thin",
                        Severity.WARNING,
                        "outer_diameter_mm",
                    )
                )
    if outer and width and width > outer:
        issues.append(
            _issue(
                "BEARING_WIDTH_EXCEEDS_OD",
                f"Width ({width} mm) exceeds the outer diameter ({outer} mm), which is not a bearing geometry",
                Severity.ERROR,
                "bearing_width_mm",
            )
        )

    static = _num(record, "static_load_rating_kn")
    dynamic = _num(record, "dynamic_load_rating_kn")
    if static and dynamic and static > dynamic * 1.6:
        issues.append(
            _issue(
                "BEARING_LOAD_RATINGS_SUSPECT",
                (
                    f"Static rating ({static} kN) far exceeds the dynamic rating ({dynamic} kN); "
                    "for rolling bearings C0 rarely exceeds 1.6 x C, so the two may be swapped"
                ),
                Severity.WARNING,
                "static_load_rating_kn",
            )
        )
    return issues


def _bearing_speed_vs_bore(record: ProductRecord) -> list[ValidationIssue]:
    """Limiting speed falls as bore grows; the n.dm product has a practical ceiling."""
    bore = _num(record, "bore_diameter_mm")
    outer = _num(record, "outer_diameter_mm")
    speed = _num(record, "max_speed_rpm")
    if not (bore and outer and speed):
        return []

    mean_diameter = (bore + outer) / 2.0
    n_dm = speed * mean_diameter
    if n_dm <= 1_200_000:                 # grease-lubricated steel bearings top out near here
        return []
    return [
        _issue(
            "BEARING_SPEED_IMPLAUSIBLE",
            (
                f"Limiting speed {speed:.0f} rpm at mean diameter {mean_diameter:.0f} mm gives "
                f"n.dm = {n_dm:,.0f} mm/min, well above the practical ceiling for a standard bearing"
            ),
            Severity.WARNING,
            "max_speed_rpm",
        )
    ]


# --------------------------------------------------------------- cable rules
# Approximate ampacity ceiling for PVC-insulated copper in free air, per mm2.
COPPER_AMPACITY_PER_MM2 = 8.0
ALUMINIUM_DERATE = 0.78


def _cable_ampacity(record: ProductRecord) -> list[ValidationIssue]:
    area = _num(record, "conductor_size_mm2")
    current = _num(record, "current_rating_a")
    if not (area and current):
        return []

    material = str(record.value_of("conductor_material", "Copper")).lower()
    ceiling = area * COPPER_AMPACITY_PER_MM2
    if "alumin" in material:
        ceiling *= ALUMINIUM_DERATE

    if current <= ceiling * 1.3:
        return []
    return [
        _issue(
            "CABLE_AMPACITY_IMPLAUSIBLE",
            (
                f"{current} A through {area} mm2 of {material} is roughly "
                f"{current / area:.1f} A/mm2, far above the practical limit of about "
                f"{ceiling / area:.0f} A/mm2"
            ),
            Severity.WARNING,
            "current_rating_a",
        )
    ]


def _cable_diameter_vs_cores(record: ProductRecord) -> list[ValidationIssue]:
    area = _num(record, "conductor_size_mm2")
    cores = _num(record, "core_count")
    outer = _num(record, "outer_diameter_mm")
    if not (area and cores and outer):
        return []

    # Conductor bundle area plus a minimum allowance for insulation and sheath.
    conductor_area = area * cores
    min_outer = 2.0 * math.sqrt(conductor_area / math.pi) * 1.4 + 2.0
    if outer >= min_outer * 0.85:
        return []
    return [
        _issue(
            "CABLE_DIAMETER_TOO_SMALL",
            (
                f"Overall diameter {outer} mm cannot contain {cores:.0f} x {area} mm2 conductors "
                f"plus insulation (about {min_outer:.1f} mm minimum)"
            ),
            Severity.WARNING,
            "outer_diameter_mm",
        )
    ]


# ---------------------------------------------------------------- pump rules
def _pump_hydraulic_power(record: ProductRecord) -> list[ValidationIssue]:
    """Hydraulic power = rho.g.Q.H; the motor must be able to supply it."""
    flow = _num(record, "flow_rate_m3h")
    head = _num(record, "head_m")
    motor_kw = _num(record, "motor_power_kw")
    if not (flow and head and motor_kw):
        return []

    hydraulic_kw = (1000.0 * 9.81 * (flow / 3600.0) * head) / 1000.0
    if hydraulic_kw <= 0:
        return []

    efficiency = hydraulic_kw / motor_kw
    if efficiency > 0.92:
        return [
            _issue(
                "PUMP_POWER_TOO_LOW",
                (
                    f"{flow} m3/h at {head} m needs about {hydraulic_kw:.2f} kW of hydraulic power, "
                    f"which a {motor_kw} kW motor cannot deliver (implied efficiency {efficiency * 100:.0f}%)"
                ),
                Severity.ERROR,
                "motor_power_kw",
            )
        ]
    if efficiency < 0.10:
        return [
            _issue(
                "PUMP_POWER_OVERSIZED",
                (
                    f"A {motor_kw} kW motor for {hydraulic_kw:.2f} kW of hydraulic duty implies "
                    f"{efficiency * 100:.0f}% efficiency; one of the three figures is probably wrong"
                ),
                Severity.WARNING,
                "motor_power_kw",
            )
        ]
    return []


def _pump_port_sizes(record: ProductRecord) -> list[ValidationIssue]:
    inlet = _num(record, "inlet_size_mm")
    outlet = _num(record, "outlet_size_mm")
    if not (inlet and outlet) or inlet >= outlet:
        return []
    return [
        _issue(
            "PUMP_PORTS_SUSPECT",
            f"Inlet ({inlet} mm) is smaller than the outlet ({outlet} mm); centrifugal pumps are normally the other way round",
            Severity.INFO,
            "inlet_size_mm",
        )
    ]


# ------------------------------------------------------------ breaker rules
def _breaker_capacity(record: ProductRecord) -> list[ValidationIssue]:
    rated = _num(record, "rated_current_a")
    breaking_ka = _num(record, "breaking_capacity_ka")
    issues: list[ValidationIssue] = []

    if rated and breaking_ka:
        if breaking_ka * 1000.0 <= rated:
            issues.append(
                _issue(
                    "BREAKING_CAPACITY_TOO_LOW",
                    (
                        f"Breaking capacity {breaking_ka} kA is not above the rated current {rated} A; "
                        "a breaker must interrupt far more than it carries"
                    ),
                    Severity.ERROR,
                    "breaking_capacity_ka",
                )
            )

    breaker_type = str(record.value_of("breaker_type", "")).upper()
    if breaker_type == "MCB" and rated and rated > 125:
        issues.append(
            _issue(
                "MCB_RATING_OUT_OF_CLASS",
                f"{rated} A exceeds the MCB range (typically up to 125 A); this is likely an MCCB",
                Severity.WARNING,
                "breaker_type",
                suggestion="MCCB",
            )
        )
    return issues


# ---------------------------------------------------------------- valve rules
def _valve_pressure_vs_size(record: ProductRecord) -> list[ValidationIssue]:
    size = _num(record, "nominal_size_dn")
    pressure = _num(record, "pressure_rating_bar")
    if not (size and pressure):
        return []
    # Large-bore valves cannot also carry very high pressure ratings.
    if size >= 300 and pressure > 250:
        return [
            _issue(
                "VALVE_RATING_IMPLAUSIBLE",
                f"DN{size:.0f} at {pressure} bar is beyond standard flange classes; check whether PSI was read as bar",
                Severity.WARNING,
                "pressure_rating_bar",
            )
        ]
    return []


# ---------------------------------------------------------------- sensor rules
def _sensor_range_vs_thread(record: ProductRecord) -> list[ValidationIssue]:
    """Inductive sensing range scales with barrel diameter; large ranges need large bodies."""
    sensing = _num(record, "sensing_range_mm")
    thread = str(record.value_of("thread_size", ""))
    sensor_type = str(record.value_of("sensor_type", "")).lower()
    if not sensing or "induct" not in sensor_type:
        return []

    barrel = None
    for token in ("M8", "M12", "M18", "M30"):
        if token.lower() in thread.lower():
            barrel = int(token[1:])
            break
    if barrel is None:
        return []

    # Even quadruple-range inductive sensors stay under roughly the barrel diameter.
    if sensing <= barrel * 1.2:
        return []
    return [
        _issue(
            "SENSOR_RANGE_IMPLAUSIBLE",
            f"{sensing} mm sensing range is not achievable in an {thread} inductive barrel",
            Severity.WARNING,
            "sensing_range_mm",
        )
    ]


# ------------------------------------------------------------------ registry
STATIC_RULES: tuple[Rule, ...] = (
    Rule("TEMP_RANGE", "Operating temperature bounds ordered correctly", ("*",), _temperature_order),
    Rule("DIMENSIONS", "Dimensional and density sanity", ("*",), _dimension_sanity),
    Rule("MOTOR_POWER", "Rated power agrees with voltage and current", ("electric_motor",), _motor_electrical_balance),
    Rule("MOTOR_SPEED", "Speed consistent with supply frequency and pole count", ("electric_motor",), _motor_speed_vs_frequency),
    Rule("MOTOR_PF", "Power factor within physical bounds", ("electric_motor",), _motor_efficiency_plausible),
    Rule("BEARING_GEOMETRY", "Bore, outer diameter and width form a real bearing", ("ball_bearing",), _bearing_geometry),
    Rule("BEARING_SPEED", "Limiting speed plausible for the bearing size", ("ball_bearing",), _bearing_speed_vs_bore),
    Rule("CABLE_AMPACITY", "Current rating plausible for the conductor area", ("power_cable",), _cable_ampacity),
    Rule("CABLE_GEOMETRY", "Overall diameter can contain the conductors", ("power_cable",), _cable_diameter_vs_cores),
    Rule("PUMP_POWER", "Motor power covers the hydraulic duty", ("centrifugal_pump",), _pump_hydraulic_power),
    Rule("PUMP_PORTS", "Inlet and outlet sizing conventional", ("centrifugal_pump",), _pump_port_sizes),
    Rule("BREAKER_CAPACITY", "Breaking capacity and class consistent with rating", ("circuit_breaker",), _breaker_capacity),
    Rule("VALVE_RATING", "Pressure rating plausible for the bore", ("industrial_valve",), _valve_pressure_vs_size),
    Rule("SENSOR_RANGE", "Sensing range achievable in the given barrel", ("proximity_sensor",), _sensor_range_vs_thread),
)


def run_rules(
    record: ProductRecord, schema: AttributeSchema
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Run every applicable rule. Returns (issues, summary)."""
    rules: list[Rule] = list(STATIC_RULES)
    rules.append(
        Rule("REQUIRED_FIELDS", "All category-required attributes present", ("*",), _required_present(schema))
    )

    issues: list[ValidationIssue] = []
    fired: list[str] = []
    applicable = 0

    for rule in rules:
        if not rule.matches(record.category):
            continue
        applicable += 1
        try:
            found = rule.check(record)
        except Exception as exc:  # noqa: BLE001 - a broken rule must not stop validation
            found = [
                _issue(
                    "RULE_ERROR",
                    f"Rule {rule.code} could not run: {exc}",
                    Severity.INFO,
                )
            ]
        if found:
            fired.append(rule.code)
            issues.extend(found)

    summary = {
        "rules_applicable": applicable,
        "rules_fired": fired,
        "errors": sum(1 for i in issues if i.severity == Severity.ERROR),
        "warnings": sum(1 for i in issues if i.severity == Severity.WARNING),
        "infos": sum(1 for i in issues if i.severity == Severity.INFO),
    }
    return issues, summary


def attach_issues(record: ProductRecord, issues: list[ValidationIssue]) -> None:
    """Route each issue onto its field, or onto the record when field-less."""
    for issue in issues:
        target = record.get(issue.field_key) if issue.field_key else None
        if target is not None:
            target.issues.append(issue)
        else:
            record.record_issues.append(issue)
