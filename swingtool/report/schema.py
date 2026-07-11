"""Report output contract (report.json).

Every finding traces to: a measured metric, the threshold applied, the cited
reference, and a confidence label. Suppressions are visible (status
not_judged + reason), never silent.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

REPORT_SCHEMA_VERSION = "1.0"

DISCLAIMER = ("Observations from ONE swing, one camera, 30 fps. A single swing "
              "is not a diagnosis. This tool is not a substitute for a "
              "qualified coach or a launch monitor.")


class ReferenceInfo(BaseModel):
    range: Optional[tuple[float, float]] = None
    center: Optional[float] = None
    direction: Optional[str] = None       # for directional rules
    min_change: Optional[float] = None
    unit: str
    source: str
    source_type: Literal["literature", "coaching_standard", "pipeline_reference"]
    tolerance_ours: bool = False
    notes: str = ""


class Comparison(BaseModel):
    metric: str
    measured: Optional[float]
    unit: str
    quality: str
    confidence: float
    reference: Optional[ReferenceInfo] = None
    status: Literal["within_range", "outside_range", "not_judged"]
    hedged: bool = False
    reason: str = ""                      # why not_judged / why hedged


class Finding(BaseModel):
    text: str
    metric: str
    measured: float
    reference: ReferenceInfo
    confidence_label: Literal["normal", "hedged"]
    severity: float                       # |deviation| / range width * confidence
    tier: Literal["biggest opportunity", "worth a look"]


class Highlight(BaseModel):
    text: str
    metric: str
    measured: float
    reference: ReferenceInfo
    hedged: bool = False


class TryThis(BaseModel):
    text: str
    metric: str
    derivation: Literal["arithmetic"]     # never causal/outcome claims
    inputs: dict[str, float]


class ProgressDelta(BaseModel):
    metric: str
    previous: float
    current: float
    note: str                             # deterministic arithmetic note


class NumberFact(BaseModel):
    label: str
    value: str
    note: str = ""


class ReportSource(BaseModel):
    metrics_path: str
    analysis_path: Optional[str] = None
    video: str


class Coverage(BaseModel):
    judged: int
    total: int


class SwingReport(BaseModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    reference_version: str
    source: ReportSource
    disclaimer: str = DISCLAIMER
    coverage: Coverage
    progress: list[ProgressDelta]
    highlights: list[Highlight]
    findings: list[Finding]
    try_this: list[TryThis]
    comparisons: list[Comparison]
    by_the_numbers: list[NumberFact]
    limitations: list[str]
