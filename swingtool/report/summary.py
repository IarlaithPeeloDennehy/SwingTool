"""Readable console rendering of a SwingReport. Plain ASCII (Windows-pipe safe)."""

from __future__ import annotations

from swingtool.report.schema import SwingReport


def _src(source: str) -> str:
    return source.split(":")[0].split(";")[0]


def format_report(r: SwingReport) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"SWING REPORT  ({r.source.video})")
    lines.append(f"Confidently judged {r.coverage.judged} of {r.coverage.total} "
                 f"measurements on this clip.")

    if r.progress:
        lines.append("")
        lines.append("PROGRESS (vs your last swing)")
        for p in r.progress:
            lines.append(f"  {p.metric}: {p.previous:.1f} -> {p.current:.1f}  ({p.note})")

    if r.highlights:
        lines.append("")
        lines.append("HIGHLIGHTS")
        for h in r.highlights:
            hedge = "  [view-dependent]" if h.hedged else ""
            lines.append(f"  + {h.text}{hedge}")
            lines.append(f"      source: {_src(h.reference.source)}")

    if r.findings:
        lines.append("")
        lines.append("WORTH A LOOK")
        for i, f in enumerate(r.findings, 1):
            tag = f" ({f.tier})" if f.tier == "biggest opportunity" else ""
            hedge = " [hedged]" if f.confidence_label == "hedged" else ""
            lines.append(f"  {i}. {f.text}{tag}{hedge}")
            lines.append(f"      source: {_src(f.reference.source)}")
        for tt in r.try_this:
            lines.append(f"      Try this: {tt.text}")

    not_judged = [c for c in r.comparisons if c.status == "not_judged" and c.measured is not None]
    if not_judged:
        lines.append("")
        lines.append("MEASURED, NOT JUDGED")
        for c in not_judged:
            unit = f" {c.unit}" if c.unit else ""
            lines.append(f"  - {c.metric}: {c.measured}{unit}")
            lines.append(f"      why not judged: {c.reason}")

    if r.by_the_numbers:
        lines.append("")
        lines.append("YOUR SWING BY THE NUMBERS")
        for n in r.by_the_numbers:
            note = f"  ({n.note})" if n.note else ""
            lines.append(f"  {n.label}: {n.value}{note}")

    lines.append("")
    lines.append("WHAT WE'D NEED TO MEASURE NEXT")
    for lim in r.limitations:
        lines.append(f"  - {lim}")

    lines.append("")
    lines.append(r.disclaimer)
    return "\n".join(lines)
