"""Highlighting of query terms in documents."""

import html
from collections import Counter
from dataclasses import dataclass

from .engine import WORD_RE, AnalyzedQuery, TermMatcher

PALETTE = [
    "#fde68a",
    "#bfdbfe",
    "#bbf7d0",
    "#fbcfe8",
    "#ddd6fe",
    "#fed7aa",
    "#a5f3fc",
    "#d9f99d",
    "#fecaca",
    "#e9d5ff",
]


def term_colors(query: AnalyzedQuery) -> dict[int, str]:
    return {t.term_id: PALETTE[i % len(PALETTE)] for i, t in enumerate(query.terms)}


@dataclass
class Segment:
    text: str
    term_id: int | None = None


def segments(text: str, query_terms: set[int], matcher: TermMatcher) -> list[Segment]:
    """Splits the text into plain and matching segments"""
    result: list[Segment] = []
    last = 0
    for match in WORD_RE.finditer(text):
        matching = [t for t in matcher.terms(match.group()) if t in query_terms]
        if not matching:
            continue
        if match.start() > last:
            result.append(Segment(text[last : match.start()]))
        result.append(Segment(match.group(), matching[0]))
        last = match.end()
    if last < len(text):
        result.append(Segment(text[last:]))
    return result


def term_counts(segs: list[Segment]) -> Counter:
    return Counter(s.term_id for s in segs if s.term_id is not None)


def snippet(
    segs: list[Segment], max_chars: int = 400, context: int = 60
) -> list[Segment]:
    """Keeps windows of text around the matches (in document order)"""
    if sum(len(s.text) for s in segs) <= max_chars:
        return segs

    # Character offsets of each segment, and windows around matches
    offsets, pos = [], 0
    for seg in segs:
        offsets.append(pos)
        pos += len(seg.text)
    windows: list[list[int]] = []
    size = 0
    for seg, start in zip(segs, offsets, strict=True):
        if seg.term_id is None:
            continue
        lo, hi = max(0, start - context), start + len(seg.text) + context
        if windows and lo <= windows[-1][1]:
            size += hi - windows[-1][1]
            windows[-1][1] = hi
        else:
            windows.append([lo, hi])
            size += hi - lo
        if size >= max_chars:
            break
    if not windows:
        windows = [[0, max_chars]]

    out: list[Segment] = []
    for w_lo, w_hi in windows:
        if w_lo > 0:
            out.append(Segment("… "))
        for seg, start in zip(segs, offsets, strict=True):
            end = start + len(seg.text)
            if end <= w_lo or start >= w_hi:
                continue
            if seg.term_id is not None:
                out.append(seg)
            else:
                out.append(Segment(seg.text[max(0, w_lo - start) : w_hi - start]))
    if windows[-1][1] < pos:
        out.append(Segment(" …"))
    return out


def to_html(segs: list[Segment], colors: dict[int, str]) -> str:
    parts = []
    for seg in segs:
        text = html.escape(seg.text)
        if seg.term_id is None:
            parts.append(text)
        else:
            color = colors.get(seg.term_id, PALETTE[0])
            parts.append(
                f'<mark class="qt" data-term="{seg.term_id}" '
                f'style="background:{color}">{text}</mark>'
            )
    return "".join(parts).replace("\n", "<br>")


QUERY_CSS = """
.qs { font-family: ui-monospace, monospace; font-size: 0.85rem; line-height: 1.9; }
.qs .op { color: #7c3aed; font-weight: 600; }
.qs .op.dropped { color: #dc2626; text-decoration: line-through; }
.qs .stop { color: #9ca3af; text-decoration: line-through; }
.qs .unknown { color: #dc2626; text-decoration: underline wavy; }
.qs .error { background: #fecaca; outline: 1px solid #dc2626; border-radius: 2px; }
"""


def query_html(nodes, colors: dict[int, str]) -> str:
    """Annotated echo of a parsed query (words and operators)"""
    from .query_syntax import Resolution, Word

    def render(node) -> str:
        if isinstance(node, Word):
            text = html.escape(node.text)
            if node.resolution is Resolution.TERM:
                color = colors.get(node.term_id, PALETTE[0])
                return (
                    f'<mark class="qt" style="background:{color}" '
                    f'title="term #{node.term_id}">{text}</mark>'
                )
            if node.resolution is Resolution.STOPWORD:
                return f'<span class="stop" title="stop word: ignored">{text}</span>'
            return f'<span class="unknown" title="not in the index">{text}</span>'
        cls = "op dropped" if node.dropped else "op"
        title = html.escape(
            f"dropped: {node.dropped}" if node.dropped else node.label, quote=True
        )
        inner = " ".join(render(child) for child in node.children)
        return (
            f'<span class="{cls}" title="{title}">{html.escape(node.label)}(</span>'
            f'{inner}<span class="{cls}">)</span>'
        )

    return " ".join(render(node) for node in nodes)


def error_html(text: str, start: int, end: int) -> str:
    """The query with the erroneous part outlined"""
    marked = html.escape(text[start:end]) or "&nbsp;"
    return (
        html.escape(text[:start])
        + f'<span class="error">{marked}</span>'
        + html.escape(text[end:])
    )
