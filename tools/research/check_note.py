#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""road-to-master research note-quality gate runner (deterministic, OFFLINE, stdlib-only).

Mechanically validates a research note (`README.md` + sibling `ledger.json`) against the
note contract defined under `tools/research/`. Every sub-gate is evaluated on every run and
the verdict is the UNION of all failing gates (no first-failure short-circuit). Exit code is
non-zero iff at least one sub-gate fails. Nothing-ran (missing note/ledger) is a FAIL, never a
silent pass.

Usage:
  check_note.py --note <domain>/<Topic>   # resolve under domains/ (falls back to fixtures/)
  check_note.py --fixture <name>          # resolve under tools/research/fixtures/<name>
  check_note.py --path <dir>              # validate an arbitrary note directory
  check_note.py --all-fixtures            # run the red/green fixture suite (manifest-driven)

See tools/research/README.md for the contract.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "fixtures")
# tools/research/check_note.py -> repo root is two dirs up.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DOMAINS_DIR = os.path.join(REPO_ROOT, "domains")
SCHEMA_PATH = os.path.join(SCRIPT_DIR, "ledger.schema.json")

KIND_ENUM = (
    "paper-claim",
    "evidence-supported",
    "reasonable-inference",
    "unproven-or-doubtful",
)
SANCTIONED_LITERALS = ("unknown", "unavailable")
TOPIC_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# AC-17 whitelist: an evidence-supported code claim's source_url must be path-shaped
# (relative `./x`, absolute `/x`, or a multi-segment `dir/file`) — never a URL scheme.
FILE_PATH_RE = re.compile(r"^(\.?/|[^:]+/)")
ISO_8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

AC_HEADING_RE = re.compile(r"^##\s+\U0001F4C7\s+Academic Context\b")
CA_HEADING_RE = re.compile(r"^##\s+\U0001F9EA\s+Critical Assessment\b")
RELATED_HEADING_RE = re.compile(r"^##\s+\U0001F517\s+Related notes\b")
LEVEL2_RE = re.compile(r"^##\s+\S")

# Repo bilingual convention (EN-default `README.md` + `README.zh-TW.md` variant sharing one
# `ledger.json`): the first line under the H1 in BOTH files is a language switcher blockquote,
# e.g. `> **English** | [繁體中文](./README.zh-TW.md)` or `> [English](./README.md) | **繁體中文**`.
# This is navigation chrome, not first-principles note body prose, so the structure gate must
# not count it as content appearing before the Academic Context section.
LANGUAGE_SWITCHER_RE = re.compile(
    r"^>\s*(?:\*\*English\*\*|\[English\]\([^)]*\))\s*\|\s*"
    r"(?:\*\*繁體中文\*\*|\[繁體中文\]\([^)]*\))\s*$"
)

# Secret detectors (regex set: AWS / GitHub / PEM / generic api-key assignment).
SECRET_PATTERNS = [
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github-token", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("pem-private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "api-key-assignment",
        re.compile(
            r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"
        ),
    ),
]

# Interrogation sub-headings that make a Critical Assessment section non-trivial.
INTERROGATION_MARKERS = [
    "problem realness",
    "problem-realness",
    "baseline",
    "ablation",
    "dataset",
    "metric",
    "novelty",
    "repackaging",
    "self-defined benchmark",
    "射箭畫靶",  # 射箭畫靶
    "problem solved",
    "problem actually solved",
    "real-world",
]
CA_MIN_CHARS = 400

# Canonical ordered list of sub-gate names (printed in this order).
GATE_ORDER = [
    "structure:academic-context",
    "structure:section-order",
    "structure:related-notes",
    "structure:substance",
    "structure:critical-assessment",
    "ledger:schema",
    "ledger:coverage",
    "ledger:code-source-not-file",
    "ledger:blank-or-unsourced-cell",
    "ledger:kind-enum",
    "ledger:no-endorsement",
    "link",
    "figure",
    "secret",
    "name",
    "domain",
    "duplicate",
    "attestation",
]


class Result(object):
    """Holds per-gate PASS/FAIL + concrete detail, never silently swallowing a reason."""

    def __init__(self):
        self.gates = {}  # name -> (ok: bool, detail: str)
        # (gate_name, reason) when evaluation could not proceed at all: "nothing-ran" for a
        # missing note/ledger, "name" for an unsafe --note/--fixture segment rejected BEFORE
        # any filesystem access. Never a silent pass.
        self.fatal = None

    @property
    def nothing_ran(self):
        return self.fatal[1] if self.fatal and self.fatal[0] == "nothing-ran" else None

    @nothing_ran.setter
    def nothing_ran(self, reason):
        self.fatal = ("nothing-ran", reason)

    def record(self, name, ok, detail=""):
        self.gates[name] = (bool(ok), detail)

    def failing(self):
        if self.fatal is not None:
            return [self.fatal[0]]
        return sorted(name for name, (ok, _d) in self.gates.items() if not ok)

    def is_pass(self):
        return len(self.failing()) == 0


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------
def find_domains():
    """Domain enum is DERIVED from live domains/ dirs (single source of truth)."""
    if not os.path.isdir(DOMAINS_DIR):
        return set()
    return set(
        d
        for d in os.listdir(DOMAINS_DIR)
        if os.path.isdir(os.path.join(DOMAINS_DIR, d)) and not d.startswith(".")
    )


def level2_headings(lines):
    """Return list of (index, text) for every level-2 (## ) heading."""
    out = []
    for i, line in enumerate(lines):
        if LEVEL2_RE.match(line):
            out.append((i, line.strip()))
    return out


def section_range(lines, heading_regex):
    """Return (start, end) line range for the level-2 section whose heading matches, or None."""
    start = None
    for i, line in enumerate(lines):
        if heading_regex.match(line):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if LEVEL2_RE.match(lines[i]):
            end = i
            break
    return (start, end)


def normalize_field(name):
    return re.sub(r"\s+", "-", name.strip().lower())


def parse_field_value_table(lines):
    """Parse the FIRST markdown table whose header is `| Field | Value |`.

    Returns list of (field, value) data rows. Value may be an empty string (blank cell).
    """
    return parse_field_value_table_result(lines)[1]


def parse_field_value_table_result(lines):
    """Return (found_header, rows) for the first `| Field | Value |` table in lines."""
    header_idx = None
    for i, line in enumerate(lines):
        cells = split_table_row(line)
        if cells is None:
            continue
        norm = [c.strip().lower() for c in cells]
        if len(norm) >= 2 and norm[0] == "field" and norm[1] == "value":
            header_idx = i
            break
    if header_idx is None:
        return False, []
    rows = []
    # Skip the separator row (|-|-|) then read data rows until the table ends.
    j = header_idx + 1
    if j < len(lines) and is_separator_row(lines[j]):
        j += 1
    while j < len(lines):
        cells = split_table_row(lines[j])
        if cells is None:
            break
        field = cells[0].strip()
        value = cells[1].strip() if len(cells) > 1 else ""
        if field:
            rows.append((field, value))
        j += 1
    return True, rows


def parse_academic_context_table(lines):
    """Return (found_header, rows) for the Academic Context section's value table.

    The contract requires the table to live inside `## 📇 Academic Context`, not merely
    somewhere else in the note. If the heading is missing, fall back to the first table
    anywhere so ledger coverage can still aggregate table-cell failures in multi-fault
    fixtures instead of hiding them behind the structure failure.
    """
    ac = section_range(lines, AC_HEADING_RE)
    if ac is None:
        return parse_field_value_table_result(lines)
    return parse_field_value_table_result(lines[ac[0]:ac[1]])


def split_table_row(line):
    """Split a markdown table row into cell strings, or None if not a table row."""
    s = line.strip()
    if not s.startswith("|"):
        return None
    inner = s.strip("|")
    return inner.split("|")


def is_separator_row(line):
    s = line.strip().strip("|")
    parts = [p.strip() for p in s.split("|")]
    return all(re.fullmatch(r":?-+:?", p or "") for p in parts) and len(parts) > 0


# Markdown link regexes. Image links (`![alt](x)`) are owned by the figure gate; text links
# (`[t](x)`) by the link gate. The negative-lookbehind keeps them from overlapping.
IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TEXT_LINK_RE = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")


def is_relative_target(target):
    t = target.strip()
    if not t:
        return False
    if t.startswith("#"):
        return False
    if t.startswith("mailto:"):
        return False
    if URL_SCHEME_RE.match(t):
        return False
    return True


def clean_link_path(target):
    t = target.strip()
    # Drop any markdown title and URL fragment/anchor.
    t = t.split(" ", 1)[0]
    t = t.split("#", 1)[0]
    return t


# --------------------------------------------------------------------------------------
# Gate evaluation
# --------------------------------------------------------------------------------------
def evaluate(note_dir):
    res = Result()
    readme_path = os.path.join(note_dir, "README.md")
    ledger_path = os.path.join(note_dir, "ledger.json")

    if not os.path.isfile(readme_path):
        res.nothing_ran = "missing note file: %s" % rel(readme_path)
        return res
    if not os.path.isfile(ledger_path):
        if _is_existing_domain_note(note_dir):
            res.fatal = (
                "duplicate",
                "DUPLICATE: target %s already exists with a non-empty README.md; caller must "
                "decide skip/merge (never silently overwrite)" % rel(note_dir),
            )
            return res
        res.nothing_ran = "missing ledger file: %s" % rel(ledger_path)
        return res

    with open(readme_path, "r", encoding="utf-8") as fh:
        note_text = fh.read()
    lines = note_text.splitlines()

    try:
        with open(ledger_path, "r", encoding="utf-8") as fh:
            ledger = json.load(fh)
    except (ValueError, UnicodeDecodeError) as exc:
        res.nothing_ran = "invalid ledger JSON (%s): %s" % (rel(ledger_path), exc)
        return res

    # A non-object ledger (e.g. a bare array) is a shape violation, not a crash: the schema
    # gate reports it against the RAW value; the semantic gates evaluate against an empty
    # dict so the union verdict still names every gate the malformed ledger cannot satisfy.
    raw_ledger = ledger
    if not isinstance(ledger, dict):
        ledger = {}
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        entries = []

    _gate_structure(res, lines, ledger, entries)
    _gate_schema(res, raw_ledger)
    _gate_ledger(res, lines, ledger, entries, note_text, note_dir)
    _gate_link(res, note_text, note_dir)
    _gate_figure(res, note_text, note_dir, entries)
    _gate_secret(res, note_text, ledger)
    _gate_name(res, ledger, note_dir)
    _gate_domain(res, ledger, note_dir)
    _gate_duplicate(res, ledger, note_dir)
    _gate_attestation(res, ledger, note_dir)
    return res


def _section_text(lines, rng):
    if rng is None:
        return ""
    return "\n".join(lines[rng[0]:rng[1]])


def _gate_structure(res, lines, ledger, entries):
    ac = section_range(lines, AC_HEADING_RE)
    ca = section_range(lines, CA_HEADING_RE)
    rel_rng = section_range(lines, RELATED_HEADING_RE)
    ac_table_found, ac_rows = parse_academic_context_table(lines)

    # academic-context: exact heading present AND the required academic-value table lives
    # inside it. A heading without `| Field | Value |` rows is a false green.
    ac_ok = ac is not None and ac_table_found and len(ac_rows) > 0
    if ac is None:
        ac_detail = "missing required section '## \U0001F4C7 Academic Context'"
    elif not ac_table_found:
        ac_detail = (
            "Academic Context section is missing the required academic-value table "
            "with header '| Field | Value |'"
        )
    elif not ac_rows:
        ac_detail = (
            "Academic Context table has no data rows; expected at least one Field/Value row"
        )
    else:
        ac_detail = ""
    res.record(
        "structure:academic-context",
        ac_ok,
        ac_detail,
    )

    # related-notes: exact heading present (may be empty).
    res.record(
        "structure:related-notes",
        rel_rng is not None,
        "" if rel_rng is not None else "missing required section '## \U0001F517 Related notes'",
    )

    # critical-assessment: present AND non-trivial.
    if ca is None:
        res.record(
            "structure:critical-assessment",
            False,
            "missing required section '## \U0001F9EA Critical Assessment'",
        )
    else:
        body = _section_text(lines, ca)
        # Strip the heading line before measuring substance.
        section_lines = body.splitlines()[1:]
        body_wo_heading = "\n".join(section_lines)
        chars = len(body_wo_heading.strip())
        # The escape hatch below CA_MIN_CHARS is an actual interrogation SUB-HEADING line
        # (###+), not a marker word buried in prose ("The baseline looks fine." must FAIL).
        sub_headings = [
            ln for ln in section_lines if re.match(r"^#{3,6}\s+\S", ln)
        ]
        has_marker_heading = any(
            any(m in h.lower() for m in INTERROGATION_MARKERS) for h in sub_headings
        )
        ok = chars >= CA_MIN_CHARS or has_marker_heading
        res.record(
            "structure:critical-assessment",
            ok,
            "" if ok else (
                "Critical Assessment is trivial: %d chars (< %d) and no interrogation "
                "sub-heading (### line naming problem-realness / baseline-ablation-dataset-"
                "metric / novelty-vs-repackaging / problem-solved / real-world-relevance)"
                % (chars, CA_MIN_CHARS)
            ),
        )

    # section-order: canonical order is Academic Context -> body -> Critical Assessment ->
    # Related notes. AC must be the first substantive section/content after the title; body
    # content must live between Academic Context and Critical Assessment; Related is last.
    order_problems = []
    positions = []
    if ac is not None:
        positions.append(("academic-context", ac[0]))
    if ca is not None:
        positions.append(("critical-assessment", ca[0]))
    if rel_rng is not None:
        positions.append(("related-notes", rel_rng[0]))
    ordered_by_line = [n for n, _ln in sorted(positions, key=lambda p: p[1])]
    expected_seq = [n for n in ["academic-context", "critical-assessment", "related-notes"]
                    if n in dict(positions)]
    if ordered_by_line != expected_seq:
        order_problems.append(
            "anchor sections out of order: got %s, expected %s" % (ordered_by_line, expected_seq)
        )
    h2 = level2_headings(lines)
    if ac is not None and h2 and h2[0][0] != ac[0]:
        before = [t for i, t in h2 if i < ac[0]]
        order_problems.append(
            "first-principles body section(s) %s precede '## \U0001F4C7 Academic Context' "
            "(it must be the first section)" % before
        )
    if ac is not None and _has_body_content(lines[:ac[0]]):
        order_problems.append(
            "first-principles body prose appears before '## \U0001F4C7 Academic Context' "
            "(body content must be between Academic Context and Critical Assessment)"
        )
    if ac is not None and ca is not None and not _has_body_content(lines[ac[1]:ca[0]]):
        order_problems.append(
            "no first-principles body content found between Academic Context and Critical Assessment"
        )
    if rel_rng is not None:
        after = [t for i, t in h2 if i > rel_rng[0]]
        if after:
            order_problems.append(
                "section(s) %s appear after '## \U0001F517 Related notes' "
                "(it must be the last section)" % after
            )
    res.record(
        "structure:section-order",
        len(order_problems) == 0,
        "" if not order_problems else (
            "sections out of order (expected Academic Context -> body -> Critical Assessment "
            "-> Related notes): " + "; ".join(order_problems)
        ),
    )

    # substance: only the canonical first-principles body region counts. Prose/math before
    # Academic Context is an order violation, not acceptable body substance.
    if ac is None or ca is None or order_problems:
        res.record("structure:substance", True, "skipped (canonical body region unavailable)")
        return
    if ac is not None and ca is not None and ac[1] <= ca[0]:
        body_lines = lines[ac[1]:ca[0]]
    else:
        body_lines = []
    body_text = "\n".join(body_lines)
    venue_kind = _venue_kind(lines)
    has_math = "$" in body_text
    has_code = "```" in body_text
    has_table = any(is_table_line(ln) for ln in body_lines)
    has_subheading = any(re.match(r"^#{2,4}\s+\S", ln) for ln in body_lines)
    if venue_kind in ("blog", "tech-report"):
        ok = has_subheading or has_math or has_code or has_table
        detail = "" if ok else "blog/tech-report note has no explained concept section in the body"
    else:
        ok = has_math or has_code or has_table
        detail = "" if ok else (
            "technical note lacks any equation ($...$), algorithm/code block (```), or table "
            "in the first-principles body"
        )
    res.record("structure:substance", ok, detail)


def is_table_line(line):
    cells = split_table_row(line)
    return cells is not None and len(cells) >= 2


def _has_body_content(lines):
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("# "):
            continue
        if LANGUAGE_SWITCHER_RE.match(s):
            # Bilingual-convention navigation chrome, not first-principles body prose.
            continue
        return True
    return False


def _venue_kind(lines):
    for field, value in parse_academic_context_table(lines)[1]:
        if normalize_field(field) in ("venue-kind", "kind", "source-kind"):
            return value.strip().lower()
    return "paper"


# --------------------------------------------------------------------------------------
# ledger:schema — hand-rolled stdlib validation of ledger.json against ledger.schema.json
# (NFR E1: the schema is the machine-checkable contract, not decoration). The tiny
# validator enforces the structural keywords the schema uses: `type`, `required`,
# `additionalProperties`, `minLength`. The value-semantic keywords are DELIBERATELY
# delegated to the dedicated gates that own them, so each violation is reported exactly
# once under its precise gate name: `enum` (kind) -> ledger:kind-enum, `const`
# (no_untrusted_code_executed) -> attestation, `pattern` (topic) -> name.
def _schema_type_ok(value, expected):
    types = expected if isinstance(expected, list) else [expected]
    for t in types:
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
    return False


def _is_iso8601_timestamp(value):
    if not isinstance(value, str) or not ISO_8601_RE.match(value):
        return False
    try:
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _validate_shape(value, schema, path, problems):
    expected_type = schema.get("type")
    if expected_type is not None and not _schema_type_ok(value, expected_type):
        problems.append(
            "%s: expected type %s, got %s (%r)"
            % (path, expected_type, type(value).__name__, value)
        )
        return  # type mismatch makes deeper keywords meaningless for this node
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                problems.append("%s: missing required field '%s'" % (path, req))
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(k for k in value if k not in props)
            if unknown:
                problems.append("%s: unknown field(s) %s not in the schema" % (path, unknown))
        for key, sub in props.items():
            if key in value:
                _validate_shape(value[key], sub, "%s.%s" % (path, key), problems)
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _validate_shape(item, items, "%s[%d]" % (path, i), problems)
    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            problems.append("%s: %r is shorter than minLength %d" % (path, value, min_len))
        if schema.get("format") == "date-time" and not _is_iso8601_timestamp(value):
            problems.append("%s: %r is not an ISO-8601 date-time" % (path, value))


def _gate_schema(res, ledger):
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
    except (OSError, ValueError) as exc:
        res.record(
            "ledger:schema",
            False,
            "cannot load contract schema %s: %s" % (rel(SCHEMA_PATH), exc),
        )
        return
    problems = []
    _validate_shape(ledger, schema, "ledger", problems)
    res.record(
        "ledger:schema",
        len(problems) == 0,
        "" if not problems else (
            "ledger.json violates %s: %s" % (rel(SCHEMA_PATH), "; ".join(problems))
        ),
    )


def _coverage_lines(lines, note_dir):
    """Return the note variant whose language matches the ledger, for coverage matching.

    Under the repo bilingual convention (EN-default `README.md` + `README.zh-TW.md` variant
    sharing one machine-readable `ledger.json`), the ledger's `quoted_snippet`/`claim_text`
    are authored against the SOURCE (Traditional-Chinese) prose — and its positional
    `note-section:body-*`/`critical-*` targets enumerate the source paragraphs. When a
    `README.zh-TW.md` sibling exists, coverage is therefore validated against it, so a faithful
    English translation is not falsely flagged as uncovered. This does NOT weaken the gate: a
    ledger row whose snippet appears in neither the note's prose still fails, and every other
    structural gate (academic-context, section-order, critical-assessment, substance, …) still
    runs against the English `README.md`. Notes without a zh-TW sibling are unaffected.
    """
    sibling = os.path.join(note_dir, "README.zh-TW.md")
    if os.path.isfile(sibling):
        try:
            with open(sibling, "r", encoding="utf-8") as fh:
                return fh.read().splitlines()
        except (OSError, UnicodeDecodeError):
            return lines
    return lines


def _gate_ledger(res, lines, ledger, entries, note_text, note_dir):
    # Coverage (Academic Context cells + body/critical claim units) is validated against the
    # note variant matching the ledger's language; see `_coverage_lines`. Every other check in
    # this gate that reasons about the default note stays on the English `lines`.
    cov_lines = _coverage_lines(lines, note_dir)
    rows = parse_academic_context_table(cov_lines)[1]
    entries_by_target = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("target"), str):
            entries_by_target.setdefault(e["target"].strip(), []).append(e)

    # coverage + blank-or-unsourced-cell.
    missing_cov = []
    blank_cells = []
    for field, value in rows:
        v = value.strip()
        if v == "":
            blank_cells.append(field)
            continue
        if v.lower() in SANCTIONED_LITERALS:
            continue  # sanctioned no-evidence token, no ledger entry required
        want = "table-cell:%s" % normalize_field(field)
        matches = entries_by_target.get(want, [])
        if not matches:
            missing_cov.append("%s (expected ledger target '%s')" % (field, want))
            continue
        value_matches = [e for e in matches if _entry_matches_expected_value(e, v)]
        if not value_matches:
            missing_cov.append(
                "%s (ledger target '%s' has no claim_text/quoted_snippet matching cell value %r)"
                % (field, want, v)
            )
            continue
        unsourced_value_matches = [e for e in value_matches if not _has_provenance_source(e)]
        if unsourced_value_matches:
            missing_cov.append(
                "%s (ledger target '%s' has matching ledger row(s) with no source_url "
                "provenance: %s)"
                % (field, want, _entry_ids(unsourced_value_matches))
            )
    missing_cov.extend(_body_claim_coverage_gaps(cov_lines, entries))
    missing_cov.extend(_critical_assessment_claim_coverage_gaps(cov_lines, entries))
    res.record(
        "ledger:coverage",
        len(missing_cov) == 0,
        "" if not missing_cov else "note claims with no ledger entry: " + "; ".join(missing_cov),
    )
    res.record(
        "ledger:blank-or-unsourced-cell",
        len(blank_cells) == 0,
        "" if not blank_cells else (
            "blank Academic Context cell(s) %s: write the literal 'unknown'/'unavailable' to make "
            "the absence explicit" % blank_cells
        ),
    )

    # kind-enum.
    bad_kinds = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        k = e.get("kind")
        if k not in KIND_ENUM:
            bad_kinds.append("%s=%r" % (e.get("claim_id", "?"), k))
    res.record(
        "ledger:kind-enum",
        len(bad_kinds) == 0,
        "" if not bad_kinds else (
            "ledger kind outside 4-value taxonomy %s: %s" % (list(KIND_ENUM), "; ".join(bad_kinds))
        ),
    )

    # code-source-not-file: an evidence-supported CODE claim is accepted ONLY when its
    # source_url is a filesystem path (FILE_PATH_RE, never a URL scheme) that resolves to a
    # real file INSIDE the inspected clone — realpath-confined to the note dir or repo root
    # subtree, so neither an absolute path (/etc/hosts) nor a `../` traversal can escape.
    # null / pathless / nonexistent / escaping sources all FAIL — a code claim can only be
    # "supported" by the file that was read.
    bad_code = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("kind") != "evidence-supported":
            continue
        target = (e.get("target") or "")
        if "code" not in target.lower():
            continue
        src = e.get("source_url")
        claim = e.get("claim_id", "?")
        if not isinstance(src, str) or not src.strip():
            bad_code.append("%s -> %r is not a file path" % (claim, src))
            continue
        s = src.strip()
        if URL_SCHEME_RE.match(s):
            bad_code.append("%s -> %s is a URL, not a clone file path" % (claim, s))
            continue
        if not FILE_PATH_RE.match(s):
            bad_code.append(
                "%s -> %r is not path-shaped (must match ^(\\.?/|[^:]+/))" % (claim, s)
            )
            continue
        resolved, escaped = _resolve_inside_clone(s, note_dir)
        if resolved is None:
            if escaped:
                bad_code.append(
                    "%s -> %s resolves OUTSIDE the inspected clone (note dir / repo root "
                    "subtree, realpath-confined); a file outside the clone cannot be "
                    "'the code that was read'" % (claim, s)
                )
            else:
                bad_code.append(
                    "%s -> %s does not resolve to a file in the inspected clone "
                    "(tried note dir and repo root)" % (claim, s)
                )
    res.record(
        "ledger:code-source-not-file",
        len(bad_code) == 0,
        "" if not bad_code else (
            "evidence-supported code claim(s) not backed by a read clone file: " + "; ".join(bad_code)
        ),
    )

    # no-endorsement.
    kinds_present = set(e.get("kind") for e in entries if isinstance(e, dict))
    has_verdict_kind = bool(kinds_present & {"reasonable-inference", "unproven-or-doubtful"})
    ca_rng = section_range(lines, CA_HEADING_RE)
    ca_text = _section_text(lines, ca_rng).lower() if ca_rng else ""
    explicit_verdict = bool(
        re.search(r"no material weaknesses", ca_text) and "evidence" in ca_text
    )
    ok = has_verdict_kind or explicit_verdict
    res.record(
        "ledger:no-endorsement",
        ok,
        "" if ok else (
            "uncritical restatement: ledger has only paper-claim/evidence-supported entries and no "
            "'reasonable-inference'/'unproven-or-doubtful' entry, and no explicit 'no material "
            "weaknesses found, evidence: ...' verdict in the Critical Assessment"
        ),
    )


def _body_claim_coverage_gaps(lines, entries):
    """Return first-principles body coverage gaps, using deterministic claim units."""
    ac = section_range(lines, AC_HEADING_RE)
    ca = section_range(lines, CA_HEADING_RE)
    if ac is None or ca is None or ac[1] > ca[0]:
        return []
    rel_rng = section_range(lines, RELATED_HEADING_RE)
    if rel_rng is not None and rel_rng[0] < ca[0]:
        # The structure gate owns this invalid order. Treating Related-notes prose as body
        # claims would create a second, misleading coverage failure for a section-order fixture.
        return []
    body_lines = lines[ac[1]:ca[0]]
    return _claim_unit_coverage_gaps(
        section_label="body",
        target_prefix="body",
        section_lines=body_lines,
        entries=entries,
    )


def _critical_assessment_claim_coverage_gaps(lines, entries):
    """Return Critical Assessment coverage gaps for non-trivial critique claims.

    The structure gate owns a missing/trivial Critical Assessment. Once the section is
    structurally valid, its critique paragraphs are note claims too and require value-led
    ledger coverage under `note-section:critical-*` targets.
    """
    ca = section_range(lines, CA_HEADING_RE)
    if ca is None or not _critical_assessment_nontrivial(lines, ca):
        return []
    rel_rng = section_range(lines, RELATED_HEADING_RE)
    if rel_rng is not None and rel_rng[0] < ca[0]:
        return []
    ca_lines = lines[ca[0] + 1:ca[1]]
    return _claim_unit_coverage_gaps(
        section_label="critical assessment",
        target_prefix="critical",
        section_lines=ca_lines,
        entries=entries,
    )


def _critical_assessment_nontrivial(lines, ca_rng):
    section_lines = _section_text(lines, ca_rng).splitlines()[1:]
    body_wo_heading = "\n".join(section_lines)
    chars = len(body_wo_heading.strip())
    sub_headings = [ln for ln in section_lines if re.match(r"^#{3,6}\s+\S", ln)]
    has_marker_heading = any(
        any(m in h.lower() for m in INTERROGATION_MARKERS) for h in sub_headings
    )
    return chars >= CA_MIN_CHARS or has_marker_heading


def _claim_unit_coverage_gaps(section_label, target_prefix, section_lines, entries):
    """Return value-led coverage gaps for deterministic note-section claim units.

    A random `note-section:*` row is not enough. Each extracted claim unit requires its
    own `note-section:<section>-<kind>-<n>` ledger target, and the matching entry must
    quote text that actually occurs in that unit.
    """
    entry_by_target = {}
    for e in entries:
        if not isinstance(e, dict) or not isinstance(e.get("target"), str):
            continue
        entry_by_target.setdefault(e["target"].strip(), []).append(e)

    claims = _extract_body_claim_units(section_lines)
    gaps = []
    for claim in claims:
        unit_id = _claim_unit_id(target_prefix, claim["id"])
        want = "note-section:%s" % unit_id
        matches = entry_by_target.get(want, [])
        if not matches:
            gaps.append(
                "%s %s claim lacks ledger target '%s' (snippet: %r)"
                % (claim["kind"], section_label, want, claim["snippet"])
            )
            continue
        value_matches = [e for e in matches if _entry_matches_claim_unit(e, claim["text"])]
        if not value_matches:
            gaps.append(
                "%s %s claim target '%s' has no quoted_snippet/claim_text found in the "
                "claim text (expected value-led coverage, not an unrelated ledger row)"
                % (claim["kind"], section_label, want)
            )
            continue
        # Code-source path validity is owned by ledger:code-source-not-file so the red
        # fixtures stay single-fault. Other paper/evidence note claims need provenance here.
        needs_source = [
            e for e in value_matches
            if e.get("kind") in ("paper-claim", "evidence-supported") and claim["kind"] != "code"
        ]
        unsourced_entries = [e for e in needs_source if not _has_provenance_source(e)]
        if unsourced_entries:
            gaps.append(
                "%s %s claim target '%s' has matching paper-claim/evidence-supported "
                "ledger row(s) with no source_url provenance: %s"
                % (claim["kind"], section_label, want, _entry_ids(unsourced_entries))
            )
    return gaps


def _entry_ids(entries):
    ids = []
    for e in entries:
        claim_id = e.get("claim_id") if isinstance(e, dict) else None
        ids.append(str(claim_id) if claim_id else "<missing claim_id>")
    return ", ".join(ids)


def _claim_unit_id(target_prefix, body_claim_id):
    if body_claim_id.startswith("body-"):
        return target_prefix + body_claim_id[len("body"):]
    return "%s-%s" % (target_prefix, body_claim_id)


def _entry_matches_expected_value(entry, expected_text):
    haystack = _normalize_claim_text(expected_text)
    for key in ("quoted_snippet", "claim_text"):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        needle = _normalize_claim_text(value)
        if needle and (needle in haystack or haystack in needle):
            return True
    return False


def _has_provenance_source(entry):
    src = entry.get("source_url")
    return isinstance(src, str) and bool(src.strip())


def _entry_matches_claim_unit(entry, claim_text):
    return _entry_matches_expected_value(entry, claim_text)


def _normalize_claim_text(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _extract_body_claim_units(lines):
    claims = []
    counters = {"prose": 0, "equation": 0, "table": 0, "code": 0}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        if IMAGE_LINK_RE.search(s):
            # Figure provenance is owned by the `figure:<path>` ledger target.
            i += 1
            continue
        if s.startswith("```"):
            block = [lines[i]]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            _append_body_claim(claims, counters, "code", "\n".join(block))
            continue
        if s == "$$":
            block = [lines[i]]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip() == "$$":
                    i += 1
                    break
                i += 1
            _append_body_claim(claims, counters, "equation", "\n".join(block))
            continue
        if s.startswith("$") and s.endswith("$") and len(s) > 1 and not is_table_line(s):
            _append_body_claim(claims, counters, "equation", lines[i])
            i += 1
            continue
        if is_table_line(s):
            block = []
            while i < len(lines) and is_table_line(lines[i].strip()):
                if not is_separator_row(lines[i]):
                    block.append(lines[i])
                i += 1
            if block:
                _append_body_claim(claims, counters, "table", "\n".join(block))
            continue

        para = [lines[i]]
        i += 1
        while i < len(lines):
            ns = lines[i].strip()
            if (
                not ns
                or ns.startswith("#")
                or ns.startswith("```")
                or ns == "$$"
                or is_table_line(ns)
                or IMAGE_LINK_RE.search(ns)
            ):
                break
            para.append(lines[i])
            i += 1
        text = "\n".join(para)
        if _is_substantive_prose_claim(text):
            _append_body_claim(claims, counters, "prose", text)
    return claims


def _is_substantive_prose_claim(text):
    cleaned = _normalize_claim_text(re.sub(r"^\s*[-*+]\s+", "", text.strip()))
    if not cleaned:
        return False
    words = re.findall(r"[a-z0-9]+", cleaned)
    if len(words) < 4:
        return False
    if "http://" in cleaned or "https://" in cleaned:
        return False
    return bool(re.search(r"[a-z0-9]", cleaned))


def _append_body_claim(claims, counters, kind, text):
    normalized = _normalize_claim_text(text)
    if not normalized:
        return
    counters[kind] += 1
    snippet = normalized[:80]
    claims.append(
        {
            "id": "body-%s-%d" % (kind, counters[kind]),
            "kind": kind,
            "text": text,
            "snippet": snippet,
        }
    )


def _resolve_inside_clone(src, note_dir):
    """Resolve a code-claim source path, confined to the clone.

    Returns (resolved_realpath_or_None, escaped). `escaped` is True when at least one
    candidate resolved to a REAL file whose realpath lies outside both the note dir and the
    repo root subtrees (absolute path or `../` traversal) — that file cannot be "the code
    that was read", so the caller reports it as a clone escape rather than a missing file.
    """
    roots = [os.path.realpath(note_dir), os.path.realpath(REPO_ROOT)]
    if os.path.isabs(src):
        candidates = [src]
    else:
        candidates = [os.path.join(note_dir, src), os.path.join(REPO_ROOT, src)]
    escaped = False
    for cand in candidates:
        resolved = os.path.realpath(cand)
        if not os.path.isfile(resolved):
            continue
        if any(resolved == r or resolved.startswith(r + os.sep) for r in roots):
            return resolved, False
        escaped = True
    return None, escaped


def _inside_repo(path):
    root = os.path.realpath(REPO_ROOT)
    resolved = os.path.realpath(path)
    return resolved == root or resolved.startswith(root + os.sep)


def _git_visible(path):
    """Return True when Git tracks the path or would include it as an unignored new file."""
    if not _inside_repo(path):
        return False
    rel_path = os.path.relpath(os.path.realpath(path), REPO_ROOT)
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", rel_path],
            cwd=REPO_ROOT,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return rel_path in [ln.strip() for ln in out.splitlines()]


def _domain_topic_from_path(note_dir):
    try:
        rel_note = os.path.relpath(os.path.realpath(note_dir), os.path.realpath(DOMAINS_DIR))
    except ValueError:
        return None
    parts = rel_note.split(os.sep)
    if rel_note.startswith("..") or os.path.isabs(rel_note) or len(parts) < 2:
        return None
    return parts[0], parts[1]


def _is_existing_domain_note(note_dir):
    dt = _domain_topic_from_path(note_dir)
    if dt is None:
        return False
    readme = os.path.join(note_dir, "README.md")
    return os.path.isfile(readme) and os.path.getsize(readme) > 0


def _gate_link(res, note_text, note_dir):
    broken = []
    for m in TEXT_LINK_RE.finditer(note_text):
        target = m.group(1)
        if not is_relative_target(target):
            continue
        path = clean_link_path(target)
        resolved = os.path.realpath(os.path.join(note_dir, path))
        if not _inside_repo(resolved):
            broken.append("%s -> %s escapes the road-to-master worktree" % (target, resolved))
        elif not os.path.exists(resolved):
            broken.append("%s -> %s" % (target, rel(resolved)))
    res.record(
        "link",
        len(broken) == 0,
        "" if not broken else "broken relative link(s): " + "; ".join(broken),
    )


def _gate_figure(res, note_text, note_dir, entries):
    fig_targets = set()
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("target"), str):
            t = e["target"].strip()
            if t.startswith("figure:") and e.get("source_url"):
                fig_targets.add(t[len("figure:"):])
    problems = []
    for m in IMAGE_LINK_RE.finditer(note_text):
        target = m.group(1)
        if not is_relative_target(target):
            continue
        path = clean_link_path(target)
        resolved = os.path.realpath(os.path.join(note_dir, path))
        if path.lower().endswith(".pdf"):
            problems.append("%s is a .pdf (not a committable figure; *.pdf is gitignored)" % path)
            continue
        if not _inside_repo(resolved):
            problems.append("%s resolves outside the road-to-master worktree (%s)" % (path, resolved))
            continue
        if not os.path.isfile(resolved):
            problems.append("%s referenced but not committed (%s absent)" % (path, rel(resolved)))
            continue
        if not _git_visible(resolved):
            problems.append(
                "%s is not tracked or committable by git (ignored/outside index visibility)" % path
            )
            continue
        if path not in fig_targets:
            problems.append("%s has no ledger figure source (expected target 'figure:%s')" % (path, path))
    res.record(
        "figure",
        len(problems) == 0,
        "" if not problems else "figure-provenance failure(s): " + "; ".join(problems),
    )


def _gate_secret(res, note_text, ledger):
    hits = []
    haystacks = [("note", note_text), ("ledger", json.dumps(ledger, ensure_ascii=False))]
    for where, text in haystacks:
        for label, pat in SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                token = m.group(0)
                masked = token[:4] + "***" if len(token) > 4 else "***"
                hits.append("%s in %s (%s)" % (label, where, masked))
    res.record(
        "secret",
        len(hits) == 0,
        "" if not hits else "committed secret(s): " + "; ".join(hits),
    )


def is_safe_segment(seg):
    """A single path segment is safe iff it matches the topic charset and is not a dot-dir."""
    return bool(TOPIC_NAME_RE.match(seg)) and seg not in (".", "..")


def _gate_name(res, ledger, note_dir):
    problems = []
    # FR5(b): sanitize the ACTUAL <Topic> folder name, not just the self-reported ledger field.
    folder = os.path.basename(os.path.normpath(os.path.abspath(note_dir)))
    if not is_safe_segment(folder):
        problems.append(
            "unsafe note folder name %r: must match ^[A-Za-z0-9._-]+$ (no '..', no path "
            "separators, no shell metachars)" % folder
        )
    topic = ledger.get("topic")
    if not isinstance(topic, str) or topic == "":
        problems.append("ledger.topic missing/empty")
    elif not is_safe_segment(topic):
        problems.append(
            "unsafe ledger.topic %r: must match ^[A-Za-z0-9._-]+$ (no '..', no path "
            "separators, no shell metachars)" % topic
        )
    res.record("name", len(problems) == 0, "; ".join(problems))


def _gate_domain(res, ledger, note_dir):
    domain = ledger.get("domain")
    valid = find_domains()
    problems = []
    if not isinstance(domain, str) or domain not in valid:
        problems.append("ledger.domain %r not in fixed set %s" % (domain, sorted(valid)))
    dt = _domain_topic_from_path(note_dir)
    if dt is not None:
        path_domain, _path_topic = dt
        if path_domain not in valid:
            problems.append("note path domain %r not in fixed set %s" % (path_domain, sorted(valid)))
        if isinstance(domain, str) and domain != path_domain:
            problems.append(
                "ledger.domain %r does not match note path domain %r" % (domain, path_domain)
            )
    res.record(
        "domain",
        len(problems) == 0,
        "; ".join(problems),
    )


def _gate_duplicate(res, ledger, note_dir):
    domain = ledger.get("domain")
    topic = ledger.get("topic")
    valid = find_domains()
    # Guard: only compute a target path once name+domain are safe/valid.
    if not isinstance(topic, str) or not is_safe_segment(topic):
        res.record("duplicate", True, "skipped (topic name unsafe; reported by name gate)")
        return
    if not isinstance(domain, str) or domain not in valid:
        res.record("duplicate", True, "skipped (domain invalid; reported by domain gate)")
        return
    target_dir = os.path.join(DOMAINS_DIR, domain, topic)
    target_readme = os.path.join(target_dir, "README.md")
    same = os.path.realpath(target_dir) == os.path.realpath(note_dir)
    exists_nonempty = os.path.isfile(target_readme) and os.path.getsize(target_readme) > 0
    if (not same) and exists_nonempty:
        res.record(
            "duplicate",
            False,
            "DUPLICATE: target %s already exists with a non-empty README.md; caller must decide "
            "skip/merge (never silently overwrite)" % rel(target_dir),
        )
    else:
        res.record("duplicate", True, "")


def _gate_attestation(res, ledger, note_dir):
    problems = []
    if ledger.get("no_untrusted_code_executed") is not True:
        problems.append(
            "ledger.no_untrusted_code_executed must be boolean true (got %r)"
            % ledger.get("no_untrusted_code_executed")
        )
    markers = []
    for root, _dirs, files in os.walk(note_dir):
        for f in files:
            if f.endswith(".exec-marker"):
                markers.append(rel(os.path.join(root, f)))
    if markers:
        problems.append("execution-marker file(s) present: %s" % markers)
    res.record(
        "attestation",
        len(problems) == 0,
        "" if not problems else "; ".join(problems),
    )


def rel(path):
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def print_report(note_dir, res):
    print("note-quality gate :: %s" % (rel(note_dir) if note_dir else "<unresolved>"))
    print("-" * 72)
    if res.fatal is not None:
        gate, reason = res.fatal
        print("  %-32s %s  %s" % (gate, "FAIL", reason))
        print("-" * 72)
        print("VERDICT: FAIL  failing gates: [%s]" % gate)
        return
    for name in GATE_ORDER:
        ok, detail = res.gates.get(name, (True, ""))
        status = "PASS" if ok else "FAIL"
        line = "  %-32s %s" % (name, status)
        if not ok and detail:
            line += "  %s" % detail
        print(line)
    print("-" * 72)
    failing = res.failing()
    if failing:
        print("VERDICT: FAIL  failing gates: [%s]" % ", ".join(failing))
    else:
        print("VERDICT: PASS  all gates green")


# --------------------------------------------------------------------------------------
# Note resolution
# --------------------------------------------------------------------------------------
def unsafe_spec_segments(spec):
    """Return the unsafe segments of a `<domain>/<Topic>` (or fixture-name) CLI input.

    FR5(b): the input is validated BEFORE it is joined onto the filesystem, so a hostile
    `--note natural_language_processing/../evil` is rejected as gate=name, never resolved.
    """
    return [seg for seg in spec.split("/") if not is_safe_segment(seg)]


def resolve_note_dir(args):
    """Return (note_dir, precheck_result). precheck_result is a fatal Result on unsafe input."""
    if args.path:
        return os.path.abspath(args.path), None
    spec = args.fixture if args.fixture else args.note
    bad = unsafe_spec_segments(spec)
    if bad:
        res = Result()
        res.fatal = (
            "name",
            "unsafe path segment(s) %r in %r rejected before filesystem access: each segment "
            "must match ^[A-Za-z0-9._-]+$ (no '..', no path separators, no shell metachars)"
            % (bad, spec),
        )
        return None, res
    if args.note and "/" in args.note:
        domain = args.note.split("/", 1)[0]
        valid = find_domains()
        if domain not in valid:
            res = Result()
            res.fatal = (
                "domain",
                "note path domain %r not in fixed set %s" % (domain, sorted(valid)),
            )
            return None, res
    if args.fixture:
        return os.path.join(FIXTURES_DIR, args.fixture), None
    cand = os.path.join(DOMAINS_DIR, args.note)
    if os.path.isdir(cand):
        return cand, None
    fx = os.path.join(FIXTURES_DIR, args.note)
    if os.path.isdir(fx):
        return fx, None
    return cand, None  # non-existent -> nothing-ran downstream


def run_all_fixtures():
    manifest_path = os.path.join(FIXTURES_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        print("ERROR: fixtures manifest not found: %s" % rel(manifest_path))
        return 2
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    print("=== research-check fixture suite (red/green) ===")
    all_ok = True
    for name in sorted(manifest.keys()):
        spec = manifest[name]
        note_dir = os.path.join(FIXTURES_DIR, name)
        res = evaluate(note_dir)
        actual = set(res.failing())
        if spec.get("expect_pass"):
            ok = len(actual) == 0
            expected_repr = "PASS (no failing gate)"
        else:
            expected = set(spec.get("expect_fail", []))
            ok = actual == expected
            expected_repr = "FAIL exactly %s" % sorted(expected)
        all_ok = all_ok and ok
        tag = "OK  " if ok else "MISS"
        print(
            "[%s] %-28s got=%s  expected=%s"
            % (tag, name, sorted(actual) if actual else "PASS", expected_repr)
        )
        if not ok:
            print("       ^ mismatch: actual failing gates %s" % sorted(actual))
    print("-" * 72)
    if all_ok:
        print("SUITE: PASS  every fixture matched its expected red/green verdict")
        return 0
    print("SUITE: FAIL  at least one fixture did not match its expected verdict")
    return 1


def main(argv):
    ap = argparse.ArgumentParser(description="road-to-master note-quality gate runner")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--note", help="<domain>/<Topic> under domains/ (falls back to fixtures/)")
    g.add_argument("--fixture", help="fixture name under tools/research/fixtures/")
    g.add_argument("--path", help="arbitrary note directory")
    g.add_argument("--all-fixtures", action="store_true", help="run the red/green fixture suite")
    args = ap.parse_args(argv)

    if args.all_fixtures:
        return run_all_fixtures()

    note_dir, precheck = resolve_note_dir(args)
    res = precheck if precheck is not None else evaluate(note_dir)
    print_report(note_dir, res)
    return 0 if res.is_pass() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
