#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic generator for the note-quality red/green fixtures.

Each fixture is the GOOD baseline plus one (or, for `multi_fault`, three) minimal deltas so it
FAILs exactly its intended gate(s). Re-run `python3 build_fixtures.py` to regenerate the
committed fixture tree and `manifest.json`. The generated files ARE the committed red/green
proof consumed by `check_note.py --all-fixtures`; this generator keeps the deltas reviewable
in one place (it is not invoked by the gate itself).
"""
import base64
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 1x1 PNG (valid, committable, non-PDF figure).
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TITLE = "# Transformer (Attention Is All You Need) — Research Note\n\n"

# The in-clone code file the GOOD ledger's evidence-supported code claim (c1) points at.
# AC-17: a code claim is "supported" only by a real, read file in the inspected clone.
CODE_FILE = (
    "def head(q, k, v, d_k):\n"
    "    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)\n"
    "    return softmax(scores, dim=-1) @ v\n"
)

AC_HEADING = "## \U0001F4C7 Academic Context\n\n"

AC_TABLE = (
    "| Field | Value |\n"
    "|-|-|\n"
    "| Title | Attention Is All You Need |\n"
    "| Venue | NIPS |\n"
    "| Year | 2017 |\n"
    "| Authors | Vaswani et al. |\n"
    "| Official Code | unknown |\n"
    "| Venue Kind | paper |\n\n"
)

# Blank-Venue variant (AC-18 blank/guessed cell).
AC_TABLE_BLANK_VENUE = AC_TABLE.replace("| Venue | NIPS |", "| Venue |  |")

AC_TABLE_ALL_UNKNOWN = (
    "| Field | Value |\n"
    "|-|-|\n"
    "| Title | unknown |\n"
    "| Venue | unknown |\n"
    "| Year | unknown |\n"
    "| Authors | unknown |\n"
    "| Official Code | unknown |\n"
    "| Venue Kind | unknown |\n\n"
)

AC_NO_TABLE = (
    "This section intentionally has prose but no academic-value table. The gate must reject\n"
    "an Academic Context heading without the required `| Field | Value |` table.\n\n"
)

BODY = (
    "## First Principles\n\n"
    "The core mechanism is scaled dot-product attention. Given queries $Q$, keys $K$ and\n"
    "values $V$ the layer computes:\n\n"
    "$$\n"
    r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"
    "\n$$\n\n"
    "| Symbol | Shape | Meaning |\n"
    "|-|-|-|\n"
    r"| $Q$ | $(L \times d_k)$ | what a token is looking for |"
    "\n"
    r"| $K$ | $(L \times d_k)$ | what a token exposes |"
    "\n"
    r"| $V$ | $(L \times d_v)$ | the content carried |"
    "\n\n"
    "A minimal reference implementation of one attention head reads:\n\n"
    "```python\n"
    "def head(q, k, v, d_k):\n"
    "    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)\n"
    "    return softmax(scores, dim=-1) @ v\n"
    "```\n\n"
    "![scaled dot-product attention](imgs/fig1.png)\n\n"
)

# Substance-stripped body: prose only, no math/table/code, but keeps the figure.
BODY_NO_SUBSTANCE = (
    "## First Principles\n\n"
    "This note describes the model in plain prose only. The layer relates queries, keys and\n"
    "values, running several projections in parallel and combining the results. No equations,\n"
    "algorithm blocks or tables are included in this deliberately thin body.\n\n"
    "![scaled dot-product attention](imgs/fig1.png)\n\n"
)

BODY_CLAIM_NO_FIGURE = (
    "## First Principles\n\n"
    "The model uses scaled dot-product attention to compare token representations.\n\n"
    "$\\text{Attention}(Q,K,V)=\\text{softmax}(QK^T)V$\n\n"
)

CA = (
    "## \U0001F9EA Critical Assessment\n\n"
    "### Problem realness and importance\n"
    "Long-range dependency modelling in sequence transduction is a real, well-motivated problem:\n"
    "RNNs serialise computation along sequence length and struggle to propagate gradients across\n"
    "long spans. The framing is not manufactured.\n\n"
    "### Baseline, ablation, dataset and metric sufficiency\n"
    "WMT14 EN-DE / EN-FR and BLEU are standard, and the ablations over head count, key dimension\n"
    "and positional encodings are reasonable, though the reported gains partly ride on\n"
    "training-budget and regularisation choices the ablations do not fully isolate.\n\n"
    "### Novelty vs engineering repackaging\n"
    "Attention predates this work; the genuine novelty is discarding recurrence entirely and\n"
    "showing a pure-attention stack is trainable and parallelisable. This is a real architectural\n"
    "claim, not a rename or a self-defined benchmark (射箭畫靶).\n\n"
    "### Is the claimed problem actually solved, and is it real-world relevant?\n"
    "The narrowed claim (competitive MT quality at lower training cost) is supported; the broader\n"
    "\"attention is all you need\" slogan is stronger than the evidence for tasks outside the\n"
    "studied setting.\n\n"
)

RELATED = (
    "## \U0001F517 Related notes\n\n"
    "- BERT and later encoder-only models build directly on this stack.\n"
)

# AC-19 trivial variant: < 400 chars AND the interrogation word only appears in PROSE,
# never as a sub-heading — the marker escape hatch must not fire on substring matches.
CA_TRIVIAL = (
    "## \U0001F9EA Critical Assessment\n\n"
    "The baseline looks fine.\n\n"
)

CA_NO_WEAKNESSES = (
    "## \U0001F9EA Critical Assessment\n\n"
    "### Problem realness\n"
    "No material weaknesses found, evidence: this synthetic fixture exists only to prove that\n"
    "a substantive body claim still requires a note-section ledger row even when all Academic\n"
    "Context cells use sanctioned unknown literals.\n\n"
)

RELATED_BROKEN = (
    "## \U0001F517 Related notes\n\n"
    "- BERT and later encoder-only models build directly on this stack.\n"
    "- [see also](../NonExistentSibling/)\n"
)

ESCAPE_PATH = "../" * 16 + "etc/hosts"

GOOD_README = TITLE + AC_HEADING + AC_TABLE + BODY + CA + RELATED

GOOD_LEDGER = {
    "note_path": "domains/natural_language_processing/GoodFixtureNote/README.md",
    "domain": "natural_language_processing",
    "topic": "GoodFixtureNote",
    "no_untrusted_code_executed": True,
    "generated_at": "2026-07-02T00:00:00Z",
    "entries": [
        {"claim_id": "t1", "claim_text": "Paper title", "kind": "evidence-supported",
         "target": "table-cell:title", "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z", "quoted_snippet": "Attention Is All You Need"},
        {"claim_id": "v1", "claim_text": "Published at NIPS", "kind": "evidence-supported",
         "target": "table-cell:venue", "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "31st Conference on Neural Information Processing Systems"},
        {"claim_id": "y1", "claim_text": "Year 2017", "kind": "evidence-supported",
         "target": "table-cell:year", "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z", "quoted_snippet": "2017"},
        {"claim_id": "a1", "claim_text": "Authors Vaswani et al.", "kind": "evidence-supported",
         "target": "table-cell:authors", "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "Ashish Vaswani, Noam Shazeer, Niki Parmar, et al."},
        {"claim_id": "vk1", "claim_text": "Peer-reviewed conference paper", "kind": "evidence-supported",
         "target": "table-cell:venue-kind", "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z", "quoted_snippet": "NIPS 2017"},
        {"claim_id": "f1", "claim_text": "Scaled dot-product attention figure",
         "kind": "evidence-supported", "target": "figure:imgs/fig1.png",
         "source_url": "https://arxiv.org/abs/1706.03762", "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "Figure 2 (left): Scaled Dot-Product Attention"},
        {"claim_id": "bp1", "claim_text": "The core mechanism is scaled dot-product attention",
         "kind": "evidence-supported", "target": "note-section:body-prose-1",
         "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "The core mechanism is scaled dot-product attention"},
        {"claim_id": "be1", "claim_text": "Scaled dot-product attention equation",
         "kind": "evidence-supported", "target": "note-section:body-equation-1",
         "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": r"\text{Attention}(Q,K,V) = \text{softmax}"},
        {"claim_id": "bt1", "claim_text": "Query, key and value tensor shapes",
         "kind": "evidence-supported", "target": "note-section:body-table-1",
         "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "Symbol | Shape | Meaning"},
        {"claim_id": "bp2", "claim_text": "A minimal reference implementation of one attention head reads",
         "kind": "evidence-supported", "target": "note-section:body-prose-2",
         "source_url": "https://arxiv.org/abs/1706.03762",
         "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "A minimal reference implementation of one attention head reads"},
        {"claim_id": "c1", "claim_text": "One attention head implementation",
         "kind": "evidence-supported", "target": "note-section:body-code-1",
         "source_url": "code/attention.py", "retrieved_at": "2026-07-02T00:00:00Z",
         "quoted_snippet": "scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)"},
        {"claim_id": "cp1", "claim_text": "Long-range dependency modelling is a real problem",
         "kind": "reasonable-inference", "target": "note-section:critical-prose-1",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "Long-range dependency modelling in sequence transduction is a real"},
        {"claim_id": "cp2", "claim_text": "Ablations do not fully isolate training-budget effects",
         "kind": "unproven-or-doubtful", "target": "note-section:critical-prose-2",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "reported gains partly ride on training-budget and regularisation choices"},
        {"claim_id": "cp3", "claim_text": "The novelty is a pure-attention trainable stack",
         "kind": "reasonable-inference", "target": "note-section:critical-prose-3",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "the genuine novelty is discarding recurrence entirely"},
        {"claim_id": "cp4", "claim_text": "The broader slogan is stronger than the evidence",
         "kind": "unproven-or-doubtful", "target": "note-section:critical-prose-4",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "The narrowed claim (competitive MT quality at lower training cost) is supported"},
        {"claim_id": "i1", "claim_text": "The slogan overreaches beyond studied tasks",
         "kind": "reasonable-inference", "target": "note-section:critical-novelty",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "broader slogan is stronger than the evidence"},
        {"claim_id": "d1", "claim_text": "Ablations do not isolate training-budget effects",
         "kind": "unproven-or-doubtful", "target": "note-section:critical-doubt",
         "source_url": None, "retrieved_at": None,
         "quoted_snippet": "gains partly ride on training-budget choices"},
    ],
}


def ledger_without(entries_id):
    led = copy.deepcopy(GOOD_LEDGER)
    led["entries"] = [e for e in led["entries"] if e["claim_id"] != entries_id]
    return led


def ledger_mutate(claim_id, **changes):
    led = copy.deepcopy(GOOD_LEDGER)
    for e in led["entries"]:
        if e["claim_id"] == claim_id:
            e.update(changes)
    return led


def ledger_with_unsourced_duplicate(claim_id):
    led = copy.deepcopy(GOOD_LEDGER)
    for e in led["entries"]:
        if e["claim_id"] == claim_id:
            duplicate = copy.deepcopy(e)
            duplicate["claim_id"] = "%s_unsourced_duplicate" % claim_id
            duplicate["source_url"] = None
            duplicate["retrieved_at"] = None
            led["entries"].append(duplicate)
            break
    return led


def ledger_for_no_substance():
    led = copy.deepcopy(GOOD_LEDGER)
    for e in led["entries"]:
        if e["claim_id"] == "bp1":
            e.update(
                {
                    "claim_text": "This note describes the model in plain prose only",
                    "target": "note-section:body-prose-1",
                    "quoted_snippet": "This note describes the model in plain prose only",
                }
            )
    return led


# fixture name -> (readme, ledger-or-None, expected-manifest-entry)
def build_specs():
    specs = {}

    specs["good"] = (GOOD_README, GOOD_LEDGER, {"expect_pass": True})

    specs["ledger_missing_cell"] = (
        GOOD_README, ledger_without("a1"), {"expect_fail": ["ledger:coverage"]},
    )
    specs["academic_context_cell_null_source"] = (
        GOOD_README,
        ledger_mutate("v1", source_url=None),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["unrelated_table_cell_ledger"] = (
        GOOD_README,
        ledger_mutate(
            "t1",
            claim_text="Completely unrelated title",
            quoted_snippet="Completely unrelated title",
        ),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["empty_ledger_body_claim"] = (
        TITLE + AC_HEADING + AC_TABLE_ALL_UNKNOWN + BODY_CLAIM_NO_FIGURE + CA_NO_WEAKNESSES + RELATED,
        ledger_mutate_top(
            topic="EmptyLedgerBodyClaim",
            entries=[],
            generated_at="2026-07-02T00:00:00Z",
        ),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["unrelated_body_claim_ledger"] = (
        TITLE + AC_HEADING + AC_TABLE_ALL_UNKNOWN + BODY_CLAIM_NO_FIGURE + CA_NO_WEAKNESSES + RELATED,
        ledger_mutate_top(
            topic="UnrelatedBodyClaimLedger",
            entries=[
                {
                    "claim_id": "unrelated1",
                    "claim_text": "This ledger row is unrelated to the first-principles body",
                    "kind": "reasonable-inference",
                    "target": "note-section:unrelated-body",
                    "source_url": None,
                    "retrieved_at": None,
                    "quoted_snippet": "not present in the body",
                }
            ],
        ),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["body_claim_null_source"] = (
        GOOD_README,
        ledger_mutate("bp1", source_url=None, retrieved_at=None),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["body_claim_unsourced_duplicate"] = (
        GOOD_README,
        ledger_with_unsourced_duplicate("bp1"),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["critical_assessment_missing_ledger"] = (
        GOOD_README, ledger_without("cp3"), {"expect_fail": ["ledger:coverage"]},
    )
    specs["unrelated_critical_assessment_ledger"] = (
        GOOD_README,
        ledger_mutate(
            "cp3",
            claim_text="This ledger row is unrelated to the critique paragraph",
            quoted_snippet="not present in the Critical Assessment",
        ),
        {"expect_fail": ["ledger:coverage"]},
    )
    specs["broken_relative_link"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + CA + RELATED_BROKEN, GOOD_LEDGER,
        {"expect_fail": ["link"]},
    )
    specs["relative_link_escapes_worktree"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + CA
        + "## \U0001F517 Related notes\n\n"
        + "- [outside repo](%s)\n" % ESCAPE_PATH,
        GOOD_LEDGER,
        {"expect_fail": ["link"]},
    )
    specs["missing_figure"] = (
        GOOD_README.replace("imgs/fig1.png", "imgs/nonexistent.png"), GOOD_LEDGER,
        {"expect_fail": ["figure"]},
    )
    specs["figure_escapes_worktree"] = (
        GOOD_README.replace("imgs/fig1.png", "imgs/%s" % ESCAPE_PATH), GOOD_LEDGER,
        {"expect_fail": ["figure"]},
    )
    specs["secret_in_snippet"] = (
        GOOD_README, ledger_mutate("d1", quoted_snippet="aws_key = AKIAIOSFODNN7EXAMPLE"),
        {"expect_fail": ["secret"]},
    )
    specs["unsafe_topic_name"] = (
        GOOD_README, ledger_mutate_top(topic="../evil"), {"expect_fail": ["name"]},
    )
    specs["invalid_domain"] = (
        GOOD_README, ledger_mutate_top(domain="quantum_computing"), {"expect_fail": ["domain"]},
    )
    specs["preexisting_folder"] = (
        GOOD_README, ledger_mutate_top(domain="natural_language_processing", topic="AttentionIsAllYouNeed"),
        {"expect_fail": ["duplicate"]},
    )
    specs["absent_ledger_file"] = (
        GOOD_README, None, {"expect_fail": ["nothing-ran"]},
    )
    specs["no_substance"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY_NO_SUBSTANCE + CA + RELATED,
        ledger_for_no_substance(),
        {"expect_fail": ["structure:substance"]},
    )
    specs["missing_academic_context"] = (
        GOOD_README.replace(AC_HEADING, "## Context\n\n", 1), GOOD_LEDGER,
        {"expect_fail": ["structure:academic-context"]},
    )
    specs["missing_academic_context_table"] = (
        TITLE + AC_HEADING + AC_NO_TABLE + BODY + CA + RELATED,
        GOOD_LEDGER,
        {"expect_fail": ["structure:academic-context"]},
    )
    specs["wrong_section_order"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + RELATED + CA, GOOD_LEDGER,
        {"expect_fail": ["structure:section-order"]},
    )
    specs["prose_before_academic_context"] = (
        TITLE + "This body claim appears too early and even includes $x+y$.\n\n"
        + AC_HEADING + AC_TABLE + BODY + CA + RELATED,
        GOOD_LEDGER,
        {"expect_fail": ["structure:section-order"]},
    )
    # AC-15 disjunct: first-principles body BEFORE Academic Context.
    specs["body_before_academic_context"] = (
        TITLE + BODY + AC_HEADING + AC_TABLE + CA + RELATED, GOOD_LEDGER,
        {"expect_fail": ["structure:section-order"]},
    )
    # AC-15 disjunct: Related notes before the body (body AFTER Related).
    specs["body_after_related_notes"] = (
        TITLE + AC_HEADING + AC_TABLE + CA + RELATED + BODY, GOOD_LEDGER,
        {"expect_fail": ["structure:section-order"]},
    )
    specs["missing_related_notes"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + CA, GOOD_LEDGER,
        {"expect_fail": ["structure:related-notes"]},
    )
    specs["code_claim_url_source"] = (
        GOOD_README,
        ledger_mutate("c1", source_url="https://github.com/harvardnlp/annotated-transformer/blob/master/attention.py"),
        {"expect_fail": ["ledger:code-source-not-file"]},
    )
    specs["blank_cell"] = (
        TITLE + AC_HEADING + AC_TABLE_BLANK_VENUE + BODY + CA + RELATED, GOOD_LEDGER,
        {"expect_fail": ["ledger:blank-or-unsourced-cell"]},
    )
    # AC-17 disjuncts beyond the URL case: null source and a path that resolves to nothing.
    specs["code_claim_null_source"] = (
        GOOD_README, ledger_mutate("c1", source_url=None),
        {"expect_fail": ["ledger:code-source-not-file"]},
    )
    specs["code_claim_missing_file_source"] = (
        GOOD_README, ledger_mutate("c1", source_url="code/not_committed.py"),
        {"expect_fail": ["ledger:code-source-not-file"]},
    )
    # AC-17 clone-containment disjuncts: a real file OUTSIDE the clone is NOT "the code that
    # was read" — neither via `../` traversal nor via an absolute path. (If /etc/hosts is
    # absent on the host, the same gate still fails with does-not-resolve.)
    specs["code_claim_escaping_source"] = (
        GOOD_README,
        ledger_mutate("c1", source_url="../" * 16 + "etc/hosts"),
        {"expect_fail": ["ledger:code-source-not-file"]},
    )
    specs["code_claim_absolute_escape"] = (
        GOOD_README, ledger_mutate("c1", source_url="/etc/hosts"),
        {"expect_fail": ["ledger:code-source-not-file"]},
    )
    # NFR E1: ledger.schema.json is ENFORCED (ledger:schema), not decorative — an entry
    # missing required fields, a wrong-typed top-level field, and an unknown top-level key
    # must all be named by the shape gate (and only by it; enum/const/pattern violations
    # belong to kind-enum/attestation/name).
    led_schema = copy.deepcopy(GOOD_LEDGER)
    led_schema["generated_at"] = 20260702
    led_schema["not_in_schema"] = True
    specs["ledger_schema_violation"] = (
        GOOD_README, led_schema, {"expect_fail": ["ledger:schema"]},
    )
    specs["malformed_retrieved_at"] = (
        GOOD_README, ledger_mutate("v1", retrieved_at="not-a-date"),
        {"expect_fail": ["ledger:schema"]},
    )
    specs["missing_critical_assessment"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + RELATED, GOOD_LEDGER,
        {"expect_fail": ["structure:critical-assessment"]},
    )
    # AC-19: section present but trivial (< 400 chars, marker word only in prose).
    specs["trivial_critical_assessment"] = (
        TITLE + AC_HEADING + AC_TABLE + BODY + CA_TRIVIAL + RELATED, GOOD_LEDGER,
        {"expect_fail": ["structure:critical-assessment"]},
    )
    # FR5(b): the hostile name lives in the ACTUAL folder name (ledger.topic stays safe);
    # the basename check must catch it even when the self-reported field lies.
    specs["unsafe_folder;name"] = (
        GOOD_README, GOOD_LEDGER, {"expect_fail": ["name"]},
    )
    specs["invalid_kind"] = (
        GOOD_README, ledger_mutate("i1", kind="verified"), {"expect_fail": ["ledger:kind-enum"]},
    )
    # Uncritical restatement: no inference/doubt entries and no verdict line.
    led_uncritical = copy.deepcopy(GOOD_LEDGER)
    for e in led_uncritical["entries"]:
        if e["kind"] in ("reasonable-inference", "unproven-or-doubtful"):
            e["kind"] = "paper-claim"
            e["source_url"] = "https://arxiv.org/abs/1706.03762"
            e["retrieved_at"] = "2026-07-02T00:00:00Z"
    specs["uncritical_restatement"] = (
        GOOD_README, led_uncritical, {"expect_fail": ["ledger:no-endorsement"]},
    )
    specs["bad_attestation"] = (
        GOOD_README, ledger_mutate_top(no_untrusted_code_executed=False),
        {"expect_fail": ["attestation"]},
    )
    specs["executed_code_marker"] = (
        GOOD_README, GOOD_LEDGER, {"expect_fail": ["attestation"]},
    )
    # Multi-fault: missing Academic Context heading + uncovered ledger cell + broken link.
    multi_readme = (TITLE + "## Context\n\n" + AC_TABLE + BODY + CA + RELATED_BROKEN)
    specs["multi_fault"] = (
        multi_readme, ledger_without("a1"),
        {"expect_fail": ["structure:academic-context", "ledger:coverage", "link"]},
    )
    return specs


def ledger_mutate_top(**changes):
    led = copy.deepcopy(GOOD_LEDGER)
    led.update(changes)
    return led


def write_fixture(name, readme, ledger):
    d = os.path.join(HERE, name)
    os.makedirs(os.path.join(d, "imgs"), exist_ok=True)
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    # Every fixture references imgs/fig1.png (except missing_figure, which points elsewhere);
    # ship the real PNG so only the intended gate fails.
    with open(os.path.join(d, "imgs", "fig1.png"), "wb") as fh:
        fh.write(PNG_BYTES)
    # Every ledger's c1 code claim points at code/attention.py; ship the real file so only
    # the intended gate fails (code_claim_missing_file_source points at a different path).
    os.makedirs(os.path.join(d, "code"), exist_ok=True)
    with open(os.path.join(d, "code", "attention.py"), "w", encoding="utf-8") as fh:
        fh.write(CODE_FILE)
    if ledger is not None:
        with open(os.path.join(d, "ledger.json"), "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    else:
        # absent_ledger_file: ensure no stale ledger.json remains.
        stale = os.path.join(d, "ledger.json")
        if os.path.exists(stale):
            os.remove(stale)
    marker = os.path.join(d, "untrusted.exec-marker")
    if name == "executed_code_marker":
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("fixture proves execution-marker detection\n")
    elif os.path.exists(marker):
        os.remove(marker)


def main():
    specs = build_specs()
    manifest = {}
    for name, (readme, ledger, expect) in specs.items():
        write_fixture(name, readme, ledger)
        manifest[name] = expect
    with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("wrote %d fixtures + manifest.json" % len(specs))


if __name__ == "__main__":
    main()
