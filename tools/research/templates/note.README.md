<!--
road-to-master research-note scaffold.
Copy this file to domains/<domain>/<Topic>/README.md, fill every section, and author a sibling
ledger.json (see tools/research/ledger.schema.json). Validate with:
    make research-check NOTE=<domain>/<Topic>

Required sections IN THIS ORDER (enforced by tools/research/check_note.py):
  1. ## 📇 Academic Context   (the academic-value table)
  2. first-principles body     (technical notes need >=1 equation/algorithm/table)
  3. ## 🧪 Critical Assessment (mandatory — interrogate the paper, do not restate it)
  4. ## 🔗 Related notes       (heading mandatory even if empty)
-->

# <Topic> — Research Note

## 📇 Academic Context

<!--
This section must contain the academic-value table below. Every non-'unknown'/'unavailable'
cell needs a ledger entry with target 'table-cell:<field-lowercased-spaces-to-hyphens>' and
a non-empty source_url. The entry's quoted_snippet or claim_text must match the actual cell
value; an unrelated table-cell target is not coverage. Use the LITERAL token 'unknown' or
'unavailable' when you genuinely have no value — never leave a cell blank (a blank cell is a
hidden guess and FAILs). Set 'Venue Kind' to paper | blog | tech-report; blog/tech-report
relaxes the body substance floor.
-->
| Field | Value |
|-|-|
| Title | <paper title> |
| Venue | <venue or unknown> |
| Year | <year or unknown> |
| Authors | <authors or unknown> |
| Official Code | <url or unknown> |
| Venue Kind | paper |

## First Principles

<!--
Explain the method from first principles. Technical papers MUST include at least one of:
an equation ($...$ / $$...$$), an algorithm/code block (```), or a worked table.
Every non-trivial body claim needs an exact ledger entry. The gate extracts deterministic body
units and expects targets such as 'note-section:body-prose-1',
'note-section:body-equation-1', 'note-section:body-table-1', and
'note-section:body-code-1'. The entry's quoted_snippet or claim_text must occur in that body
unit; an unrelated 'note-section:*' row is not coverage. Non-code paper-claim/evidence-supported
body entries each need their own non-empty source_url; a sourced duplicate does not mask an
unsourced duplicate. Reasonable-inference/unproven-or-doubtful entries may be unsourced. An empty
ledger is valid only for a note that makes no claims.
Reference committed figures as ![alt](imgs/<file>.png) — each referenced figure needs a
ledger entry with target 'figure:imgs/<file>.png' and a source_url. .pdf is not a committable
figure (road-to-master .gitignore ignores *.pdf).
-->

## 🧪 Critical Assessment

<!--
Interrogate the paper from first principles — do NOT restate it. Cover:
  ### Problem realness and importance
  ### Baseline, ablation, dataset and metric sufficiency
  ### Novelty vs engineering repackaging   (rename? self-defined benchmark / 射箭畫靶?)
  ### Is the claimed problem actually solved, and is it real-world relevant?
The gate requires this section to be non-trivial: >= 400 chars OR at least one actual '###'
sub-heading naming an interrogation topic above (a marker word buried in prose does not count).
Every non-trivial critique claim also needs a value-led ledger entry, with deterministic targets
such as 'note-section:critical-prose-1', 'note-section:critical-prose-2', ...; the entry's
quoted_snippet or claim_text must occur in that critique claim. If the critique entry is a
paper-claim or evidence-supported claim, include source_url; author's inferences/doubts may use null.
A note may not silently endorse a paper: the ledger must carry at least one
'reasonable-inference' or 'unproven-or-doubtful' entry, OR this section must state an explicit
"no material weaknesses found, evidence: ..." verdict.
-->

## 🔗 Related notes

<!-- Link related notes with relative links, e.g. [SASRec](../SASRec/). May be empty, but keep the heading. -->
