"""Freeze singular-wording controls without changing baseline or GPU scheduling."""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
replacements = {
    "Preserve each reference character's": "Preserve the reference character's",
    "Use the pictures for character appearance only, not their original pose": "Use the picture for character appearance only, not the original pose",
    "Exactly 1 distinct people are visible, each appearing once.": "Exactly one person is visible, appearing once.",
    "They share one continuous floor": "The person stands on one continuous floor",
    "all heads and action-relevant hands fully visible with space between faces": "the head and action-relevant hands fully visible with space around the face",
    "Each person raises": "The person raises",
    "All characters are calmly posed and fully clothed.": "The person is calmly posed and fully clothed.",
    "referenced character designs": "referenced character design",
    "The referenced characters share one friendly library scene.": "The referenced character stands in one friendly library scene.",
    "All subjects are already holding": "The subject is already holding",
    "and calmly maintain it": "and calmly maintains it",
}
controls = []
for cohort in ['anime', 'live']:
    baseline_id = f'{cohort}-01-wave-r1'
    raw = (ROOT / 'cases' / baseline_id / 'case.json').read_bytes()
    case = json.loads(raw)
    cid = baseline_id + '-singular-wording'
    target = ROOT / 'cases' / cid
    assert not target.exists(), 'Do not overwrite a frozen control'
    case['id'] = cid
    for key in ['prompt', 'h3_prompt']:
        for before, after in replacements.items():
            case[key] = case[key].replace(before, after)
    assert case['prompt'] != json.loads(raw)['prompt']
    target.mkdir()
    (target / 'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2) + '\n')
    (target / 'prompt.txt').write_text(case['prompt'] + '\n')
    (target / 'h3-prompt.txt').write_text(case['h3_prompt'])
    controls.append(dict(case_id=cid, baseline_case_id=baseline_id,
                         baseline_sha256=hashlib.sha256(raw).hexdigest(), qwen_supported=True))
(ROOT / 'singular-control-index.json').write_text(json.dumps(dict(
    registered_epoch=time.time(), controls=controls, replacements=replacements,
    intervention='Singular grammatical wording bundle, not a single-token causal test',
    execution='Codex first; GPU controls are not admitted outside the existing worker lock',
), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(controls))
