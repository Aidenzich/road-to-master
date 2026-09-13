import unittest
from quality_summary import summarize, METRICS

def row(cid, model, state='succeeded', score=2):
    return dict(case_id=cid, model=model, count=2, action='book', repeat=1,
                state=state, reviewed=score is not None, **{m: score for m in METRICS})

class SummaryTests(unittest.TestCase):
    def test_missing_failure_is_not_a_visual_zero_or_matched(self):
        rows = [row('a', m) for m in ['codex', 'qwen', 'h3']]
        rows += [row('b', 'codex'), row('b', 'qwen', 'failed', None), row('b', 'h3', score=0)]
        summary = summarize(rows)
        self.assertEqual(summary['matched_case_ids'], ['a'])
        qwen = next(g for g in summary['groups'] if g['scope'] == 'all_available_baseline' and g['dimension'] == 'count' and g['model'] == 'qwen')
        self.assertEqual(qwen['arms'], 2)
        self.assertEqual(qwen['metrics']['exact_count'], dict(reviewed=1, clear_pass=1, partial_or_uncertain=0, clear_failure=0, missing_or_inapplicable=1))

    def test_partial_and_unreviewed_remain_distinct(self):
        rows = [row('a', 'codex', score=1), row('b', 'codex', score=None)]
        summary = summarize(rows)
        group = next(g for g in summary['groups'] if g['scope'] == 'all_available_baseline' and g['dimension'] == 'count' and g['model'] == 'codex')
        self.assertEqual(group['metrics']['exact_count']['partial_or_uncertain'], 1)
        self.assertEqual(group['metrics']['exact_count']['missing_or_inapplicable'], 1)
        self.assertEqual(summary['matched_case_ids'], [])

if __name__ == '__main__':
    unittest.main()
