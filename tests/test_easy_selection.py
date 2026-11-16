"""Guard against misleading easy labels and complex tables entering the split."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from select_omnidocbench_easy import exclusion_reasons


class EasySelectionTest(unittest.TestCase):
    def row(self, **overrides):
        return dict(subset='v1.5', layout='single_column', special_issue=['None'],
                    data_source='book', annotation_count=5,
                    category_counts={'text_block': 3}, **overrides)

    def test_unlabelled_note_excluded(self):
        row = self.row()
        row['data_source'] = 'note'
        self.assertTrue(exclusion_reasons(row, {'layout_dets': []}))

    def test_hard_label_excluded_even_if_single_column(self):
        row = self.row()
        row['subset'] = 'equation_hard'
        self.assertIn('not_base_v1.5', exclusion_reasons(row, {'layout_dets': []}))

    def test_benign_table_tags_allowed_but_merged_cells_rejected(self):
        row = self.row()
        row['special_issue'] = ['table_full_line', 'table_horizontal']
        row['category_counts']['table'] = 1
        annotation = {'layout_dets': [{'category_type': 'table', 'html': '<table><tr><td>a</td></tr></table>'}]}
        self.assertEqual(exclusion_reasons(row, annotation), [])
        annotation['layout_dets'][0]['html'] = '<table><tr><td colspan="2">a</td></tr></table>'
        self.assertTrue(exclusion_reasons(row, annotation))

    def test_inline_math_density_and_escaped_dollars(self):
        annotation = {'layout_dets': [{'category_type': 'text_block', 'text': '$x$ ' * 5}]}
        self.assertIn('more_than_4_inline_math_pairs', exclusion_reasons(self.row(), annotation))
        annotation['layout_dets'][0]['text'] = r'\$10 ' * 20
        self.assertEqual(exclusion_reasons(self.row(), annotation), [])


if __name__ == '__main__':
    unittest.main()
