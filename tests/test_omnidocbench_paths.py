"""Regression checks for Windows filename aliases; no GPU or Docker required."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omnidocbench_paths import page_key, prepare_mapping, prepare_predictions, sha, stage_input, write_json


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.item = {'pilot_index': 0, 'input_sha256': hashlib.sha256(b'image bytes').hexdigest(),
                     'image_path': 'long-' + 'x' * 170 + '.png'}
        self.gt = [{'page_info': {'image_path': 'images/' + self.item['image_path'],
                                 'page_attribute': {'subset': 'layout_hard'}},
                    'layout_dets': [{'order': 0, 'text': 'unchanged', 'poly': [1, 2, 3, 4]}]}]

    def tearDown(self):
        self.temp.cleanup()

    def test_long_name_and_gt_round_trip(self):
        original = copy.deepcopy(self.gt)
        mapping = prepare_mapping(self.root, [self.item], self.gt)
        alias = mapping['pages'][0]['input_alias']
        self.assertLess(len(alias), 30)
        mapped = json.loads((self.root / 'scoring-gt.json').read_text(encoding='utf-8'))
        mapped[0]['page_info']['image_path'] = original[0]['page_info']['image_path']
        self.assertEqual(mapped, original)
        self.assertEqual(self.gt, original)
        self.assertEqual(prepare_mapping(self.root, [self.item], self.gt), mapping)

    def test_duplicate_ids_and_mapping_change_rejected(self):
        with self.assertRaises(ValueError):
            prepare_mapping(self.root, [self.item, self.item], self.gt * 2)
        prepare_mapping(self.root, [self.item], self.gt)
        with self.assertRaises(ValueError):
            prepare_mapping(self.root, [{**self.item, 'pilot_index': 1}], self.gt)

    def test_input_copy_and_hash_failure(self):
        source = self.root / self.item['image_path']
        source.write_bytes(b'image bytes')
        alias = stage_input(source, self.root / 'inputs', self.item)
        self.assertEqual(alias.stem, page_key(self.item))
        self.assertEqual(sha(alias), sha(source))
        alias.write_bytes(b'bad copy')
        stage_input(source, self.root / 'inputs', self.item)
        self.assertEqual(sha(alias), sha(source))
        source.write_bytes(b'changed source')
        with self.assertRaises(ValueError):
            stage_input(source, self.root / 'inputs', self.item)

    def test_legacy_success_and_stale_placeholder(self):
        key = page_key(self.item)
        method = self.root / 'methods' / 'mineru_pro'
        # Legacy status may name any existing validated Markdown artifact.
        old = self.root / 'legacy.md'
        old.write_text('prediction', encoding='utf-8')
        status = method / 'status' / f'{key}.json'
        write_json(status, {'status': 'success', 'input_sha256': self.item['input_sha256'],
                            'markdown': str(old), 'timing': {'markdown_sha256': sha(old)}})
        coverage = prepare_predictions(self.root, 'mineru_pro', [self.item])
        self.assertEqual(coverage[0]['prediction'], 'native_markdown')
        target = method / 'scoring' / 'markdown' / f'{key}.md'
        self.assertEqual(target.read_bytes(), old.read_bytes())
        old.write_text('corrupt', encoding='utf-8')
        with self.assertRaises(ValueError):
            prepare_predictions(self.root, 'mineru_pro', [self.item])
        write_json(status, {'status': 'failed'})
        for _ in range(2):
            coverage = prepare_predictions(self.root, 'mineru_pro', [self.item])
            self.assertEqual(coverage[0]['prediction'], 'empty_failure_placeholder')
            self.assertEqual(target.read_bytes(), b'')


if __name__ == '__main__':
    unittest.main()
