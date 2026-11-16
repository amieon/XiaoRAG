import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visual_region_routing import GateConfig, route_visual_groups
from visual_routing import VisualGate


class VisualRoutingTests(unittest.TestCase):
    def setUp(self):
        self.images = []
        for _ in range(4):
            image = np.full((32, 48, 3), 255, dtype=np.uint8)
            image[8:10, 3:45] = 0
            image[18:20, 3:45] = 0
            self.images.append(image)

    def _group(self, queries, metadata=None):
        group = {
            'images': self.images[:len(queries)],
            'queries': queries,
            'vlm_block_ids': [(0, i) for i in range(len(queries))],
            'figure_token_maps': [{} for _ in queries],
        }
        if metadata is not None:
            group['region_metadata'] = metadata
        return group

    def test_low_risk_ocr_uses_light_and_keeps_order(self):
        group = self._group(['OCR:', 'Table Recognition:', 'OCR:'], [
            {'layout_label': 'text', 'box': [0, 0, 48, 32], 'invalid_layout_fields': []},
            {'layout_label': 'table', 'box': [0, 0, 48, 32], 'invalid_layout_fields': []},
            {'layout_label': 'text', 'box': [0, 0, 48, 32], 'invalid_layout_fields': []},
        ])
        light_calls, vlm_calls = [], []

        def light(images):
            light_calls.append(images)
            return [{'result': 'light-0'}, {'result': 'light-2'}]

        def vlm(groups):
            subset = groups[(1, 2)]
            vlm_calls.append(subset['queries'])
            subset['vlm_results'] = [{'result': 'table'}]

        records = route_visual_groups({(1, 2): group}, light, vlm,
                                      gate=VisualGate(GateConfig()))
        self.assertEqual([x['result'] for x in group['vlm_results']],
                         ['light-0', 'table', 'light-2'])
        self.assertEqual(vlm_calls, [['Table Recognition:']])
        self.assertEqual(len(light_calls), 1)
        self.assertEqual([x['engine'] for x in records], ['light_ocr', 'vlm', 'light_ocr'])
        self.assertTrue(all(x['gate']['uses_recognized_text'] is False for x in records))

    def test_formula_risk_upgrades_ocr_to_vlm(self):
        group = self._group(['OCR:'], [
            {'layout_label': 'text', 'formula_overlap': 0.4, 'box': [0, 0, 48, 32],
             'invalid_layout_fields': []},
        ])

        def light(_):
            self.fail('formula risk must not use light OCR')

        def vlm(groups):
            groups[(1, 2)]['vlm_results'] = [{'result': 'vlm'}]

        records = route_visual_groups({(1, 2): group}, light, vlm,
                                      gate=VisualGate(GateConfig()))
        self.assertEqual(group['vlm_results'][0]['result'], 'vlm')
        self.assertEqual(records[0]['engine'], 'vlm')
        self.assertIn('layout_formula_overlap', records[0]['gate']['reasons'])

    def test_missing_metadata_is_conservative(self):
        group = self._group(['OCR:'])

        def light(_):
            self.fail('missing metadata must use VLM')

        def vlm(groups):
            groups[(1, 2)]['vlm_results'] = [{'result': 'vlm'}]

        records = route_visual_groups({(1, 2): group}, light, vlm,
                                      gate=VisualGate(GateConfig()))
        self.assertEqual(records[0]['engine'], 'vlm')
        self.assertIn('invalid_layout_metadata', records[0]['gate']['reasons'])

    def test_count_mismatch_fails_fast(self):
        group = self._group(['OCR:'])
        group['region_metadata'] = []
        with self.assertRaisesRegex(ValueError, 'Region metadata count mismatch'):
            route_visual_groups({(1, 2): group}, lambda _: [], lambda _: None,
                                gate=VisualGate(GateConfig()))


if __name__ == '__main__':
    unittest.main()


