import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hybrid_region_routing import route_groups


class Image:
    shape = (10, 20, 3)


class RoutingTests(unittest.TestCase):
    def test_interleaved_results_keep_identity_and_do_not_fallback(self):
        images = [Image() for _ in range(4)]
        group = {'images': images, 'queries': ['OCR:', 'Table Recognition:', 'OCR:', 'Formula Recognition:'],
                 'vlm_block_ids': [(0, i) for i in range(4)], 'curr_vlm_block_idx': 0}
        calls = []
        def light(xs):
            self.assertEqual(xs, [images[0], images[2]])
            return [{'result': ''}, {'result': 'light2'}]
        def vlm(groups):
            g = groups[(1, 2)]
            self.assertEqual(g['images'], [images[1], images[3]])
            calls.append(g['queries'])
            g['vlm_results'] = [{'result': 'table'}, {'result': 'formula'}]
        audit = route_groups({(1, 2): group}, light, vlm, hybrid=True)
        self.assertEqual([x['result'] for x in group['vlm_results']], ['', 'table', 'light2', 'formula'])
        self.assertEqual([x['block_id'] for x in audit], [[0, i] for i in range(4)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(group['curr_vlm_block_idx'], 0)

    def test_control_uses_vlm_for_all_and_checks_response_count(self):
        group = {'images': [Image()], 'queries': ['OCR:'], 'vlm_block_ids': [(0, 2)]}
        def no_light(xs):
            self.fail('Control must not call OCR')
        def vlm(groups):
            groups[(1, 2)]['vlm_results'] = [{'result': 'control'}]
        records = route_groups({(1, 2): group}, no_light, vlm, hybrid=False)
        self.assertEqual(records[0]['engine'], 'vlm')
        with self.assertRaisesRegex(ValueError, 'OCR result count mismatch'):
            route_groups({(1, 2): group}, lambda xs: [], vlm, hybrid=True)


if __name__ == '__main__':
    unittest.main()
