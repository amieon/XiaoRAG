import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from table_routing import TableGate, TableGateConfig
from visual_region_routing import Region, GateConfig, route_visual_groups, page_contexts
from visual_routing import VisualGate
from hybrid_table_backend import TableBackend


def grid(xs=(8, 100, 235, 392), ys=(8, 62, 130, 232)):
    im = np.full((ys[-1]+9, xs[-1]+9, 3), 255, dtype=np.uint8)
    for x in xs:
        cv2.line(im, (x, ys[0]), (x, ys[-1]), (0, 0, 0), 2)
    for y in ys:
        cv2.line(im, (xs[0], y), (xs[-1], y), (0, 0, 0), 2)
    for y in ys[:-1]:
        for x in xs[:-1]:
            cv2.putText(im, '42', (x+12, y+32), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1)
    return im


def metadata(**kw):
    return dict(layout_label='table', invalid_layout_fields=[], formula_overlap=0, **kw)


class TableGateTests(unittest.TestCase):
    def assess(self, image, meta=None):
        return TableGate().assess(Region(image, metadata() if meta is None else meta))

    def test_unequal_width_regular_grid(self):
        decision = self.assess(grid())
        self.assertEqual(decision['branch'], 'table_pipeline', decision)
        self.assertEqual(decision['features']['cells'], 9)
        self.assertFalse(decision['uses_recognized_text'])

    def test_wireless_and_horizontal_only(self):
        for mode in ('blank', 'rows'):
            im = np.full((241, 401, 3), 255, dtype=np.uint8)
            if mode == 'rows':
                for y in (8,62,130,232):
                    cv2.line(im, (8,y), (392,y), (0,0,0), 2)
            self.assertEqual(self.assess(im)['branch'], 'vlm')

    def test_local_merged_cell_and_broken_separator(self):
        for end in (58, 35):
            im = grid()
            im[12:end, 97:104] = 255
            d = self.assess(im)
            self.assertEqual(d['branch'], 'vlm', d)
            self.assertIn('broken_or_merged_grid', d['reasons'])

    def test_nested_partial_line(self):
        im = grid()
        cv2.line(im, (285,145), (285,220), (0,0,0), 2)
        self.assertEqual(self.assess(im)['branch'], 'vlm')

    def test_content_risk_fill_line_and_triangle(self):
        # A short fill-in line is below the global grid detector threshold,
        # but is long and thin relative to its cell.
        im = grid()
        im[9:61, 9:99] = 255
        cv2.line(im, (20, 30), (90, 30), (0, 0, 0), 2)
        decision = self.assess(im)
        self.assertEqual(decision['branch'], 'vlm', decision)
        self.assertIn('fill_in_line_or_uncovered_stroke', decision['reasons'])
        self.assertGreaterEqual(decision['features']['content_line_count'], 1)

        # An isolated three-sided mark is a geometric symbol risk.
        im = grid()
        im[8:62, 8:100] = 255
        cv2.line(im, (8, 8), (100, 8), (0, 0, 0), 2)
        cv2.line(im, (8, 62), (100, 62), (0, 0, 0), 2)
        cv2.polylines(im, [np.array([[30, 52], [45, 20], [60, 52]], dtype=np.int32)],
                         True, (0, 0, 0), 2)
        decision = self.assess(im)
        self.assertEqual(decision['branch'], 'vlm', decision)
        self.assertIn('geometric_symbol', decision['reasons'])
        self.assertGreaterEqual(decision['features']['content_symbol_count'], 1)

    def test_real_failure_fixture_routes_to_vlm(self):
        fixture = Path(__file__).resolve().parents[1] / 'var/table-failure-diagnosis-20261025/input.png'
        if not fixture.is_file():
            self.skipTest('diagnosis fixture is not present')
        image = cv2.imread(str(fixture))
        decision = self.assess(image)
        self.assertEqual(decision['branch'], 'vlm', decision)
        self.assertIn('geometric_symbol', decision['reasons'])

    def test_quality_size_and_rotation(self):
        im = grid()
        low = (im.astype(float)*.08+210).astype('uint8')
        self.assertIn('low_contrast', self.assess(low)['reasons'])
        self.assertEqual(self.assess(cv2.resize(im,(80,48)))['branch'],'vlm')
        rotated = cv2.warpAffine(im, cv2.getRotationMatrix2D((200,120),5,1), (401,241), borderValue=(255,255,255))
        self.assertEqual(self.assess(rotated)['branch'],'vlm')

    def test_metadata_risks(self):
        for meta in ({}, {'invalid_layout_fields':['box']}, metadata(angle=5), metadata(angle=180),
                     metadata(image_overlap=.1), metadata(math_risk=True),
                     metadata(has_figure_tokens=True), metadata(angle=float('nan')),
                     metadata(vertical='false')):
            with self.subTest(meta=meta):
                self.assertEqual(self.assess(grid(),meta)['branch'],'vlm')
        meta = metadata(); meta['formula_overlap']=.2
        self.assertEqual(self.assess(grid(),meta)['branch'],'vlm')

    def test_detector_image_overlap_and_angle_survive(self):
        contexts = page_contexts([{'label':'table','box':[0,0,400,240]}], [
            {'label':'table','box':[0,0,400,240],'angle':5},
            {'label':'image','box':[20,20,80,80]}])
        self.assertEqual(contexts[0]['image_overlap'],1)
        self.assertEqual(contexts[0]['angle'],5)

    def test_three_branches_keep_ids_prompts_and_order(self):
        text_im=np.full((32,48,3),255,dtype=np.uint8)
        text_im[8:10,3:45]=0; text_im[18:20,3:45]=0
        group={'images':[text_im,grid(),grid(),grid()],
               'queries':['OCR:','Table Recognition:','Formula Recognition:','Table Recognition:'],
               'vlm_block_ids':[(0,0),(0,2),(1,1),(1,3)], 'curr_vlm_block_idx':0,
               'figure_token_maps':[{}, {}, {}, {'figure':'x'}],
               'region_metadata':[{'layout_label':'text','invalid_layout_fields':[]},metadata(),metadata(),metadata()]}
        def vlm(groups):
            subset=groups[(1,2)]
            self.assertEqual(subset['vlm_block_ids'],[(1,1),(1,3)])
            self.assertEqual(subset['queries'],['Formula Recognition:','Table Recognition:'])
            subset['vlm_results']=[{'result':'formula'},{'result':'figure table'}]
        records=route_visual_groups({(1,2):group}, lambda xs:[{'result':'text'}], vlm,
            gate=VisualGate(GateConfig()),table_gate=TableGate(),
            recognize_table=lambda xs:[{'result':'<table><tr><td>42</td></tr></table>'}])
        self.assertEqual([r['engine'] for r in records],['light_ocr','table_pipeline','vlm','vlm'])
        self.assertEqual(group['vlm_results'][2]['result'],'formula')
        self.assertEqual(group['curr_vlm_block_idx'],0)
        self.assertEqual(records[1]['block_id'],[0,2])

    def test_no_table_backend_or_result_count_is_error(self):
        with self.assertRaisesRegex(ValueError,'requires a table recognizer'):
            route_visual_groups({},None,None,gate=VisualGate(GateConfig()),table_gate=TableGate())
        group={'images':[grid()],'queries':['Table Recognition:'],'vlm_block_ids':[(0,0)],'region_metadata':[metadata()]}
        vlm=Mock()
        with self.assertRaisesRegex(ValueError,'Table result count mismatch'):
            route_visual_groups({(1,2):group},None,vlm,gate=VisualGate(GateConfig()),table_gate=TableGate(),recognize_table=lambda xs:[])
        vlm.assert_not_called()

    def test_old_table_route_stays_vlm(self):
        group={'images':[grid()],'queries':['Table Recognition:'],'vlm_block_ids':[(0,0)],'region_metadata':[metadata()]}
        def vlm(groups): groups[(1,2)]['vlm_results']=[{'result':'old'}]
        records=route_visual_groups({(1,2):group},None,vlm,gate=VisualGate(GateConfig()))
        self.assertEqual(records[0]['engine'],'vlm')

    def test_backend_contract_color_escape_and_empty_failure(self):
        backend=TableBackend.__new__(TableBackend)
        backend.ocr=Mock()
        backend.ocr.ocr.return_value=[[[[[0,0],[10,0],[10,10],[0,10]],('A&B',.9)]]]
        backend.structure=Mock(return_value=SimpleNamespace(pred_html='<table><tr><td>A&amp;B</td></tr></table>'))
        bgr=np.zeros((20,20,3),dtype=np.uint8); bgr[:,:,0]=255
        self.assertIn('<table>',backend.recognize(bgr)['result'])
        rgb, lines=backend.structure.call_args.args
        self.assertEqual(rgb[0,0].tolist(),[0,0,255])
        self.assertEqual(lines[0][1],'A&amp;B')
        backend.structure.return_value.pred_html=''
        with self.assertRaisesRegex(RuntimeError,'no HTML table'):
            backend.recognize(bgr)


if __name__=='__main__': unittest.main()

