import os 

conf = {}

conf['gen_data_dir'] = 'v0.2.2'
# conf['gen_data_dir'] = 'gen_data_dummy'

conf['use_bounded'] = False
conf['use_same_font_per_img'] = False
conf['use_same_font_per_para'] = True
conf['data_dir'] = './data'
conf['base_dir'] = '/home/SENSETIME/zhangfahong/Datasets/ocr_jpn_det/'
# gen_data_dir = 'gen_data' + '_bounded' if use_bounded else ''


# conf[temp_dir = './my_data/template'
conf['temp_dir'] = os.path.join(conf['base_dir'], 'template/texture_v2')
# conf['temp_dir'] = os.path.join(conf['base_dir'], 'template/template/')
conf['gt_dir'] = os.path.join(conf['base_dir'], 'template/gts')
conf['gt_line_dir'] = os.path.join(conf['base_dir'], 'template/gts_line')
# conf['texture_dir'] = os.path.join(conf['base_dir'], 'template/texture')
conf['texture_dir'] = os.path.join(conf['base_dir'], 'template/texture_v2')
conf['icon_dir'] = os.path.join(conf['base_dir'], 'template/icon')
conf['bg_dir'] = os.path.join(conf['base_dir'], 'template/background')
conf['prob_render_icon'] = 1
conf['icon_HW_ratio_thre'] = [2.0/6.0, 6.0/2.0]
conf['single_col_prob'] = 0.25

""" random bb generator arguments """
conf['max_H'] = 1248
conf['max_line_h'] = 30
conf['min_h'] = 12
conf['first_shrunk_ratio'] = 0.15
conf['shrunk_ratio'] = 0.01
conf['sparsity_fac'] = 0



base_id_lexicon = ['123456789',
                   'abcdefghij', 'ABCDEFGHIJ']
conf['id_lexicon'] = [[x + ' ' for x in base_id_lexicon[0]],
                      [x + '. ' for x in base_id_lexicon[0]],
                      ['(' + x + ') ' for x in base_id_lexicon[0]],
                      [x + ') ' for x in base_id_lexicon[0]],

                      [x + ' ' for x in base_id_lexicon[1]],
                      [x + '. ' for x in base_id_lexicon[1]],
                      ['(' + x + ') ' for x in base_id_lexicon[1]],
                      [x + ') ' for x in base_id_lexicon[1]],

                      [x + ' ' for x in base_id_lexicon[2]],
                      [x + '. ' for x in base_id_lexicon[2]],
                      ['(' + x + ') ' for x in base_id_lexicon[2]],
                      [x + ') ' for x in base_id_lexicon[2]]]

conf['prob_gen_prefix'] = 0.6


