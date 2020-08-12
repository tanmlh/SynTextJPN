import text_utils
import synthgen
import cv2
import numpy as np
import os
import pdb
import time
import tqdm
import _pickle as pkl
import importlib
import argparse
from PIL import Image
from my_utils import utils

class MyRenderer(synthgen.RendererV3):

    def get_font_num(self):
        return len(self.text_renderer.font_state.fonts)

    def my_sample_font(self):
        self.my_font = self.text_renderer.font_state.sample()

    def my_place_text(self, rgb, collision_mask, given_bb=False, bound_bb=None, font_idx=None,
                      seed=None, text_prefix='', height_seed=None):

        # font = self.text_renderer.font_state.sample(font_idx)
        font = self.text_renderer.font_state.init_font(self.my_font)

        render_res = self.text_renderer.render_sample(font,collision_mask, given_bb, bound_bb,
                                                      text_prefix, height_seed)
        if render_res is None: # rendering not successful
            return #None
        else:
            text_mask,loc,bb,text = render_res

        # update the collision mask with text:
        collision_mask += (255 * (text_mask>0)).astype('uint8')

        # warp the object mask back onto the image:
        text_mask_orig = text_mask.copy()
        bb_orig = bb.copy()
        # text_mask = self.warpHomography(text_mask,H,rgb.shape[:2][::-1])
        # bb = self.homographyBB(bb,Hinv)

        # if not self.bb_filter(bb_orig, bb, text):
            #warn("bad charBB statistics")
            # return #None

        # get the minimum height of the character-BB:
        min_h = self.get_min_h(bb,text)

        #feathering:
        # text_mask = self.feather(text_mask, min_h)

        im_final = self.colorizer.color(rgb,[text_mask],np.array([min_h]), seed=seed)

        return im_final, text, bb, collision_mask

def render_init_collision_mask(H, W, bound_bb_list, expand_ratio=0.2):

    num_char = len(bound_bb_list)

    seg_map = np.ones((H, W)) * 255
    for bb in bound_bb_list:

        top_left = [round(bb[0]), round(bb[1])]
        bottom_right = [round(bb[4]), round(bb[5])]
        seg_map[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0

    return seg_map

def merge_char_bb(bb):
    # bb: (2, 4, num_char)
    min_v = bb.min(axis=2)
    max_v = bb.max(axis=2)

    top_left = min_v.min(axis=1)
    bottom_right = max_v.max(axis=1)

    return np.array([top_left[0], top_left[1], bottom_right[0], top_left[1], bottom_right[0],
                     bottom_right[1], top_left[0], bottom_right[1]])

def bb2str(bb):
    bb_list = bb.reshape(8).tolist()
    bb_list = [str(round(x)) for x in bb_list]

    return ','.join(bb_list)

def get_current_split(path):
    max_split = 0
    if os.path.exists(path):
        for f in os.listdir(path):
            if len(f) > 5 and f[:5] == 'split':
                max_split = max(max_split, int(f.split('_')[-1]))
    return max_split+1

def get_rgb_bb_coll(temp_path, gt_path, use_bounded, max_H):

    rgb = cv2.imread(temp_path)
    if max(rgb.shape[:2]) > max_H:
        rgb, resize_scale = utils.cv2_resize(rgb, max_H)

    H, W, C = rgb.shape

    if use_bounded:
        with open(gt_path, 'r') as gt_f:
            lines = gt_f.readlines()
            bound_bb_list = []
            for line in lines:
                temp = line.strip().split(',')[:8]
                for i, x in enumerate(temp):
                    temp[i] = round(int(temp[i]) * resize_scale)
                bb = np.array(temp)
                if bb[2] <= bb[0] or bb[5] <= bb[1]:
                    print('{}: bad bb: {}'.format(gt_path, bb))
                    continue
                bound_bb_list.append(bb)
            gt_f.close()

        collision_mask = render_init_collision_mask(H, W, bound_bb_list, 0.2)
    else:
        bound_bb_list = None
        collision_mask = np.zeros((H, W))

    return rgb, bound_bb_list, collision_mask


if __name__ == '__main__':

    "argument preparation"
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default='conf.conf_0')
    args = parser.parse_args()
    conf = importlib.import_module(args.conf_path).conf

    use_bounded = conf['use_bounded']
    use_same_font_per_img = conf['use_same_font_per_img']
    use_same_font_per_para = conf['use_same_font_per_para']
    data_dir = conf['data_dir']
    base_dir = conf['base_dir']
    gen_data_dir = conf['gen_data_dir']
    temp_dir = conf['temp_dir']
    gt_dir = conf['gt_dir']
    gt_line_dir = conf['gt_line_dir']
    texture_img_dir = conf['texture_dir']
    icon_img_dir = conf['icon_dir']
    bg_img_dir = conf['bg_dir']

    max_H = conf['max_H']
    max_line_h = conf['max_line_h']
    min_h = conf['min_h']
    shrunk_ratio = conf['shrunk_ratio']
    first_shrunk_ratio = conf['first_shrunk_ratio']
    sparsity_fac = conf['sparsity_fac']
    single_col_prob = conf['single_col_prob']

    prefix_list = conf['id_lexicon']
    prob_gen_prefix = conf['prob_gen_prefix']
    prob_render_icon = conf['prob_render_icon']

    icon_HW_ratio_thre = conf['icon_HW_ratio_thre']
    
    lang = 'JPN'


    renderer = MyRenderer(data_dir, max_time=5, lang=lang)

    """ get background image and corresbonding bounding boxes (if exists) """
    init_rgbs = []
    init_collision_masks = []
    bound_bb_lists = []


    for temp_file in os.listdir(temp_dir):
        if temp_file[0] == '.':
            continue

        collision_mask = None
        temp_path = os.path.join(temp_dir, temp_file)
        gt_path = os.path.join(gt_dir, temp_file+'.txt')
        gt_line_path = os.path.join(gt_line_dir, temp_file+'.txt')


        rgb, bound_bb_list, collision_mask = get_rgb_bb_coll(temp_path, gt_path, use_bounded, max_H)

        init_rgbs.append(rgb)
        init_collision_masks.append(collision_mask)
        bound_bb_lists.append(bound_bb_list)

        if os.path.exists(gt_line_path):
            rgb, bound_bb_list, collision_mask = get_rgb_bb_coll(temp_path, gt_line_path, use_bounded, max_H)
            init_rgbs.append(rgb)
            init_collision_masks.append(collision_mask)
            bound_bb_lists.append(bound_bb_list)

            # vis_mask = np.tile(np.expand_dims(collision_mask, 2), (1, 1, 3))
            # cv2.imwrite('./my_results/{}.png'.format('mask_' + temp_file), vis_mask)

    print('total number of template files: {}'.format(len(init_rgbs)))
    icon_renderer = utils.IconRenderer(icon_img_dir, prob_render_icon, icon_HW_ratio_thre,
                                       remove_bb=True)
    # bg_renderer = utils.IconRenderer(bg_img_dir, prob_render_icon, icon_HW_ratio_thre,
    #                                  remove_bb=False)

    while True:

        """ get a new split dir to store the result """
        split = get_current_split(os.path.join(base_dir, gen_data_dir))
        res_img_dir = os.path.join(base_dir, gen_data_dir, 'split_{}/image'.format(str(split)))
        res_gt_dir = os.path.join(base_dir, gen_data_dir, 'split_{}/gts'.format(str(split)))

        if not os.path.exists(res_img_dir):
            os.makedirs(res_img_dir)

        if not os.path.exists(res_gt_dir):
            os.makedirs(res_gt_dir)

        """ generate 1000 images per split """
        for cnt in tqdm.tqdm(range(1000)):

            temp_file_id = np.random.randint(len(init_rgbs))

            collision_mask = init_collision_masks[temp_file_id].copy()
            rgb = init_rgbs[temp_file_id].copy()
            if np.random.rand() < 0.5:
                rgb = np.rot90(rgb)
            cur_max_H = max(rgb.shape)

            if use_bounded:
                bound_bb_list = bound_bb_lists[temp_file_id]
                pre_generator = utils.PrefixGenerator(prob_gen_prefix, 0, prefix_list)
            else:
                rand_fac = np.random.randint(1, 4)
                bb_generator = utils.RandomBBGenerator(cur_max_H, max_line_h, min_h, shrunk_ratio,
                                                       sparsity_fac, first_shrunk_ratio,
                                                       single_col_prob)

                bound_bb_list = bb_generator.gen_random_bb_list(rgb.shape[:2])
                num_para = bb_generator.para_id
                pre_generator = utils.PrefixGenerator(prob_gen_prefix, num_para, prefix_list)

                # rgb, bound_bb_list = bg_renderer.render_icon(rgb, bound_bb_list, stuff='bg')
                rgb, bound_bb_list = icon_renderer.render_icon(rgb, bound_bb_list)


            res_img_file = '{}/{}.png'.format(res_img_dir, str(cnt))
            res_gt_file = '{}/{}.txt'.format(res_gt_dir, str(cnt))
            bbs = []
            texts = []

            renderer.my_sample_font()
            last_para_id = -1

            for bound_bb in bound_bb_list:

                para_id = bound_bb[8]
                if use_same_font_per_para:
                    if para_id != last_para_id:
                        renderer.my_sample_font()
                        last_para_id = para_id
                    seed = cnt * split * para_id

                elif use_same_font_per_img:
                    seed = cnt * split

                else:
                    seed = None
                    renderer.my_sample_font()


                render_res = None
                fail_cnt = 0
                while render_res is None and fail_cnt < 5:
                    try:
                        top_left = [bound_bb[0], bound_bb[1]]
                        bottom_right = [bound_bb[4], bound_bb[5]]

                        render_res = renderer.my_place_text(rgb, collision_mask, True,
                                                            np.array(top_left+bottom_right),
                                                            seed=seed,
                                                            text_prefix=pre_generator.gen_prefix(para_id),
                                                            height_seed=seed*1007)
                    except Exception as exc:
                        print(exc)

                    fail_cnt += 1

                if render_res is None:
                    continue

                rgb, text, bb, collision_mask = render_res
                bbs.append(merge_char_bb(bb))
                texts.append(text.replace('\n', ''))

                # collision_mask = render_init_collision_mask(rgb.shape[0], rgb.shape[1], bbs, 0)
                # vis_mask = np.tile(np.expand_dims(collision_mask, 2), (1, 1, 3)) * 0.5 + 0.5 * rgb
                # cv2.imwrite('./my_results/{}.png'.format('mask_' + str(cnt)), vis_mask)
           
            with open(res_gt_file, 'w') as f:
                for i in range(len(bbs)):
                    line = bb2str(bbs[i])
                    f.write(line + ',' + texts[i] + '\n')

                f.close()

            if not cv2.imwrite(res_img_file, rgb):
                print('Failed to write into: {}'.format(res_img_file))
