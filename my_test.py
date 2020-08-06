import text_utils
import synthgen
import cv2
import numpy as np
import os
import pdb
import time
import tqdm

class MyRenderer(synthgen.RendererV3):

    def my_place_text(self, rgb, collision_mask, given_bb=False, bound_bb=None):

        font = self.text_renderer.font_state.sample()
        font = self.text_renderer.font_state.init_font(font)

        render_res = self.text_renderer.render_sample(font,collision_mask, given_bb, bound_bb)
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
        text_mask = self.feather(text_mask, min_h)

        im_final = self.colorizer.color(rgb,[text_mask],np.array([min_h]))

        return im_final, text, bb, collision_mask

def render_init_collision_mask(H, W, bound_bb_list, expand_ratio=0.2):

    num_char = len(bound_bb_list)

    seg_map = np.ones((H, W)) * 255
    for bb in bound_bb_list:

        top_left = [bb[0], bb[1]]
        bottom_right = [bb[4], bb[5]]
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


def cv2_resize(img, max_H):
    H, W, C = img.shape
    ratio = W / H
    scale = 0
    if H > W:
        res = cv2.resize(img, (round(max_H * ratio), max_H))
        scale = max_H / H
    else:
        res = cv2.resize(img, (max_H, round(max_H / ratio)))
        scale = max_H / W

    return res, scale

def get_current_split(path):
    max_split = 0
    if os.path.exists(path):
        for f in os.listdir(path):
            if len(f) > 5 and f[:5] == 'split':
                max_split = max(max_split, int(f.split('_')[-1]))
    return max_split+1

def get_rgb_bb_coll(temp_path, gt_path, use_bounded, max_H):

    rgb = cv2.imread(temp_path)
    rgb, resize_scale = cv2_resize(rgb, max_H)
    H, W, C = rgb.shape


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

    if use_bounded:
        collision_mask = render_init_collision_mask(H, W, bound_bb_list, 0.2)
    else:
        collision_mask = np.zeros((H, W))

    return rgb, bound_bb_list, collision_mask

use_bounded = True
    
data_dir = './data'
base_dir = '/home/SENSETIME/zhangfahong/Datasets/ocr_jpn_det/'
# gen_data_dir = 'gen_data' + '_bounded' if use_bounded else ''
gen_data_dir = 'gen_data' + '_dummy' if use_bounded else ''


split = get_current_split(os.path.join(base_dir, gen_data_dir))
# temp_dir = './my_data/template'
temp_dir = os.path.join(base_dir, 'template/template/')
gt_dir = os.path.join(base_dir, 'template/gts')
gt_line_dir = os.path.join(base_dir, 'template/gts_line')

lang = 'JPN'
max_H = 1248


font_state = text_utils.FontState(data_dir)
text_renderer = text_utils.RenderFont(data_dir, lang)
renderer = MyRenderer(data_dir, max_time=5, lang=lang)

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

    # init_rgbs.append(rgb)
    # init_collision_masks.append(collision_mask)
    # bound_bb_lists.append(bound_bb_list)


    if os.path.exists(gt_line_path):
        rgb, bound_bb_list, collision_mask = get_rgb_bb_coll(temp_path, gt_line_path, use_bounded, max_H)

        init_rgbs.append(rgb)
        init_collision_masks.append(collision_mask)
        bound_bb_lists.append(bound_bb_list)

        vis_mask = np.tile(np.expand_dims(collision_mask, 2), (1, 1, 3))
        cv2.imwrite('./my_results/{}.png'.format('mask_' + temp_file), vis_mask)



print('total number of template file: {}'.format(len(init_rgbs)))

while True:

    res_img_dir = os.path.join(base_dir, gen_data_dir, 'split_{}/image'.format(str(split)))
    res_gt_dir = os.path.join(base_dir, gen_data_dir, 'split_{}/gts'.format(str(split)))

    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
    if not os.path.exists(res_gt_dir):
        os.makedirs(res_gt_dir)

    for cnt in tqdm.tqdm(range(1000)):

        temp_file_id = np.random.randint(len(init_rgbs))

        collision_mask = init_collision_masks[temp_file_id].copy()
        rgb = init_rgbs[temp_file_id].copy()
        bound_bb_list = bound_bb_lists[temp_file_id]


        num_word_per_temp = np.random.randint(3, 50)

        res_img_file = '{}/{}.png'.format(res_img_dir, str(cnt))
        res_gt_file = '{}/{}.txt'.format(res_gt_dir, str(cnt))
        bbs = []
        texts = []

        if use_bounded:

            for bound_bb in bound_bb_list:
                render_res = None
                fail_cnt = 0
                while render_res is None and fail_cnt < 5:
                    try:
                        top_left = [bound_bb[0], bound_bb[1]]
                        bottom_right = [bound_bb[4], bound_bb[5]]

                        render_res = renderer.my_place_text(rgb, collision_mask, use_bounded,
                                                            np.array(top_left+bottom_right))
                    except Exception as exc:
                        pass
                        # print(exc)

                    fail_cnt += 1

                if render_res is None:
                    continue

                rgb, text, bb, collision_mask = render_res
                bbs.append(merge_char_bb(bb))
                texts.append(text.replace('\n', ''))
        else:
 
            for i in range(num_word_per_temp):
                render_res = None
                fail_cnt = 0
                while render_res is None and fail_cnt < 10:
                    try:
                        render_res = renderer.my_place_text(rgb, collision_mask)
                    except Exception as exc:
                        print(exc)
                    fail_cnt += 1

                if render_res is None:
                    break

                rgb, text, bb, collision_mask = render_res
                bbs.append(merge_char_bb(bb))
                texts.append(text.replace('\n', ''))


        with open(res_gt_file, 'w') as f:
            for i in range(len(bbs)):
                line = bb2str(bbs[i])
                f.write(line + ',' + texts[i] + '\n')

            f.close()

        if cv2.imwrite(res_img_file, rgb):
            print('Successfully write into: {}'.format(res_img_file))
        else:
            print('Failed to write into: {}'.format(res_img_file))

    # split += 1
    split = get_current_split(os.path.join(base_dir, gen_data_dir))
