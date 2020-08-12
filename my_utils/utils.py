import os
import numpy as np
from PIL import Image
import pdb
import cv2

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

class RandomBBGenerator:
    def __init__(self, max_h, max_line_h, min_h, shrunk_ratio, sparsity_fac, first_shrunk_ratio=0.1,
                 single_col_prob=0.5):
        self.para_id = 0
        self.instance_id = 0
        self.max_h = max_h
        self.max_line_h = max_line_h
        self.min_h = min_h
        self.first_shrunk_ratio = first_shrunk_ratio
        self.shrunk_ratio = shrunk_ratio
        self.sparsity_fac = sparsity_fac
        self.single_col_prob = single_col_prob
        self.render_prob = 1 - np.random.rand() * sparsity_fac
        x = np.array(range(max_h))
        self.prob = np.sin((x - min_h) / max_h * np.pi / 2.0)


    def shrunk_bb(self, top_left, bottom_right, shrunk_ratio):
        H = bottom_right[1] - top_left[1]
        W = bottom_right[0] - top_left[0]

        t = [top_left[0] + shrunk_ratio * W, top_left[1] + shrunk_ratio * H]
        b = [bottom_right[0] - shrunk_ratio * W, bottom_right[1] - shrunk_ratio * H]

        t = [round(t[0]), round(t[1])]
        b = [round(b[0]), round(b[1])]

        return [t, b]

    def random_split(self, top_left, bottom_right, shrunk_ratio):

        prob = self.prob
        min_h = self.min_h
        max_line_h = self.max_line_h
        shrunk_ratio = self.shrunk_ratio


        top_left, bottom_right = self.shrunk_bb(top_left, bottom_right, shrunk_ratio)
        H = bottom_right[1] - top_left[1]
        W = bottom_right[0] - top_left[0]
        bb_list = []

        if H < min_h or W < min_h:
            return []

        if np.random.rand() < prob[H]:
            t1 = top_left
            b1 = [bottom_right[0], (top_left[1] + bottom_right[1]) // 2]
            t2 = [top_left[0], (top_left[1] + bottom_right[1]) // 2]
            b2 = bottom_right

            bb_list += self.random_split(t1, b1, shrunk_ratio)
            bb_list += self.random_split(t2, b2, shrunk_ratio)

        elif np.random.rand() < prob[W]:
            t1 = top_left
            b1 = [(top_left[0]+bottom_right[0]) // 2, bottom_right[1]]
            t2 = [(top_left[0]+bottom_right[0]) // 2, top_left[1]]
            b2 = bottom_right

            bb_list += self.random_split(t1, b1, shrunk_ratio)
            bb_list += self.random_split(t2, b2, shrunk_ratio)

        else:
            if np.random.rand() < self.render_prob:
                
                random_choice = np.random.choice(['multi_line', 'grid']) if H < 200 and W < 200 else 'multi_line'
                if random_choice == 'multi_line':

                    cur_h = np.random.randint(min_h, max_line_h+1)
                    num_line = H // cur_h

                    if num_line >= 1:
                        h_line = H // num_line
                        flag = 'single' if np.random.rand() < self.single_col_prob else 'multi'
                        for i in range(num_line):
                            t = [top_left[0], top_left[1] + h_line * i]
                            b = [bottom_right[0], top_left[1] + h_line * (i+1)]
                            s_t, s_b = self.shrunk_bb(t, b, shrunk_ratio)

                            if flag == 'single':
                                v_split = round(cur_h * 1.5)

                                grid_t = [(s_t[0]+s_b[0])//2-v_split//2, s_t[1]]
                                grid_b = [(s_t[0]+s_b[0])//2+v_split//2, s_b[1]]
                                grid_t, grid_b = self.shrunk_bb(grid_t, grid_b, shrunk_ratio)
                                bb_list.append([grid_t[0], grid_t[1], grid_b[0], grid_t[1],
                                                grid_b[0], grid_b[1], grid_t[0], grid_b[1],
                                                self.para_id, self.instance_id, False, False])
                                self.instance_id += 1
                            else:
                                bb_list.append([s_t[0], s_t[1], s_b[0], s_t[1], s_b[0], s_b[1], s_t[0], s_b[1],
                                               self.para_id, self.instance_id, False, True])
                                self.instance_id += 1



                elif random_choice == 'grid':
                    cur_h = np.random.randint(H//2, H+1)
                    num_line = H // cur_h
                    if num_line >= 1:
                        h_line = H // num_line
                        for i in range(num_line):
                            t = [top_left[0], top_left[1] + h_line * i]
                            b = [bottom_right[0], top_left[1] + h_line * (i+1)]
                            s_t, s_b = self.shrunk_bb(t, b, shrunk_ratio)

                            v_split = cur_h
                            num_cols = round((s_b[0]-s_t[0]) // v_split)
                            for j in range(num_cols):
                                grid_t = [s_t[0]+j*v_split, s_t[1]]
                                grid_b = [s_t[0]+(j+1)*v_split, s_b[1]]
                                bb_list.append([grid_t[0], grid_t[1], grid_b[0], grid_t[1],
                                                grid_b[0], grid_b[1], grid_t[0], grid_b[1],
                                                self.para_id, self.instance_id, True, False])
                                self.instance_id += 1

                self.para_id += 1



        return bb_list


    def gen_random_bb_list(self, img_shape):
        H, W = img_shape

        top_left = [0, 0]
        bottom_right = [img_shape[1], img_shape[0]]
        t, b = self.shrunk_bb(top_left, bottom_right, self.first_shrunk_ratio)

        res = self.random_split(t, b, self.shrunk_ratio)

        return res

class PrefixGenerator:
    def __init__(self, prob, num_group, prefix_list):
        self.prefix_list = prefix_list
        self.num_group = num_group
        self.prefix_map = {}
        self.p_prefix_map = {}
        for i in range(num_group):
            rand_idx = np.random.randint(len(prefix_list))
            self.prefix_map[i] = prefix_list[rand_idx] if np.random.rand() < prob else None
            self.p_prefix_map[i] = 0

    def gen_prefix(self, group_id):
        if group_id >= self.num_group or self.prefix_map[group_id] is None:
            return ''
        prefix = self.prefix_map[group_id][self.p_prefix_map[group_id]]
        self.p_prefix_map[group_id] = (self.p_prefix_map[group_id]+1) % len(self.prefix_map[group_id])

        return prefix

class IconRenderer:
    def __init__(self, icon_img_dir, prob_render, HW_ratio_thre, remove_bb=True):
        self.prob_render = prob_render
        self.icon_img_list = []
        self.HW_ratio_thre = HW_ratio_thre
        self.remove_bb = remove_bb
        for path in os.listdir(icon_img_dir):
            img_path = os.path.join(icon_img_dir, path)
            img = Image.open(img_path)
            img = np.array(img)
            self.icon_img_list.append(img)
            if len(img.shape) != 3:
                print(path)

    def render_icon(self, rgb, bound_bb_list, stuff='icon'):
        new_bound_bb_list = []
        for bound_bb in bound_bb_list:

            top_left = [bound_bb[0], bound_bb[1]]
            bottom_right = [bound_bb[4], bound_bb[5]]
            h = bottom_right[1] - top_left[1]
            w = bottom_right[0] - top_left[0]

            valid_img_list = []
            for icon_img in self.icon_img_list:
                icon_h, icon_w, _ = icon_img.shape
                if h/w < self.HW_ratio_thre[1] and h/w >= 1:
                    valid_img_list.append(icon_img)
                if h/w > self.HW_ratio_thre[0] and h/w < 1:
                    valid_img_list.append(icon_img)

            flag = bound_bb[10] if stuff == 'icon' else bound_bb[11]
            if flag and valid_img_list != [] and np.random.rand() < self.prob_render:
                icon_img_id = np.random.randint(len(valid_img_list))
                icon_img = valid_img_list[icon_img_id]
                rgb = self.put_icon(rgb, icon_img, bound_bb, stuff)
            else:
                new_bound_bb_list.append(bound_bb)

        if self.remove_bb:
            bound_bb_list = new_bound_bb_list

        return rgb, bound_bb_list

    def put_icon(self, rgb, icon_img, bound_bb, stuff='icon'):
        top_left = [bound_bb[0], bound_bb[1]]
        bottom_right = [bound_bb[4], bound_bb[5]]
        h = bottom_right[1] - top_left[1]
        w = bottom_right[0] - top_left[0]
        if stuff == 'icon':
            resized_icon, _ = cv2_resize(icon_img, min(h, w))
        else:
            resized_icon, _ = cv2_resize(icon_img, max(h, w))

        icon_h, icon_w, C = resized_icon.shape

        loc = top_left
        

        if stuff == 'icon':
            y, x = np.where(resized_icon[:, :, -1] > 0)
            if icon_h / icon_w > h / w:
                loc[0] += (w - icon_w) // 2
            else:
                loc[1] += (h - icon_h) // 2
            if C == 3:
                resized_icon = resized_icon[:, :, [2, 1, 0]]
        else:
            y, x = np.where(np.zeros((h, w)) > -1)

        rgb[y+loc[1], x+loc[0], :] = resized_icon[y, x, :3]

        return rgb









