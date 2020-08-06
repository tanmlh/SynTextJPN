# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cp
import os
import pdb

pygame.freetype.init()

ys = np.arange(8, 200)
A = np.c_[ys, np.ones_like(ys)]

xs = []
models = {}  # linear model

# FS = FontState()
# plt.figure()
# plt.hold(True)

# FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
# self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]

fonts_dir = '/home/SENSETIME/zhangfahong/Datasets/ocr_jpn_det/corpus/20200805/chn'

for font_path in os.listdir(fonts_dir):
    
# for i in xrange(len(FS.fonts)):
    font = freetype.Font(os.path.join(fonts_dir, font_path), size=12)
    h = []
    for y in ys:
        h.append(font.get_sized_glyph_height(y.item()))
    h = np.array(h)
    m, _, _, _ = np.linalg.lstsq(A, h)
    models[font.name] = m
    print("{}:\t{}".format(font_path, font.name))
    xs.append(h)

with open('font_px2pt.cp', 'wb') as f:
    cp.dump(models, f)
# plt.plot(xs,ys[i])
# plt.show()
