# import os
# import cv2
# import openslide
# import numpy as np
# import imageio
# '''滑动窗口'''
# def sliding_window(image, stepSize, windowSize, height, width, count):
#     for y in range(0, image.shape[0], stepSize):
#         for x in range(0, image.shape[1], stepSize):
#             if (y + windowSize[1]) <= height and (x + windowSize[0]) <= width:  # 没超出下边界，也超出下边界
#                 slide = image[y:y + windowSize[1], x:x + windowSize[0], :]
#                 slide_shrink = cv2.resize(slide, (256, 256), interpolation=cv2.INTER_AREA)
#                 # slide_shrink_gray = cv2.cvtColor(slide_shrink,cv2.COLOR_BGR2GRAY)
#                 imageio.imwrite("/bigdata/projects/beidi/git/vpu-tilt/slide/" + str(count) + '.png', slide_shrink)
#                 count = count + 1  # count持续加1
#             if (y + windowSize[1]) > height and (x + windowSize[0]) > width:  # 超出右边界，但没超出下边界 或者 超出下边界，但没超出右边界
#                 continue
#             if (y + windowSize[1]) > height and (x + windowSize[0]) <= width:  # 超出下边界，也超出下边界
#                 break
#     return count
#
# if __name__ == "__main__":
#     stepSize = int(1 * 256)  # 步长就是0.5倍的滑窗大小
#     windowSize = [256, 256]  # 滑窗大小
#     path = r'/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15328'  # 文件路径
#     count = 0
#
#     # filelist = os.listdir(path)  # 列举图片名
#     # image = openslide.open_slide(path + '.svs')
#     image = openslide.OpenSlide(path + '.svs')
#
#     image = np.array(image.read_region((0, 0), 0, image.dimensions))
#     height, width = image.shape[:2]
#     print(height, width)
#     size1 = (int(round(width / 256) * 256), int(round(height / 256) * 256))  # round-四舍五入 ;int-图像大小必须是整数
#     print(size1)
#     img_shrink = cv2.resize(image, size1, interpolation=cv2.INTER_AREA)  # 改变图像大小
#     count = sliding_window(img_shrink, stepSize, windowSize, size1[1], size1[0],count)  # count要返回，不然下一个图像滑窗会覆盖原来的图像
#
#     # for item in filelist:
#     #     total_num_file = len(filelist)  # 单个文件夹内图片的总数
#     #     if item.endswith('.jpg') or item.endswith('.JPG'):  # 查询文件后缀名
#     #         image = cv2.imread(item)
#     #         height, width = image.shape[:2]
#     #         print(height, width)
#     #
#     #         size1 = (int(round(width / 256) * 256), int(round(height / 256) * 256))  # round-四舍五入 ;int-图像大小必须是整数
#     #         print(size1)
#     #         img_shrink = cv2.resize(image, size1, interpolation=cv2.INTER_AREA)  # 改变图像大小
#     #         count = sliding_window(img_shrink, stepSize, windowSize, size1[1], size1[0],
#     #                                count)  # count要返回，不然下一个图像滑窗会覆盖原来的图像
'''
# https://youtu.be/QntLBvUZR5c
"""
OpenSlide can read virtual slides in several formats:

Aperio (.svs, .tif)
Hamamatsu (.ndpi, .vms, .vmu)
Leica (.scn)
MIRAX (.mrxs)
Philips (.tiff)
Sakura (.svslide)
Trestle (.tif)
Ventana (.bif, .tif)
Generic tiled TIFF (.tif)

OpenSlide allows reading a small amount of image data at the resolution
closest to a desired zoom level.

pip install openslide-python

then download the latest windows binaries
https://openslide.org/download/

Extract the contents to a place that you can locate later.

If you are getting the error: [WinError 126] The specified module could not be found

Open the lowlevel.py file located in:
    lib\site-packages\openslide

Add this at the top, after from __future__ import division, in the lowlevel.py
os.environ['PATH'] = "path+to+binary" + ";" + os.environ['PATH']
path+to+binary is the path to your windows binaries that you just downloaded.

In my case, it looks like this.

import os
os.environ['PATH'] = "C:/Users/Admin/anaconda3/envs/py37/lib/site-packages/openslide/openslide-win64-20171122/bin" + ";" + os.environ['PATH']

A few useful commands to locate the sitepackages directory

import sys
for p in sys.path:
    print(p)


"""

# import pyvips
from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Load the slide file (svs) into an object.
slide = open_slide("/bigdata/projects/beidi/dataset/urine/slide_cancer/BD21-17908.svs")

slide_props = slide.properties
print(slide_props)

print("Vendor is:", slide_props['openslide.vendor'])
print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

# Objective used to capture the image
objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
print("The objective power is: ", objective)

# get slide dimensions for the level 0 - max resolution level
slide_dims = slide.dimensions
print(slide_dims)

# Get a thumbnail of the image and visualize
# slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
# slide_thumb_600.show()

# Convert thumbnail to numpy array
# slide_thumb_600_np = np.array(slide_thumb_600)
# plt.figure(figsize=(8, 8))
# plt.imshow(slide_thumb_600_np)

# Get slide dims at each level. Remember that whole slide images store information
# as pyramid at various levels
dims = slide.level_dimensions
num_levels = len(dims)
print("Number of levels in this image are:", num_levels)
print("Dimensions of various levels in this image are:", dims)

# By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("Each level is downsampled by an amount of: ", factors)

# Copy an image from a level
level3_dim = dims[2]
# Give pixel coordinates (top left pixel in the original large image)
# Also give the level number (for level 3 we are providing a valueof 2)
# Size of your output image
# Remember that the output would be a RGBA image (Not, RGB)
level3_img = slide.read_region((0, 0), 2, level3_dim)  # Pillow object, mode=RGBA

# Convert the image to RGB
level3_img_RGB = level3_img.convert('RGB')
# level3_img_RGB.show()

# Convert the image into numpy array for processing
level3_img_np = np.array(level3_img_RGB)
# plt.imshow(level3_img_np)

# Return the best level for displaying the given downsample.
SCALE_FACTOR = 32
best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)
# Here it returns the best level to be 2 (third level)
# If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)
#################################

# Generating tiles for deep learning training or other processing purposes
# We can use read_region function and slide over the large image to extract tiles
# but an easier approach would be to use DeepZoom based generator.
# https://openslide.org/api/python/

from openslide.deepzoom import DeepZoomGenerator

# Generate object for tiles using the DeepZoomGenerator

# Here, we have divided our svs into tiles of size 256 with no overlap.

# The tiles object also contains data at many levels.
# To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)

print("The dimensions of data in each level are: ", tiles.level_dimensions)

# Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

# How many tiles at a specific level?
level_num = 11
print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
print("This means there are ", tiles.level_tiles[level_num][0] * tiles.level_tiles[level_num][1],
      " total tiles in this level")

# # Dimensions of the tile (tile size) for a specific tile from a specific layer
# tile_dims = tiles.get_tile_dimensions(11, (3, 4))  # Provide deep zoom level and address (column, row)
#
# # Tile count at the highest resolution level (level 16 in our tiles)
# tile_count_in_large_image = tiles.level_tiles[16]  # 126 x 151 (32001/256 = 126 with no overlap pixels)
# # Check tile size for some random tile
# tile_dims = tiles.get_tile_dimensions(16, (120, 140))
# # Last tiles may not have full 256x256 dimensions as our large image is not exactly divisible by 256
# tile_dims = tiles.get_tile_dimensions(16, (125, 150))
#
# single_tile = tiles.get_tile(16, (62, 70))  # Provide deep zoom level and address (column, row)
# single_tile_RGB = single_tile.convert('RGB')
# single_tile_RGB.show()
'''
### Saving each tile to local directory
import os
from openslide import open_slide
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
import glob
from matplotlib import pyplot as plt

data_class = 'benign'
# data_class = 'atypical'
# data_class = 'suspicious'
# data_class = 'cancer'

path = r'/bigdata/projects/beidi/dataset/urine/slide_'+data_class  # 文件路径
# path = r'/fast/beidi/data/slide/'+data_class  # 文件路径
filelist = glob.glob(path  + '/BD*.svs')

import pandas as pd
import cv2 as cv

data = pd.read_excel('fold_train.xlsx')
print(data.head())
list_train0 = data['train_'+data_class].dropna().tolist()
list_test0 = data['test_'+data_class].dropna().tolist()
list_train = []
list_test = []
for i in list_train0:
    list_train.append("".join(i.split()))
for i in list_test0:
    list_test.append("".join(i.split()))

print(list_train)
print(list_test)

filelist=[
    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15730.svs'
    # '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15957.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15958.svs',
#           '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15491.svs',
#           '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15960.svs',
#           '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15959.svs'
       ]
# filelist=[
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15966.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-16109.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15801.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15960.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15959.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15487.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15958.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15868.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15480.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15962.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15302.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15491.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15492.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15493.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15663.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15809.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15956.svs',
#     '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-15957.svs',
#    '/bigdata/projects/beidi/dataset/urine/slide_benign/BD22-16110.svs',
#     ''
# ]
# filelist=[
    # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-22762.svs',
    # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-22637.svs',
    #       '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-21875.svs',
    #       '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-21225.svs',
    #       '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-21189.svs',
    #       '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-20695.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-20697.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-20624.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-20487.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-19607.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-19487.svs',
          # '/bigdata/projects/beidi/dataset/urine/slide_atypical/BD22-19387.svs',
       # ]
# filelist = [
# '/bigdata/projects/beidi/dataset/urine/slide_suspicious/BD22-12707.svs',
# '/bigdata/projects/beidi/dataset/urine/slide_suspicious/BD22-17354.svs',
# '/bigdata/projects/beidi/dataset/urine/slide_suspicious/BD22-17361.svs',
#
# ]
# for item in filelist:
#     if item.endswith('.svs'):  # 查询文件后缀名
#         # tile_dir = "/bigdata/projects/beidi/git/vpu-tilt/data64/train/" + data_class
#         if (item.split('/')[-1].split('.')[0] in list_train):
#             tile_dir = "/bigdata/projects/beidi/data/tile256/train/" + data_class
#             # tile_dir = "/fast/beidi/data/tilt128/train/" + data_class
#         elif (item.split('/')[-1].split('.')[0] in list_test):
#             tile_dir = "/bigdata/projects/beidi/data/tile256/test/"+ data_class
#             # tile_dir = "/fast/beidi/data/tilt128/train/tilt128/test/" + data_class
#         else:
#             continue
#         slide = open_slide(item)
#         slide_props = slide.properties
#         print("We are processing slide", item)
#         # print(slide_props)
#         print("Vendor is:", slide_props['openslide.vendor'])
#         print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
#         print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])
#         slide_dims = slide.dimensions
#         print(slide_dims)
#
#         tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
#         # tiles = DeepZoomGenerator(slide, tile_size=64, overlap=0, limit_bounds=False)
#         # tiles = DeepZoomGenerator(slide, tile_size=128, overlap=0, limit_bounds=False)
#         cols, rows = tiles.level_tiles[15]
#         save_dir = tile_dir +'/'+ item.split('/')[-1].split('.')[0] +'/'+tile_dir.split('/')[-1]
#         not_save_dir = tile_dir +'/'+ item.split('/')[-1].split('.')[0] +'/'+tile_dir.split('/')[-2]
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#             # os.makedirs(not_save_dir)
#             count_save = 0
#             count_not_save = 0
#             for row in range(rows-1):
#                 for col in range(cols-1):
#                     tile_name = os.path.join(save_dir, '%d_%d' % (col, row))
#                     wtile_name = os.path.join(not_save_dir, '%d_%d' % (col, row))
#                     # print("Now saving tile with title: ", tile_name)
#                     # middle
#                     temp_tile = tiles.get_tile(15, (col, row))
#                     temp_tile_RGB = temp_tile.convert('RGB')
#                     temp_tile_np = np.array(temp_tile_RGB)
#
#                     mid_point = int(temp_tile_np.shape[0]/2)
#                     # if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40,mid_point+100].mean() < 230:
#                     if temp_tile_np.std()>20 and temp_tile_np.mean()<233 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40:mid_point+100].mean() < 230:
#                     # if temp_tile_np.std()>20 and temp_tile_np.mean()<235:
#                         if (temp_tile_np[:, :, 0].mean() < 235):
#                             print(temp_tile_np.std())
#                             print("Processing tile number:", tile_name)
#                             count_save += 1
#                             plt.imsave(tile_name +'_m' + ".png", temp_tile_np)
#                     else:
#                         count_not_save += 1
#                     #right
#                     if col+0.5 < cols-1:
#                         temp_tile = tiles.get_tile(15, (col+0.5, row))
#                         temp_tile_RGB = temp_tile.convert('RGB')
#                         temp_tile_np = np.array(temp_tile_RGB)
#                         mid_point = int(temp_tile_np.shape[0]/2)
#                         # if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40,mid_point+100].mean() < 230:
#                         # if temp_tile_np.std()>20 and temp_tile_np.mean()<235 :
#                         if temp_tile_np.std()>20 and temp_tile_np.mean()<233 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40:mid_point+100].mean() < 230:
#                             if (temp_tile_np[:,:,0].mean() < 235):
#                                 print("Processing tile number:", tile_name,'_r')
#                                 count_save += 1
#                                 plt.imsave(tile_name + '_r' + ".png", temp_tile_np)
#                         else:
#                             count_not_save += 1
#                     #down
#                     if row + 0.5 < row - 1:
#                         temp_tile = tiles.get_tile(15, (col, row+0.5))
#                         temp_tile_RGB = temp_tile.convert('RGB')
#                         temp_tile_np = np.array(temp_tile_RGB)
#                         mid_point = int(temp_tile_np.shape[0]/2)
#                         # if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40,mid_point+100].mean() < 230:
#                         if temp_tile_np.std()>20 and temp_tile_np.mean()<233 and temp_tile_np[mid_point-40:mid_point+100,mid_point-40:mid_point+100].mean() < 230:
#                         # if temp_tile_np.std()>20 and temp_tile_np.mean()<235:
#                             if (temp_tile_np[:, :, 0].mean() < 235):
#                                 print("Processing tile number:", tile_name,'_d')
#                                 count_save += 1
#                                 plt.imsave(tile_name + '_d' + ".png", temp_tile_np)
#                         else:
#                             # print("NOT PROCESSING TILE:", tile_name)
#                             # plt.imsave(wtile_name + ".png", temp_tile_np)
#                             count_not_save += 1
#             print('saved tile: ',count_save)
#             print('did not saved tile: ',count_not_save)

''''''
# from PIL import Image
# import numpy as np
#
# dir = '/bigdata/projects/beidi/data/tile96/train/benign/BD22-16108/benign/17_4_m.png'
# img = Image.open(dir)
# data = np.array(img, dtype='uint8')
#
# print(data.std())


# -----------------------------------------------------------------------------------——
import random
import glob
import numpy as np
from PIL import Image
import os
file_path = '/bigdata/projects/beidi/data/tile256_rand100_new/'+'*/*' + '/BD*'
file_list = glob.glob(file_path)
# print(file_list)

for i in file_list:

    img_path = i + '/*/' + '*'
    # print(img_path)
    patch_list = glob.glob(img_path)
    if len(patch_list)<100:
        print(i)
        print(len(patch_list))

    # for j in patch_list:
    #     img = Image.open(j)
    #     data = np.array(img, dtype='uint8')
    #     if data.shape[1] <256 or data.shape[0] <256:
    #         print(data.shape)
    #         os.remove(j)

    # patch_list = glob.glob(img_path)
    # random.shuffle(file_list)
    # save_list = patch_list[:100]
    # for j in patch_list:
    #     img = Image.open(j)
    #     data = np.array(img, dtype='uint8')
    #     if j not in save_list:
    #         os.remove(j)
# -----------------------------------------------------------------------------------——
# 数据增强工具
# import Augmentor
# import glob
#
# file_path = '/bigdata/projects/beidi/data/tile256_rand100_new/'+'*/*' + '/BD*'
# file_list = glob.glob(file_path)
#
#
# for i in file_list:
#     img_path = i + '/*/' + '*'
#     # print(img_path)
#     patch_list = glob.glob(img_path)
#     l = len(patch_list)
#     if len(patch_list) < 100:
#         print(i)
#         print(len(patch_list))
# # 确定原始图像存储路径以及掩码mask文件存储路径
#         p = Augmentor.Pipeline(os.path.join(i,i.split('/')[-2]))
#         print(os.path.join(i,i.split('/')[-2]))
#         p.ground_truth(os.path.join(i,i.split('/')[-2]))
#
#         # 图像旋转：按照概率0.8执行，最大左旋角度10，最大右旋角度10
#         # rotate操作默认在对原图像进行旋转之后进行裁剪，输出与原图像同样大小的增强图像
#         p.rotate(probability=0.8, max_left_rotation=10, max_right_rotation=10)
#
#         # 图像上下镜像： 按照概率0.5执行
#         p.flip_top_bottom(probability=0.5)
#
#         # 图像左右镜像： 按照概率0.5执行
#         p.flip_left_right(probability=0.5)
#
#         # 图像等比缩放，按照概率1执行，整个图片放大，像素变多
#         # p.scale(probability=1, scale_factor=1.3)
#
#         # 图像放大：放大后像素不变，先根据percentage_area放大，后按照原图像素大小进行裁剪
#         # 按照概率0.4执行，面积为原始图0.9倍
#         p.zoom_random(probability=0.4, percentage_area=0.9)
#
#         # 最终扩充的数据样本数
#         p.sample(100-len(patch_list))
#
#
