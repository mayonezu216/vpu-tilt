import os

import numpy as np
import glob
from PIL import Image
import imageio
folder = '/fast/beidi/crossvit/data/scale256/'+'*/*/BD*/*'
folderlist = glob.glob(folder)
print(folderlist)

for i in folderlist:
    print(i)
    os.makedirs(i.split('256')[0]+'256to128'+i.split('256')[1])
assert 2==3
file_path = '/fast/beidi/crossvit/data/scale256/'+'*/*' + '/BD*/*/*.png'
file_list = glob.glob(file_path)
for i in file_list:
    print(i)
    # if not os.path.exists(i.split()
    img = Image.open(i)
    # print(i.split('256')[0]+'256to128'+ i.split('256')[1].split('.')[0] + '_a' + ".png")

    data = np.array(img, dtype='uint8')
    imageio.imwrite(i.split('256')[0]+'256to128'+ i.split('256')[1].split('.')[0] + '_a' + ".png", data[:128,:128])
    imageio.imwrite(i.split('256')[0]+'256to128'+ i.split('256')[1].split('.')[0] + '_b' + ".png", data[128:,:128])
    imageio.imwrite(i.split('256')[0]+'256to128'+ i.split('256')[1].split('.')[0] + '_c' + ".png", data[:128,128:])
    imageio.imwrite(i.split('256')[0]+'256to128'+ i.split('256')[1].split('.')[0] + '_d' + ".png", data[128:,128:])
