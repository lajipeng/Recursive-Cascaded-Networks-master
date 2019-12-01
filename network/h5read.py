import h5py  #导入工具包
import numpy as np
import json
import os
#HDF5的写入：
# imgData = np.zeros((30,3,128,256))
# f = h5py.File('HDF5_FILE.h5','w')   #创建一个h5文件，文件指针是f
# f['data'] = imgData                 #将数据写入文件的主键data下面
# f['labels'] = range(100)            #将数据写入文件的主键labels下面
# f.close()                           #关闭文件


path = './datasets/msd_dataset.h5'
file = h5py.File(path, "r")

group = file['datasets']

print('2')
       

# dataset = './datasets/liver.json'
# with open(os.path.join(dataset), 'r') as f:
#         cfg = json.load(f)
#         image_size = cfg.get('image_size', [128, 128, 128])
#         image_type = cfg.get('image_type')