#!/usr/bin/env python
# coding: utf-8

# In[3]:



# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


from patchify import patchify
if not os.path.exists('bags'):
    os.makedirs('bags')
white = [230,230,230]
diff = 25
boundaries = [([white[2] - diff, white[1] - diff, white[0] - diff],
               [white[2]+25,white[1]+25,white[0]+25])]
for file in os.listdir('Gleason_masks_train'):
    bag_name = file[5:-4]
    if bag_name.startswith('ZT80'):
        continue
    else:
        os.makedirs(os.path.join('bagso',bag_name))
        mask = cv2.imread(os.path.join('Gleason_masks_train',file))
    img = cv2.imread(os.path.join('TMA_images',bag_name+'.jpg'))
    img_patches = patchify(img,(310,310,3),step=310)
    mask_patches = patchify(mask,(310,310,3),step=310)
    indices = []
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            img_patch = img_patches[i,j,0]
            mask_patch = mask_patches[i,j,0]
            for (lower,upper) in boundaries:
                lower = np.array(lower,dtype=np.uint8)
                upper = np.array(upper,dtype=np.uint8)
                masko = cv2.inRange(img_patch,lower,upper)
                ratio_white = cv2.countNonZero(masko) / (mask_patch.size/3)
                if ratio_white < 0.25:
                    print(ratio_white)
                    cv2.imwrite(os.path.join('bags/'+bag_name,str(i) +'_'+str(j)+'_.jpg'),img_patch)


# In[6]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
white = [255,255,255]
diff = 20
boundaries_white = [([white[2] - diff, white[1] - diff, white[0] - diff],
               [white[2],white[1],white[0]])]
green = [20,255,20]
boundaries_green = [([green[2] - diff, green[1] - diff, green[0] - diff],
               [green[2],green[1],green[0]])]
blue = [20,20,255]
boundaries_blue = [([blue[2] - diff, blue[1] - diff, blue[0] - diff],
               [blue[2],blue[1],blue[0]])]
yellow = [255,255,20]
boundaries_yellow = [([yellow[2] - diff,yellow[1] - diff, yellow[0] - diff],
               [yellow[2],yellow[1],yellow[0]])]
red = [255,20,20]
boundaries_red = [([red[2] - diff, red[1] - diff, red[0] - diff],
               [red[2],red[1],red[0]])]
train_labels = {}
for file in os.listdir('Gleason_masks_train'):
    img = cv2.imread(os.path.join('Gleason_masks_train',file))
    ratios = []
    for boundaries in [boundaries_green,boundaries_blue,boundaries_yellow,boundaries_red]:
        for (lower,upper) in boundaries:
            lower = np.array(lower,dtype=np.uint8)
            upper = np.array(upper,dtype=np.uint8)
            mask = cv2.inRange(img,lower,upper)
            ratio = cv2.countNonZero(mask) / (img.size/3)
            ratios.append(ratio)
    sorted_ratios = np.argsort(ratios)
    if sorted_ratios[-1] == 0:
        train_labels[file[5:-4]] = 0
#     else:
#         train_labels[file[5:-4]] = 1
    elif sorted_ratios[-1] == 1 and ratios[sorted_ratios[-2]] == 0:
        train_labels[file[5:-4]] = 1
    elif sorted_ratios[-1] == 1 and sorted_ratios[-2] == 2:
        train_labels[file[5:-4]] = 2
    elif sorted_ratios[-1] == 1 and sorted_ratios[-2] == 3:
        train_labels[file[5:-4]] = 3
    elif sorted_ratios[-1] == 2 and ratios[sorted_ratios[-2]] == 0:
        train_labels[file[5:-4]] = 4
    elif sorted_ratios[-1] == 2 and sorted_ratios[-2] == 1:
        train_labels[file[5:-4]] = 5
    elif sorted_ratios[-1] == 2 and sorted_ratios[-2] == 3:
        train_labels[file[5:-4]] = 6
    elif sorted_ratios[-1] == 3 and ratios[sorted_ratios[-2]] == 0:
        train_labels[file[5:-4]] = 7
    elif sorted_ratios[-1] == 3 and sorted_ratios[-2] == 1:
        train_labels[file[5:-4]] = 8
    elif sorted_ratios[-1] == 3 and sorted_ratios[-2] == 2:
        train_labels[file[5:-4]] = 9



import pandas as pd
df = pd.DataFrame(index=train_labels.keys())
df['Labels'] = train_labels.values()
df.head()


# In[9]:


df.to_csv('train_labels.csv')


# In[10]:



