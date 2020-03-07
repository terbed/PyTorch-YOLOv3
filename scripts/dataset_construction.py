import os
import random
import numpy as np

path_img = '/media/terbe/sztaki/DATA/BabyCropper/data/images/'
path_labels = '/media/terbe/sztaki/DATA/BabyCropper/data/labels/'

# Create sets for names
img_set = set()
label_set = set()

# Fill image set with names
with os.scandir(path_img) as entries:
    for entry in entries:
        current_name = entry.name.split('.')[0]
        print(current_name)
        img_set.add(current_name)


# # Create empty labels for images starting with empty
# for name in img_set:
#     prename = name.split('_')[0]
#     if prename == 'empty':
#         tmp = name.split('.')[0]
#         out_name = path_labels+tmp+'.txt'
#         print(out_name)
#         open(out_name, 'a').close()


# Fill label set with names
with os.scandir(path_labels) as entries:
    for entry in entries:
        current_name = entry.name.split('.')[0]
        print(current_name)
        label_set.add(current_name)


# Filter out mismatches in label and images names
intersec = img_set.intersection(label_set)

###############################################
# Select training and validation set randomly
###############################################
arr = list(intersec)
random.shuffle(arr)     # shuffle list
n = len(arr)
print(f'Number of samples: {n}')

n_train = int(n * 0.75)   # number of training samples
print(f'Number of training samples: {n_train}')

train = []
val = []
counter = 0
for item in arr:
    if counter < n_train:
        train.append('data/custom/images/' + item + '.png')
    else:
        val.append('data/custom/images/' + item + '.png')
    counter += 1

print(len(train))
print(len(val))
np.savetxt('train.txt', train, fmt='%s')
np.savetxt('valid.txt', val, fmt='%s')
