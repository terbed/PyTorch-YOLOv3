import shutil
import os


############################
# Read lines from file
############################
file = open('../valid.txt', 'r')                       # open file
lines = file.read().split('\n')                     # store lines in list
lines = [x for x in lines if len(x) > 0]            # get rid of empty lines
lines = [x.rstrip().lstrip() for x in lines]        # get rid of frige white spaces
lines = [x for x in lines if x[0] != '#']           # get rid of comments
lines = [x.split('/')[-1] for x in lines]           # keep only the names

root = '/media/terbe/sztaki/DATA/BabyCropper/data/images/'
out_dir = '/media/terbe/sztaki/DATA/BabyCropper/data/val_images'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for name in lines:
    shutil.copy(root+name, out_dir)
    print(f'cp {root+name} -> {out_dir}')

