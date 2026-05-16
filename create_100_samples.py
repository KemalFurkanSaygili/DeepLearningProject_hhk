import os
import random
import shutil

src_dir = '../NEU-DET/train/images'
dst_dir = 'NEU-DET_100_samples'
os.makedirs(dst_dir, exist_ok=True)

classes = os.listdir(src_dir)
images_per_class = 100 // len(classes)
total_copied = 0

for c in classes:
    c_path = os.path.join(src_dir, c)
    if not os.path.isdir(c_path):
        continue
    c_dst = os.path.join(dst_dir, c)
    os.makedirs(c_dst, exist_ok=True)
    
    imgs = os.listdir(c_path)
    selected = random.sample(imgs, min(images_per_class, len(imgs)))
    
    for img in selected:
        shutil.copy(os.path.join(c_path, img), os.path.join(c_dst, img))
        total_copied += 1

print(f'Successfully copied {total_copied} images to {dst_dir}')
