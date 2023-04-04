# parameters
# DATASET="liver2_mini"
DATASET="carotid-mini"
# available colormaps: https://matplotlib.org/stable/gallery/color/colormap_reference.html

import numpy as np
from PIL import Image
from os import listdir
from os import path
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib import cm

normalize_per_image = True

# read all existing images
dir_input_images="../extract/data/"+DATASET+"/images/"
all_images = [f for f in listdir(dir_input_images) if path.isfile(path.join(dir_input_images, f))]
print(all_images[:5])
all_images.sort()


# open image
# image_name_full='real02.jpg' # for liver mini
image_name_full='img0001.jpg' # for carotid
image_path=dir_input_images+image_name_full
im = np.array(Image.open(image_path))
# plt.imshow(im)
# plt.show()

# normalizing step
if normalize_per_image == True:
    # normalize per image
    max_value = im.max()
else:
    # normalize per dataset
    max_value = max(all_images)  # is a string! so  wrong! need to read all the images and then find the maximum
print("max value=", max_value)

# im2 = np.array(im * 1.0/ max_value)
im2 = im * 1.0 / max_value
print("new max=", im2.max())

diff = im - im2



# # change to colormap
# colored_image = Image.fromarray(cm.jet(im, bytes=True))
# plt.imshow(colored_image)
# plt.show()

# colored_image = Image.fromarray(cm.jet(im2, bytes=True))
# plt.imshow(colored_image)
# plt.show()

# # change to colormap
# colored_image3 = Image.fromarray(cm.jet(diff, bytes=True))
# plt.imshow(colored_image3)
# plt.show()



# # apply to all images
# for image_name_full in all_images:
#   print(image_name_full)
#   image_path=dir_input_images+image_name_full
#   im = np.array(Image.open(image_path))
#   im = im * 1.0/im.max()
#   colored_image = Image.fromarray(cm.jet(im, bytes=True)).convert('RGB')
#   # plt.imshow(colored_image)
#   # plt.savefig(dir_colored_images+image_name_full) #saves also plot and ticks etc
#   colored_image.save(dir_colored_images+image_name_full)

max_value = im.max()

# apply to all images
for image_name_full in all_images:
  print(image_name_full)
  image_path=dir_input_images+image_name_full
  im = np.array(Image.open(image_path))
  if im.max() > max_value:
     max_value = im.max()

print("New value per dataset=", max_value)


invert = True

if invert == False:
    # create a folder for colored images
    dir_colored_images="../extract/data/"+DATASET+"-color-norm/images/"
    if not path.exists(dir_colored_images):
        makedirs(dir_colored_images)

    # apply to all images
    for image_name_full in all_images:
        print(image_name_full)
        image_path=dir_input_images+image_name_full
        im = np.array(Image.open(image_path))
        im = im * 1.0/max_value
        colored_image = Image.fromarray(cm.jet(im, bytes=True)).convert('RGB')
        # plt.imshow(colored_image)
        # plt.savefig(dir_colored_images+image_name_full) #saves also plot and ticks etc
        colored_image.save(dir_colored_images+image_name_full)

else:
    # create a folder for colored images
    dir_colored_images="../extract/data/"+DATASET+"-color-norm-inverted/images/"
    if not path.exists(dir_colored_images):
        makedirs(dir_colored_images)

    # apply to all images
    for image_name_full in all_images:
        print(image_name_full)
        image_path=dir_input_images+image_name_full
        im = np.array(Image.open(image_path))
        im = 1 - im * 1.0/max_value
        colored_image = Image.fromarray(cm.jet(im, bytes=True)).convert('RGB')
        # plt.imshow(colored_image)
        # plt.savefig(dir_colored_images+image_name_full) #saves also plot and ticks etc
        colored_image.save(dir_colored_images+image_name_full)
