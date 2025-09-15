import os
from convert_png_to_tif import convert_png_to_tif


# tif = os.path.join(os.getcwd(), "data/train/images")
# mask = os.path.join(os.getcwd(), "data/train/annotations")
# out = os.path.join(os.getcwd(), "data/train/geo_annotations")

tif = os.path.join(os.getcwd(), "data/klin/images_patches")
# mask = os.path.join(os.getcwd(), "data/klin/annotations")
# out = os.path.join(os.getcwd(), "data/klin/geo_annotations")
mask = os.path.join(os.getcwd(), "data/klin/blackend_annotations")
out = os.path.join(os.getcwd(), "data/klin/geo_blackend_annotations")

# tif = os.path.join(os.getcwd(), "data/train/images")
# mask = os.path.join(os.getcwd(), "data/train/blackend_annotations")
# out = os.path.join(os.getcwd(), "data/train/geo_blackend_annotations")
convert_png_to_tif(tif, mask, out)