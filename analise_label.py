import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import os 
from dotenv import load_dotenv

load_dotenv()

source = os.getenv("SOURCE_DATA")

seg = tiff.imread(source+r"\Point41\SegmentationInterior.tif")
marker = tiff.imread(source+r"\Point41\HLA_Class_1.tif")

img_normalizada = (marker - marker.min()) / (marker.max() - marker.min() + 1e-8)

thresh = threshold_otsu(img_normalizada)
bin_img = (img_normalizada > thresh).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(seg, cmap='gray')
plt.title("Segmentação")
plt.axis('off')

plt.figure(figsize=(6, 6))
plt.imshow(bin_img, cmap='gray')
plt.title("biomarcador binarizado")
plt.axis('off')

overlay = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
overlay[..., 0] = (seg > 0).astype(np.uint8) * 255
overlay[..., 1] = bin_img * 255   

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.title("Sobreposição: Segmentação (vermelho) + dsDNA bin (verde)")
plt.axis('off')
plt.show()
