import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
from skimage.filters import threshold_otsu

load_dotenv()
source = os.getenv("SOURCE_DATA")
folder_path = os.path.join(source, "Point17")

bio_files = [
    f for f in os.listdir(folder_path)
    if f.endswith(".tif") and "Segmentation" not in f
]
bio_files.sort()

for f in bio_files:
    path = os.path.join(folder_path, f)
    image = tiff.imread(path)

    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

    thresh = threshold_otsu(image_norm)
    bin_image = (image_norm > thresh).astype(np.uint8)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_norm, cmap='gray')
    plt.title(f"Normalizada - {f}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(bin_image, cmap='gray')
    plt.title("Binarizada (Otsu)")
    plt.axis('off')

    plt.show()
