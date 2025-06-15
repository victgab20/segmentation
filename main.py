import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import threshold_otsu
import re
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv()

def dice_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    if total == 0:
        return 1.0 
    return 2.0 * intersection / total

root_path = os.getenv("SOURCE_DATA")
resultados = []

point_folders = [
    x for x in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, x)) and re.match(r'^Point\d+$', x)
]

for point_folder in sorted(point_folders, key=lambda x: int(x.replace("Point", ""))):
    folder_path = os.path.join(root_path, point_folder)

    path_seg = os.path.join(folder_path, 'Segmentation.tif')
    path_seg_interior = os.path.join(folder_path, 'SegmentationInterior.tif')

    try:
        mask_seg = io.imread(path_seg)
        mask_seg = (mask_seg > 0).astype(np.uint8)

        mask_seg_interior = io.imread(path_seg_interior)
        mask_seg_interior = (mask_seg_interior > 0).astype(np.uint8)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar máscaras em {point_folder}: {e}")
        continue

    for filename in os.listdir(folder_path):
        if not (filename.lower().endswith('.tif') or filename.lower().endswith('.tiff')):
            continue
        if 'Segmentation' in filename:
            continue

        image_path = os.path.join(folder_path, filename)

        try:
            image = io.imread(image_path)

            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

            thresh = threshold_otsu(image_norm)
            bin_image = (image_norm > thresh).astype(np.uint8)

            if bin_image.shape != mask_seg.shape:
                print(f"[AVISO] Dimensão diferente em {filename} dentro de {point_folder}")
                continue

            dice_seg = dice_score(bin_image, mask_seg)
            dice_seg_interior = dice_score(bin_image, mask_seg_interior)

            resultados.append({
                'Pasta': point_folder,
                'Arquivo': filename,
                'Dice_Segmentation': dice_seg,
                'Dice_SegmentationInterior': dice_seg_interior
            })

        except Exception as e:
            print(f"[ERRO] Erro ao processar {filename} em {point_folder}: {e}")

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('dice_tnbc.csv', index=False)
print("✅ Resultado salvo em dice_tnbc.csv")
