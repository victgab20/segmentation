import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import threshold_otsu
import re
from tqdm import tqdm
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

canais_desejados = [
    'dsDNA.tif',
    'H3K27me3.tif',
    'H3K9ac.tif',
    'CD45.tif',
    'HLA-DR.tif',
    'Beta catenin.tif',
    'P.tif',
    'Na.tif'
]

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

    combined_bin = None
    canais_utilizados = []

    for canal in canais_desejados:
        image_path = os.path.join(folder_path, canal)
        if not os.path.exists(image_path):
            print(f"[AVISO] Canal {canal} não encontrado em {point_folder}")
            continue

        try:
            image = io.imread(image_path)
            image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            thresh = threshold_otsu(image_norm)
            bin_image = (image_norm > thresh).astype(np.uint8)

            if bin_image.shape != mask_seg.shape:
                print(f"[AVISO] Dimensão diferente em {canal} dentro de {point_folder}")
                continue

            canais_utilizados.append(canal)

            if combined_bin is None:
                combined_bin = bin_image
            else:
                combined_bin = np.logical_or(combined_bin, bin_image).astype(np.uint8)

        except Exception as e:
            print(f"[ERRO] Erro ao processar {canal} em {point_folder}: {e}")

    if combined_bin is None:
        print(f"[AVISO] Nenhum canal válido encontrado em {point_folder}")
        continue

    dice_seg = dice_score(combined_bin, mask_seg)
    dice_seg_interior = dice_score(combined_bin, mask_seg_interior)

    resultados.append({
        'Pasta': point_folder,
        'Canais_Combinados': ", ".join(canais_utilizados),
        'Dice_Segmentation': dice_seg,
        'Dice_SegmentationInterior': dice_seg_interior
    })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('dice_canais_combinados2.csv', index=False)
print("✅ Resultado salvo em dice_canais_combinados2.csv")
