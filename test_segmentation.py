# dice_segmentation_topN.py

import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import threshold_otsu
from dotenv import load_dotenv
import re

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
df = pd.read_csv('dice_tnbc.csv')

media_dice = df.groupby('Arquivo')['Dice_Segmentation'].mean()
top_canais = media_dice.sort_values(ascending=False).index.tolist()

resultados = []

for n in range(2, 11):
    canais_n = top_canais[:n]
    
    for pasta in sorted(df['Pasta'].unique(), key=lambda x: int(x.replace('Point', ''))):
        folder_path = os.path.join(root_path, pasta)

        try:
            mask_seg = io.imread(os.path.join(folder_path, 'Segmentation.tif'))
            mask_seg = (mask_seg > 0).astype(np.uint8)
        except Exception as e:
            print(f"[ERRO] Máscara ausente em {pasta}: {e}")
            continue

        imagens = []
        for canal in canais_n:
            path_canal = os.path.join(folder_path, canal)
            if not os.path.exists(path_canal):
                continue
            try:
                img = io.imread(path_canal)
                img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
                imagens.append(img_norm)
            except Exception:
                continue

        if len(imagens) != n:
            continue

        imagem_comb = np.mean(imagens, axis=0)
        thresh = threshold_otsu(imagem_comb)
        bin_comb = (imagem_comb > thresh).astype(np.uint8)

        if bin_comb.shape != mask_seg.shape:
            continue

        dice_seg = dice_score(bin_comb, mask_seg)

        resultados.append({
            'Pasta': pasta,
            'TopN_Canais': n,
            'Canais_Usados': ', '.join(canais_n),
            'Dice_Segmentation': dice_seg
        })

df_result = pd.DataFrame(resultados)
df_result.to_csv('dice_tnbc_segmentation_topN.csv', index=False)
print("✅ Resultado salvo em dice_tnbc_segmentation_topN.csv")
