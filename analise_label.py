import os
import numpy as np
import tifffile as tiff
from skimage.filters import threshold_otsu
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def dice_score(mask1, mask2):
    """Calcula a similaridade de Dice entre duas máscaras booleanas"""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 1.0  # Se ambas estiverem vazias
    return 2.0 * intersection / total

# Caminhos
base_path = r"C:\Users\victo\Downloads\Mestrado\TNBC\TNBCShareData\Point1"
label_path = r"C:\Users\victo\Downloads\Mestrado\TNBC\TNBCcellTypes\P1_labeledImage.tiff"

# Carrega a imagem label e cria máscara para label 10
label = tiff.imread(label_path)
tumor_mask = (label == 10)  # Shape esperado (2048, 2048)

# Lista de arquivos válidos (exclui Segmentation e SegmentationInterior)
files = [
    f for f in os.listdir(base_path)
    if f.endswith('.tif') and 'Segmentation' not in f
]

dice_results = {}

for fname in files:
    fpath = os.path.join(base_path, fname)
    img = tiff.imread(fpath)

    # Normalização: Otsu threshold para binarização
    if img.shape != (2048, 2048):
        print(f"Aviso: {fname} tem shape {img.shape}, esperado (2048, 2048). Ignorando.")
        continue

    try:
        thresh = threshold_otsu(img)
        binary_mask = img > thresh
    except ValueError:
        print(f"Erro ao calcular Otsu em {fname}. Pulando.")
        continue

    # Calcula Dice
    dice = dice_score(tumor_mask, binary_mask)

    # Normaliza para intervalo [0, 16]
    dice_results[fname] = dice 

# Exibe os resultados
for k, v in dice_results.items():
    print(f"{k}: Dice normalizado = {v:.3f}")
