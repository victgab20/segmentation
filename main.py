import os
from glob import glob
import numpy as np
import tifffile as tiff
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


import csv

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RESULTADO_CSV = "resultados_segmentacao.csv"

IMAGENS_DIR = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\raw_images"
MASCARAS_DIR = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\segmentation\masks"
TAMANHO_IMG = 512
BATCH_SIZE = 4
N_EPOCAS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preparar_pares(imagens_dir, mascaras_dir):
    pares = []
    for caminho_tiff in glob(os.path.join(imagens_dir, "*.tiff")):
        nome_base = os.path.splitext(os.path.basename(caminho_tiff))[0]
        pasta_mask = os.path.join(mascaras_dir, nome_base)
        caminho_mask = os.path.join(pasta_mask, "segmentation.png")
        if os.path.exists(caminho_mask):
            pares.append({"imagem": caminho_tiff, "mascara": caminho_mask})
        else:
            print(f"Máscara NÃO encontrada para: {nome_base}")
    print(f"Total de pares encontrados: {len(pares)}")
    return pares


class SegmentacaoPorCanal(Dataset):
    def __init__(self, lista_pares, canal, transform=None):
        self.lista = lista_pares
        self.canal = canal
        self.transform = transform

    def __getitem__(self, idx):
        img = tiff.imread(self.lista[idx]['imagem'])[self.canal].astype(np.float32)
        img /= 65535.0
        mask = np.array(Image.open(self.lista[idx]['mascara'])) > 0
        mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img.unsqueeze(0), mask.unsqueeze(0)

    def __len__(self):
        return len(self.lista)

transform_treino = A.Compose([
    A.Resize(TAMANHO_IMG, TAMANHO_IMG),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

transform_val = A.Compose([
    A.Resize(TAMANHO_IMG, TAMANHO_IMG),
    A.Normalize(),
    ToTensorV2()
])

def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1.0
    intersect = (pred * target).sum()
    return ((2.0 * intersect + smooth) / (pred.sum() + target.sum() + smooth)).item()


def treinar_modelo(model, loader_treino, loader_val, epocas=5):
    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for epoca in range(epocas):
        model.train()
        for x, y in loader_treino:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = bce(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        dice_scores = []
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                score = dice_score(out, y)
                dice_scores.append(score)

        print(f"Época {epoca+1}/{epocas} - Dice médio (val): {np.mean(dice_scores):.4f}")


if __name__ == "__main__":
    pares = preparar_pares(IMAGENS_DIR, MASCARAS_DIR)
    train_val, test = train_test_split(pares, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    resultados = []

    for canal in range(44):
        print(f"\n Canal {canal}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"modelo_canal_{canal}.pth")
        if os.path.exists(checkpoint_path):
            print("Checkpoint encontrado. Pulando treinamento.")
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=1,
                classes=1,
                activation=None
            )
            model.load_state_dict(torch.load(checkpoint_path))
            model.to(DEVICE)
        else:
            print("Iniciando treinamento...")
            ds_train = SegmentacaoPorCanal(train, canal, transform=transform_treino)
            ds_val = SegmentacaoPorCanal(val, canal, transform=transform_val)

            train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=2)

            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=1,
                classes=1,
                activation=None
            )

            treinar_modelo(model, train_loader, val_loader, epocas=N_EPOCAS)
            torch.save(model.state_dict(), checkpoint_path)
            print("Modelo salvo.")

        ds_val = SegmentacaoPorCanal(val, canal, transform=transform_val)
        val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=2)

        model.eval()
        all_scores = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                score = dice_score(out, y)
                all_scores.append(score)

        dice_medio = np.mean(all_scores)
        resultado_canal = {'canal': canal, 'dice_val': dice_medio}
        resultados.append(resultado_canal)

        with open(RESULTADO_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            if canal == 0 and not os.path.exists(RESULTADO_CSV):
                writer.writerow(['canal', 'dice_val'])
            writer.writerow([canal, dice_medio])

    print("\n Resultados finais por canal (ordenados):")
    for r in sorted(resultados, key=lambda x: x['dice_val'], reverse=True):
        print(f"Canal {r['canal']:>2} — Dice Médio: {r['dice_val']:.4f}")