import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

data = pd.read_csv("dice_tnbc.csv")
print(data)

top10_seg = (
    data.sort_values(by=["Pasta", "Dice_Segmentation"], ascending=[True, False])
         .groupby("Pasta")
         .head(10)
)

top10_seg_int = (
    data.sort_values(by=["Pasta", "Dice_SegmentationInterior"], ascending=[True, False])
         .groupby("Pasta")
         .head(10)
)


top10_seg.to_csv("top10_dice_segmentation.csv", index=False)
top10_seg_int.to_csv("top10_dice_segmentation_interior.csv", index=False)

print("Arquivos salvos com sucesso:")
print("- top10_dice_segmentation.csv")
print("- top10_dice_segmentation_interior.csv")
