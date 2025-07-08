import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dice_tnbc_segmentation_interior_topN.csv")

mdice_por_topn = df.groupby("TopN_Canais")["Dice_SegmentationInterior"].mean().reset_index()

mdice_por_topn.columns = ["TopN_Canais", "mDice"]

print(mdice_por_topn)

plt.figure(figsize=(8, 5))
plt.plot(mdice_por_topn["TopN_Canais"], mdice_por_topn["mDice"], marker='o')
plt.xlabel("Número de Canais Usados (Top N)")
plt.ylabel("mDice")
plt.title("mDice por número de canais utilizados")
plt.grid(True)
plt.tight_layout()
plt.show()
