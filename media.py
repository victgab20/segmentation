import pandas as pd

resultDaniel = pd.read_csv("dice_canais_combinados.csv")
resultVictor = pd.read_csv("dice_canais_combinados2.csv")

mDice_daniel = resultDaniel["Dice_SegmentationInterior"].mean()
mDice_victor = resultVictor["Dice_SegmentationInterior"].mean()

print(f"mDice (Daniel): {mDice_daniel:.4f}")
print(f"mDice (Victor): {mDice_victor:.4f}")