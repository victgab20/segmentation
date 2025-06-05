import pandas as pd
import matplotlib.pyplot as plt

resultados = pd.read_csv("resultados_segmentacao.csv")

# resultados.rename()

resultados = resultados.rename(columns={"0": "Canal","9.523896324026282e-05": "Dice Médio" })


plt.figure(figsize=(14, 10))
plt.bar(resultados["Canal"], resultados["Dice Médio"])
plt.title("Análise dos Valores Médios do Dice por Canal")
plt.xlabel("Canal")
plt.ylabel("Dice Médio")
plt.xticks(resultados["Canal"]) 
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()