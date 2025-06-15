import os
import tifffile
import pandas as pd
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv()

root_dir = os.getenv("SOURCE_DATA")

dados = []

for point in tqdm(sorted(os.listdir(root_dir))):
    point_path = os.path.join(root_dir, point)
    
    if not os.path.isdir(point_path):
        continue

    for filename in os.listdir(point_path):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        filepath = os.path.join(point_path, filename)

        try:
            img = tifffile.imread(filepath)
            dados.append({
                'Ponto': point,
                'Arquivo': filename,
                'Shape': img.shape,
                'Dtype': str(img.dtype),
                'Min': img.min(),
                'Max': img.max()
            })
        except Exception as e:
            print(f"Erro ao ler '{filepath}': {e}")
            continue

df = pd.DataFrame(dados)

df.to_csv('analise_tnbc.csv', index=False)

print("Análise final salva como 'analise_tnbc.csv'")
