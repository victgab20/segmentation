import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


caminho_imagem = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\raw_images\TA459_multipleCores2_Run-4_Point27.tiff"
caminho_mascara = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\segmentation\masks\TA459_multipleCores2_Run-4_Point27\segmentation.png"
caminho_mascara_interna = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\segmentation\masks\TA459_multipleCores2_Run-4_Point27\segmentation_interior.png"

imagem = tiff.imread(caminho_imagem)
# mascara = png.imread(caminho_mascara)

mascara = Image.open(caminho_mascara).convert('L')
mascara_interior = Image.open(caminho_mascara_interna).convert('L')
mascara_np = np.array(mascara)
mascara_interior_np = np.array(mascara_interior)

print("Formato da imagem:", imagem.shape)
print("Tipo de dado:", imagem.dtype)

# imagem = np.transpose(imagem, (2, 0, 1))
# plt.figure(figsize=(8, 8))
# plt.imshow(mascara_np, cmap='gray')
# plt.title("Máscara PNG")
# plt.axis('off')
# plt.show()


# plt.figure(figsize=(8, 8))
# plt.imshow(mascara_interior_np, cmap='gray')
# plt.title("Máscara PNG")
# plt.axis('off')
# plt.show()

print(len(imagem))

# plt.imshow(imagem[33], cmap='gray')
# plt.title('Slice 0 da imagem TIFF')
# plt.axis('off')
# plt.show()



# num_canais = imagem.shape[0]
# ncols = 8  # número de colunas por linha
# nrows = int(np.ceil(num_canais / ncols))

# fig, axs = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows))

# for i in range(nrows * ncols):
#     ax = axs[i // ncols, i % ncols]
#     if i < num_canais:
#         ax.imshow(imagem[i], cmap='gray')
#         ax.set_title(f'Canal {i}')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
# plt.imshow(imagem[1], cmap='gray')
# plt.title('Slice 0 da imagem TIFF')
# plt.axis('off')
# plt.show()
# plt.imshow(imagem[2], cmap='gray')
# plt.title('Slice 0 da imagem TIFF')
# plt.axis('off')
# plt.show()

for i in range(44):
    plt.imshow(imagem[i], cmap='gray')
    plt.title(f'Canal {[i]}')
    plt.axis('off')
    plt.show()