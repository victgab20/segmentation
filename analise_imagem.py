import tifffile as tiff
import matplotlib.pyplot as plt


caminho_imagem = r"C:\Users\victo\Downloads\Mestrado\Keren_et_al\raw_images\TA459_multipleCores2_Run-4_Point9.tiff"

imagem = tiff.imread(caminho_imagem)

print("Formato da imagem:", imagem.shape)
print("Tipo de dado:", imagem.dtype)


plt.imshow(imagem[0], cmap='gray')
plt.title('Slice 0 da imagem TIFF')
plt.axis('off')
plt.show()