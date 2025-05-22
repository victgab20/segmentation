1. Carregamento e associação dos dados
O código começa buscando todas as imagens .tiff na pasta de imagens e automaticamente associa cada uma com sua máscara de segmentação (segmentation.png) localizada em uma subpasta correspondente.

Cada imagem contém 44 canais (ou bandas), e a máscara é uma imagem binária 2D indicando a região de interesse.

2. Divisão do dataset
Os pares de imagem-máscara são divididos em:

Treinamento (72%)

Validação (18%)

Teste (10%)

Isso garante que os modelos sejam treinados em um subconjunto, validados em outro e testados em dados completamente não vistos (se necessário depois).

3. Preparação por canal
Para cada um dos 44 canais, a pipeline executa um ciclo completo:

Extrai apenas aquele canal da imagem (como se fosse uma imagem 2D individual).

Redimensiona a imagem para 512x512 para reduzir custo computacional.

Aplica data augmentation (flip, rotação) no conjunto de treinamento para melhorar a generalização.

Cria um Dataset e DataLoader específico para aquele canal.

4. Criação do modelo U-Net
Um novo modelo U-Net 2D com resnet18 como codificador é instanciado para cada canal.

O modelo aceita imagens com 1 canal de entrada e produz máscaras binárias.

O encoder não usa pesos pré-treinados para evitar viés de domínios RGB.

5. Treinamento
O modelo é treinado por algumas épocas (ex: 5) com BCEWithLogitsLoss.

A cada época, ele é avaliado no conjunto de validação usando a métrica Dice Score, que mede a sobreposição entre a máscara predita e a real.

6. Avaliação
Após o treinamento de cada canal, o Dice Score médio é calculado para a validação.

Esse valor é registrado para comparação com os demais canais.

7. Comparação final
Ao final da execução, o script imprime um ranking com os 44 canais, ordenando do melhor Dice Score para o pior.

Isso revela quais canais carregam mais informação relevante para segmentar a região de interesse.