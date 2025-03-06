#### Siga as instruções do notebook python para capturar o modelo no formato <ins>SavedModel</ins> 

[Carregando modelo fine tuned de classificação para patologia](https://colab.research.google.com/github/amandafbri/ia-para-patologia/blob/main/carregando_modelo_treinado.ipynb)

#### Ou utilize essas instruções em python para capturar o modelo e salvar no formato <ins>SavedModel</ins>

```python
# Salva o modelo em formato .keras (formato unificado)
model.save("fine_tuned_[MODELO]_foundation.keras")
import tensorflow as tf
# Carrega o modelo salvo
model_finetuned = tf.keras.models.load_model("fine_tuned_[MODELO]_foundation.keras")
# Exporta o modelo para o formato SavedModel (gera uma pasta)
model_finetuned.export("fine_tuned_[MODELO]_foundation_tf")
# Compacta a pasta gerada
!zip -r fine_tuned_[MODELO]_foundation_tf.zip fine_tuned_[MODELO]_foundation_tf
# Baixa o arquivo ZIP para o seu computador
from google.colab import files
files.download('fine_tuned_[MODELO]_foundation_tf.zip')
```