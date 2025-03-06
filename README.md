# Demonstração dos Modelos Health AI Developer Foundations (HAI-DEF)

Este repositório contém demonstrações de classificadores de imagens clínicas utilizando os modelos de **Path Foundation**, **Derm Foundation** e **CXR Foundation**. A aplicação foi construída com **Streamlit** para fornecer uma interface visual para carregar e classificar imagens de amostras histopatológicas.

## Funcionalidades

- escolher o modelo que desejado
- Carregar uma imagem clínica em **PNG** ou **JPEG**.
- Classificar a imagem com base em um modelo treinado.
  
## Como Usar

### Requisitos

1. **Python 3.x**: Certifique-se de ter o Python instalado em sua máquina.
2. **Ambiente Virtual**: É recomendável criar um ambiente virtual para gerenciar as dependências. Para isso, siga os passos abaixo:

   #### Linux/Mac:
   ```bash
   python3 -m venv <nome_do_ambiente>
   source <nome_do_ambiente>/bin/activate
   ```

    #### Windows:
    ```bash
    python -m venv <nome_do_ambiente>
    source <nome_do_ambiente>\Scripts\activate
    ```
3. Instalar dependências: Após ativar o ambiente virtual, instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

## Onde Baixar os Modelos

Para acessar os modelos HAI-DEF com e sem fine tuning já convertidos no format SavedModel, acesse o drive: 
> https://drive.google.com/drive/u/0/folders/1RYolROgnYPcVdLDcAR97JO6EVG_L2Y2V

Baixe os modelos para sua máquina local e adicione-os à pasta **saved_models**

<ins>NÃO FAÇA UPLOAD DOS MODELOS AO GITHUB.</ins> Apenas deixe-os em sua máquina local

Se desejar fazer outro fine tuning, modificar proporção de treino e teste, ou até mesmo adicionar outros modelos, siga a próxima instrução:

## Instruções para Conversão do Modelo

O modelo treinado deve ser convertido para o formato SavedModel para ser carregado de maneira portátil. Para isso, siga as instruções no arquivo convert_model_instruction. O link do notebook que descreve o passo a passo para a conversão do modelo está incluído nesse arquivo.

## Executando a Aplicação

Após configurar o ambiente virtual e instalar as dependências, execute o seguinte comando para rodar a aplicação:

```bash
streamlit run app.py
```
Isso abrirá a interface do Streamlit no seu navegador, onde você poderá navegar entre as soluções e carregar imagens para visualizar as classificações

## Tecnologias Utilizadas

- Streamlit: Para criar a interface interativa e visual.
- TensorFlow: Para o modelo de classificação de imagens histopatológicas.
- HAI-DEF: Conjunto de modelos para análise de dados médicos.
- Path Foundation: Modelo para análise de imagens histopatológicas.
- Derm Foundation: Modelo para análise de imagens dermatológicas.
- CXR Foundation: Modelo para análise de imagens de raio X de peito.

## Futuras modifiações

- Fazer Deploy dos modelos e configurar chamadas de API para sua utilização