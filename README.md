<div align="justify">

# Fine-Tuning com Hugging Face Transformers: Classificação de sentimentos com IMDb

Este é um um exemplo prático de fine-tuning usando o Hugging Face Transformers para ajustar um modelo pré-treinado (como BERT) em uma tarefa de classificação de sentimentos usando o dataset IMDb.

## 1. Introdução

O fine-tuning é uma técnica amplamente utilizada no campo de Processamento de Linguagem Natural (PLN) para adaptar modelos pré-treinados a tarefas específicas. Nesta aplicação, utilizamos a biblioteca Hugging Face Transformers para ajustar o modelo BERT (Bidirectional Encoder Representations from Transformers) em uma tarefa de classificação de sentimentos com o dataset IMDb, um conhecido conjunto de dados contendo avaliações de filmes rotuladas como positivas ou negativas.

### 1.1 O que é Hugging Face?

Hugging Face é uma plataforma e biblioteca de código aberto que fornece ferramentas e modelos para tarefas de PLN. A biblioteca transformers da Hugging Face é especialmente conhecida por disponibilizar uma ampla gama de modelos pré-treinados para tarefas como classificação de textos, resposta a perguntas, resumo de textos e muito mais. Esses modelos pré-treinados podem ser adaptados a diferentes domínios ou aplicações por meio de fine-tuning

### 1.2. O que é o BERT?

O BERT, desenvolvido pelo Google, é um dos modelos mais influentes no campo de PLN. Ele foi projetado para entender o contexto bidirecional de uma frase, o que significa que ele analisa tanto o lado esquerdo quanto o direito de cada palavra no texto simultaneamente. Graças ao pré-treinamento em grandes quantidades de texto, o BERT é capaz de capturar nuances linguísticas e semânticas, tornando-o ideal para uma variedade de tarefas de linguagem.

### 1.3. Por que Fine-Tuning?

Modelos como o BERT são pré-treinados em grandes quantidades de dados genéricos, mas não estão otimizados para tarefas específicas. O fine-tuning ajusta os parâmetros do modelo para que ele performe bem em um conjunto de dados específico, como o IMDb. Isso evita a necessidade de treinar um modelo do zero, economizando tempo e recursos computacionais.

### 1.4. O Dataset IMDb

O IMDb (Internet Movie Database) é um dos conjuntos de dados mais populares para classificação de sentimentos. Ele contém 50.000 avaliações de filmes em inglês, divididas igualmente entre rótulos positivos e negativos. Este dataset é amplamente usado como benchmark para modelos de PLN em tarefas de análise de sentimentos.

### 1.5. Benefícios e Aplicações

O fine-tuning é essencial para adaptar modelos pré-treinados a cenários específicos. No caso da classificação de sentimentos, ele permite que empresas analisem opiniões de clientes, melhorem produtos com base no feedback ou até detectem comentários tóxicos em redes sociais.

Com a combinação da biblioteca Hugging Face e o poder do BERT, é possível resolver problemas complexos de PLN de forma eficiente e acessível.

## 2. Fluxo de trabalho
### 2.1. Instalar bibliotecas necessárias


```python
# Instalar bibliotecas necessárias
!pip install transformers datasets torch scikit-learn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```



</div>

