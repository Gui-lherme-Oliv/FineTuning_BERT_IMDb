<div align="justify">

# Fine-Tuning com Hugging Face Transformers: Classificação de sentimentos com IMDb

Este é um um exemplo prático de fine-tuning usando o Hugging Face Transformers para ajustar um modelo pré-treinado (como BERT) em uma tarefa de classificação de sentimentos usando o dataset IMDb.

## 1. Introdução

O fine-tuning é uma técnica amplamente utilizada no campo de Processamento de Linguagem Natural (PLN) para adaptar modelos pré-treinados a tarefas específicas. Nesta aplicação, foi utilizada a biblioteca Hugging Face Transformers para ajustar o modelo BERT (Bidirectional Encoder Representations from Transformers) em uma tarefa de classificação de sentimentos com o dataset IMDb, um conhecido conjunto de dados contendo avaliações de filmes rotuladas como positivas ou negativas.

### 1.1 O que é Hugging Face?

Hugging Face é uma plataforma e biblioteca de código aberto que fornece ferramentas e modelos para tarefas de PLN. A biblioteca transformers da Hugging Face é especialmente conhecida por disponibilizar uma ampla gama de modelos pré-treinados para tarefas como classificação de textos, resposta a perguntas, resumo de textos e muito mais. Esses modelos pré-treinados podem ser adaptados a diferentes domínios ou aplicações por meio de fine-tuning

### 1.2. O que é o BERT?

O BERT, desenvolvido pelo Google, é um dos modelos mais influentes no campo de PLN. Ele foi projetado para entender o contexto bidirecional de uma frase, o que significa que ele analisa tanto o lado esquerdo quanto o direito de cada palavra no texto simultaneamente. Graças ao pré-treinamento em grandes quantidades de texto, o BERT é capaz de capturar nuances linguísticas e semânticas, tornando-o ideal para uma variedade de tarefas de linguagem.

### 1.3. Por que Fine-Tuning?

Modelos como o BERT são pré-treinados em grandes quantidades de dados genéricos, mas não estão otimizados para tarefas específicas. O fine-tuning ajusta os parâmetros do modelo para que ele performe bem em um conjunto de dados específico, como o IMDb. Isso evita a necessidade de treinar um modelo do zero, economizando tempo e recursos computacionais.

### 1.4. O dataset IMDb

O IMDb (Internet Movie Database) é um dos conjuntos de dados mais populares para classificação de sentimentos. Ele contém 50.000 avaliações de filmes em inglês, divididas igualmente entre rótulos positivos e negativos. Este dataset é amplamente usado como benchmark (padrão de referência ou ponto de comparação) para modelos de PLN em tarefas de análise de sentimentos.

### 1.5. Benefícios e Aplicações

O fine-tuning é essencial para adaptar modelos pré-treinados a cenários específicos. No caso da classificação de sentimentos, ele permite que empresas analisem opiniões de clientes, melhorem produtos com base no feedback ou até detectem comentários tóxicos em redes sociais.

Com a combinação da biblioteca Hugging Face e o poder do BERT, é possível resolver problemas complexos de PLN de forma eficiente e acessível.

## 2. Fluxo de trabalho
### 2.1. Instalar bibliotecas necessárias
```python
!pip install transformers datasets torch scikit-learn

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```
Neste trecho de código, o objetivo principal é preparar o ambiente e importar as bibliotecas necessárias para o fine-tuning de um modelo de linguagem pré-treinado, como o BERT, para a tarefa de classificação de sentimentos no conjunto de dados IMDb.

Primeiramente, é realizada a instalação das bibliotecas essenciais, como transformers, que fornece ferramentas para trabalhar com modelos da Hugging Face, datasets, que facilita o carregamento e manipulação de datasets, torch, a biblioteca central para computação em redes neurais, e scikit-learn, que é útil para métricas de avaliação como precisão e recall.

Em seguida, o código importa as classes e funções específicas que serão utilizadas no processo. AutoTokenizer e AutoModelForSequenceClassification são ferramentas da Hugging Face para carregar automaticamente o tokenizador e o modelo pré-treinado adequados para a tarefa de classificação de sequências de texto. A classe Trainer e TrainingArguments são usadas para definir como o modelo será treinado e para gerenciar parâmetros como taxa de aprendizado e número de épocas. A função pipeline é útil para criar um pipeline de inferência fácil de usar. A importação de load_dataset permite carregar o dataset IMDb diretamente, e as métricas accuracy_score e precision_recall_fscore_support da sklearn serão usadas para avaliar o desempenho do modelo treinado.

Esses passos são essenciais para garantir que o código tenha todas as ferramentas necessárias para carregar os dados, treinar o modelo e avaliar seu desempenho de forma eficaz.

### 2.2. Carregamento e Divisão do dataset
```python
# Carregamento do dataset IMDb
dataset = load_dataset("imdb")

# Divisão em treinamento, validação e teste
train_data = dataset["train"].train_test_split(test_size=0.1)["train"]
val_data = dataset["train"].train_test_split(test_size=0.1)["test"]
test_data = dataset["test"]
```
Neste trecho, o foco está no carregamento do dataset IMDb e na preparação dos dados para o processo de treinamento, validação e teste do modelo de classificação de sentimentos.

Carregamento do dataset: A função load_dataset("imdb") é usada para baixar e carregar automaticamente o conjunto de dados IMDb, que é amplamente utilizado em tarefas de análise de sentimentos. Ele contém resenhas de filmes rotuladas como positivas ou negativas, tornando-o adequado para tarefas de classificação binária.

Divisão dos dados: 
- O conjunto de dados IMDb já vem dividido em partes de treinamento (train) e teste (test), mas aqui o conjunto de treinamento é adicionalmente particionado em uma fração para validação.
- A função train_test_split é utilizada no conjunto de treinamento original para criar duas subdivisões: uma para treinamento final (train_data) e outra para validação (val_data). Esse particionamento é útil para ajustar os hiperparâmetros e avaliar o modelo durante o treinamento.
- Por fim, o conjunto de teste (test_data) é deixado intacto para ser usado na avaliação final do desempenho do modelo.

Essa separação dos dados em diferentes conjuntos é crucial para garantir que o modelo seja treinado de forma eficaz e que seu desempenho seja avaliado corretamente em dados que ele não viu durante o treinamento.

### 2.3. Configuração do tokenizador e modelo
```python
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
Neste trecho, o foco é configurar o tokenizador e especificar o modelo pré-treinado que será ajustado para a tarefa de classificação de sentimentos.

Definição do modelo pré-treinado: A variável model_name é definida como "bert-base-uncased", indicando que será usado o modelo BERT em sua versão base, com 12 camadas e sem diferenciação de maiúsculas e minúsculas nos textos (uncased). Essa escolha é apropriada para muitas tarefas de NLP, como análise de sentimentos, por sua robustez e eficácia em entender o contexto das palavras.

Configuração do tokenizador: O AutoTokenizer.from_pretrained(model_name) carrega automaticamente o tokenizador pré-treinado correspondente ao modelo especificado. O tokenizador é responsável por transformar o texto em uma sequência de tokens numéricos que o modelo pode processar. No caso do BERT, isso inclui:
- Dividir o texto em subpalavras ou palavras inteiras.
- Adicionar tokens especiais, como [CLS] no início e [SEP] entre ou no final das sequências.
- Criar máscaras de atenção que indicam quais tokens devem ser considerados no processamento.

Essa etapa é essencial porque o tokenizador garante que o formato dos dados de entrada esteja alinhado com o modelo pré-treinado, mantendo a consistência necessária para o fine-tuning.

### 2.4. Tokenização dos textos
```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_data = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_data = val_data.map(tokenize_function, batched=True, remove_columns=["text"])
test_data = test_data.map(tokenize_function, batched=True, remove_columns=["text"])
```
Neste trecho, a principal tarefa é transformar os textos brutos do dataset em uma representação numérica que o modelo pode processar, utilizando o tokenizador configurado anteriormente. Além disso, os dados são preparados para treinamento, validação e teste.

Definição da função de tokenização:
- A função tokenize_function aplica o tokenizador ao texto de entrada.
- O parâmetro truncation=True garante que textos mais longos sejam truncados para não exceder o tamanho máximo especificado (neste caso, 128 tokens).
- O parâmetro padding="max_length" ajusta todas as sequências ao mesmo comprimento (128 tokens), adicionando padding (preenchimento) se necessário. Isso é importante para processamento em batch (em paralelo) no modelo.
- O argumento max_length=128 define explicitamente o comprimento máximo das sequências de entrada, equilibrando a quantidade de informações e a eficiência computacional.

Mapeamento da função nos dados:
- A função map aplica a tokenize_function a todas as amostras do conjunto de dados (train_data, val_data e test_data) de forma eficiente e em lotes (batched=True), otimizando o desempenho.
- O parâmetro remove_columns=["text"] descarta a coluna original de texto após a tokenização, pois o modelo trabalha apenas com a entrada tokenizada.

Esse processo garante que cada texto seja transformado em uma sequência numérica compatível com o modelo BERT. A tokenização também padroniza o comprimento das sequências, o que facilita o processamento em lotes durante o treinamento e a avaliação.

### 2.5. Preparação dos dados para o Trainer
```python
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
```
Neste trecho, o foco é preparar os dados tokenizados para serem utilizados no treinamento e na avaliação do modelo com a classe Trainer.

Configuração do formato de dados:
- O método set_format converte os dados para o formato PyTorch (type="torch"), que é necessário para treinar o modelo utilizando o framework PyTorch.
- As colunas especificadas ("input_ids", "attention_mask" e "label") são as únicas mantidas no dataset para o treinamento. Essas colunas representam:
    - input_ids: Sequências tokenizadas do texto.
    - attention_mask: Máscara que indica quais tokens devem ser considerados (1) e quais são padding (0).
    - label: Rótulo associado a cada exemplo (positivo ou negativo no caso da classificação de sentimentos).

Por que isso é necessário?
- O Trainer da biblioteca transformers espera que os dados estejam em formato PyTorch e contenham essas colunas específicas. Essa configuração garante que as entradas do modelo e os rótulos sejam passados corretamente durante o treinamento e a avaliação.

Ao final dessa etapa, os dados estão no formato adequado para serem processados diretamente pelo modelo durante o treinamento e a validação, mantendo a eficiência e a compatibilidade.

### 2.6. Configuração do modelo
```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```
Neste trecho, o objetivo é configurar o modelo pré-treinado para a tarefa específica de classificação de sentimentos.

Carregamento do modelo pré-treinado:
- O método AutoModelForSequenceClassification.from_pretrained é utilizado para carregar o modelo BERT pré-treinado (bert-base-uncased), configurando-o para a tarefa de classificação de sequência.
- O modelo é ajustado para o número de rótulos da tarefa com o parâmetro num_labels=2, já que o dataset IMDb trata de uma classificação binária (sentimentos positivos ou negativos).

Por que essa configuração é importante?:
- A arquitetura BERT original é genérica e pode ser usada para várias tarefas de NLP, mas para tarefas específicas como classificação de sentimentos, uma camada adicional de classificação (normalmente uma camada linear) é adicionada ao final. Essa camada mapeia as representações internas do modelo para as probabilidades dos rótulos de saída.
- O uso de num_labels=2 informa ao modelo que ele deve prever duas classes distintas, ajustando a saída da camada de classificação para atender a esse requisito.

Com essa configuração, o modelo pré-treinado está pronto para ser ajustado no dataset IMDb, aprendendo os padrões específicos da tarefa de análise de sentimentos.

### 2.7. Função personalizada para calcular métricas
```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
```
Neste trecho, é criada uma função personalizada para calcular as métricas de avaliação do modelo durante o treinamento e a validação.

Entrada e extração das previsões e rótulos:
- A função recebe um objeto pred, que contém as previsões do modelo (pred.predictions) e os rótulos reais (pred.label_ids).
- As previsões são processadas com argmax(-1) para selecionar o rótulo mais provável (o índice com maior valor de probabilidade em cada amostra).

Cálculo de métricas de desempenho:
- Precisão (Precision): Mede a proporção de previsões positivas corretas em relação ao total de previsões positivas feitas pelo modelo.
- Recall: Mede a proporção de verdadeiros positivos capturados em relação ao total de positivos reais no conjunto de dados.
- F1-Score: Calcula a média harmônica entre precisão e recall, oferecendo uma visão balanceada do desempenho, especialmente útil quando as classes estão desbalanceadas.
- Acurácia (Accuracy): Mede a proporção de previsões corretas (positivas e negativas) em relação ao total de amostras.

Retorno das métricas: A função retorna as métricas como um dicionário, facilitando sua integração no processo de treinamento com o Trainer.

Por que isso é necessário?
Durante o treinamento, o Trainer avalia periodicamente o modelo no conjunto de validação. Ter uma função personalizada como essa permite calcular métricas adicionais além da acurácia padrão, como precisão, recall e F1-Score, fornecendo uma visão mais completa do desempenho do modelo na tarefa de classificação de sentimentos.

### 2.8. Configuração do treinamento
```python
training_args = TrainingArguments(
    output_dir="./results",          # Pasta para salvar resultados
    evaluation_strategy="epoch",    # Avaliação a cada época
    learning_rate=2e-5,             # Taxa de aprendizado
    per_device_train_batch_size=16, # Batch size para treino
    per_device_eval_batch_size=16,  # Batch size para validação
    num_train_epochs=3,             # Número de épocas
    weight_decay=0.01,              # Regularização L2 (weight decay)
    save_strategy="epoch",          # Salvar modelo a cada época
    logging_dir="./logs",           # Diretório para logs
    report_to="none",               # Desativa serviços externos (W&B e outros serviços de monitoramento)
)
```
Este trecho define os argumentos necessários para configurar o processo de treinamento do modelo usando a classe TrainingArguments da biblioteca Hugging Face. Esses parâmetros controlam como o treinamento será executado, otimizando o desempenho e a eficiência.

Configurações Importantes:
Diretórios de Resultados e Logs:
- output_dir="./results": Define onde os resultados e checkpoints do modelo treinado serão salvos.
- logging_dir="./logs": Especifica o local para armazenar logs do treinamento, úteis para monitoramento e depuração.

Estratégias de Avaliação e Salvamento:
- evaluation_strategy="epoch": Realiza avaliação no conjunto de validação ao final de cada época, permitindo acompanhar o desempenho do modelo durante o treinamento.
- save_strategy="epoch": Salva os checkpoints do modelo ao final de cada época, garantindo que você possa retomar o treinamento ou usar o melhor modelo salvo.

Hiperparâmetros de Treinamento:
- learning_rate=2e-5: Define a taxa de aprendizado, que controla a velocidade de ajuste dos pesos do modelo. Este valor é tipicamente pequeno para modelos pré-treinados como o BERT, prevenindo oscilações no ajuste fino.
- weight_decay=0.01: Aplica regularização L2 para evitar overfitting, penalizando pesos excessivamente grandes no modelo.
- num_train_epochs=3: Especifica o número de passagens completas pelos dados de treinamento, um valor comum para tarefas de ajuste fino.
- per_device_train_batch_size=16 e per_device_eval_batch_size=16: Determinam o número de amostras processadas por dispositivo (como uma GPU) em cada passo. Um batch menor ajuda a economizar memória, mas pode aumentar o tempo de treinamento.

Monitoramento:
- report_to="none": Desativa o envio de logs para serviços externos, como Weights & Biases (W&B), mantendo o foco no ambiente local.

Por que isso é necessário?
Essas configurações permitem controlar todos os aspectos do treinamento, desde o gerenciamento de recursos até o acompanhamento do desempenho e a garantia de salvamento dos resultados. Com essas definições, o processo de fine-tuning é eficiente, bem monitorado e facilmente replicável.

### 2.9. Inicialização do Trainer
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Passa as métricas personalizadas
)
```
Aqui, o código inicializa o Trainer, uma classe da biblioteca Hugging Face que simplifica o processo de treinamento e avaliação de modelos de linguagem. O Trainer combina todos os componentes previamente configurados para realizar o ajuste fino do modelo.

Componentes do Trainer:
- model=model: Refere-se ao modelo pré-treinado configurado anteriormente (BERT com uma camada de classificação). O Trainer usará esse modelo para treinar e realizar inferências.
- args=training_args: Passa os argumentos de treinamento definidos na etapa anterior, como taxa de aprendizado, número de épocas, estratégias de avaliação e diretórios para resultados.
- train_dataset=train_data e eval_dataset=val_data: Especificam os conjuntos de dados tokenizados para treinamento e validação. O Trainer usará esses datasets para calcular as perdas, ajustar os pesos do modelo e avaliar o desempenho ao final de cada época.
- tokenizer=tokenizer: Fornece o tokenizador utilizado para preparar os dados. Isso é necessário para garantir que as inferências realizadas durante a validação ou avaliação sigam o mesmo processo de tokenização usado durante o treinamento.
- compute_metrics=compute_metrics: Passa a função personalizada para cálculo de métricas, como precisão, recall, F1-score e acurácia. Essas métricas serão calculadas durante a validação, fornecendo uma visão detalhada do desempenho do modelo.

Por que isso é importante?
O Trainer automatiza e gerencia o treinamento, validação e avaliação do modelo, reduzindo significativamente a complexidade de implementação. Ele lida com aspectos como gradiente, otimização e execução em GPU, permitindo que você se concentre mais na análise dos resultados e ajustes do modelo. A integração do tokenizador e da função de métricas garante consistência nos dados e avaliações precisas.

### 2.10.



</div>

