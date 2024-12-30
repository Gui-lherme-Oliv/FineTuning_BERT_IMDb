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

Por que essa configuração é importante?
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

Essa etapa é necessária pois durante o treinamento, o Trainer avalia periodicamente o modelo no conjunto de validação. Ter uma função personalizada como essa permite calcular métricas adicionais além da acurácia padrão, como precisão, recall e F1-Score, fornecendo uma visão mais completa do desempenho do modelo na tarefa de classificação de sentimentos.

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

Diretórios de resultados e logs:
- output_dir="./results": Define onde os resultados e checkpoints do modelo treinado serão salvos.
- logging_dir="./logs": Especifica o local para armazenar logs do treinamento, úteis para monitoramento e depuração.

Estratégias de avaliação e salvamento:
- evaluation_strategy="epoch": Realiza avaliação no conjunto de validação ao final de cada época, permitindo acompanhar o desempenho do modelo durante o treinamento.
- save_strategy="epoch": Salva os checkpoints do modelo ao final de cada época, garantindo que seja possível retomar o treinamento ou usar o melhor modelo salvo.

Hiperparâmetros de treinamento:
- learning_rate=2e-5: Define a taxa de aprendizado, que controla a velocidade de ajuste dos pesos do modelo. Este valor é tipicamente pequeno para modelos pré-treinados como o BERT, prevenindo oscilações no ajuste fino.
- weight_decay=0.01: Aplica regularização L2 para evitar overfitting, penalizando pesos excessivamente grandes no modelo.
- num_train_epochs=3: Especifica o número de passagens completas pelos dados de treinamento, um valor comum para tarefas de ajuste fino.
- per_device_train_batch_size=16 e per_device_eval_batch_size=16: Determinam o número de amostras processadas por dispositivo (como uma GPU) em cada passo. Um batch menor ajuda a economizar memória, mas pode aumentar o tempo de treinamento.

Monitoramento:
- report_to="none": Desativa o envio de logs para serviços externos, como Weights & Biases (W&B), mantendo o foco no ambiente local.

Essas configurações são necessárias pois permitem controlar todos os aspectos do treinamento, desde o gerenciamento de recursos até o acompanhamento do desempenho e a garantia de salvamento dos resultados. Com essas definições, o processo de fine-tuning é eficiente, bem monitorado e facilmente replicável.

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

O Trainer automatiza e gerencia o treinamento, validação e avaliação do modelo, reduzindo significativamente a complexidade de implementação. Ele lida com aspectos como gradiente, otimização e execução em GPU, permitindo que você se concentre mais na análise dos resultados e ajustes do modelo. A integração do tokenizador e da função de métricas garante consistência nos dados e avaliações precisas.

### 2.10. Fine-Tuning (Treinamento do modelo)
```python
trainer.train()
```
Este trecho executa o treinamento do modelo configurado, iniciando o processo de fine-tuning do BERT para a tarefa de classificação de sentimentos com base nos dados IMDb.

O que acontece durante o treinamento:
- Processamento dos dados: O Trainer utiliza o conjunto de dados de treinamento (train_data) e o tokenizador para alimentar o modelo com entradas tokenizadas, máscaras de atenção e rótulos.
- Ajuste dos pesos: Durante o treinamento, o modelo passa por iterações (steps) onde:
    - Calcula-se a função de perda (normalmente Cross-Entropy para classificação).
    - Realiza-se o backpropagation, ajustando os pesos do modelo com base nos gradientes calculados.
    - Os pesos são atualizados seguindo a taxa de aprendizado especificada (learning_rate=2e-5) e outros parâmetros de otimização.
- Avaliação periódica: Ao final de cada época, o Trainer avalia o modelo usando o conjunto de validação (val_data) para calcular métricas como acurácia, precisão, recall e F1-Score. Isso permite monitorar o desempenho do modelo durante o treinamento.
- Salvamento de checkpoints: Conforme configurado em training_args, o modelo é salvo automaticamente ao final de cada época. Esses checkpoints permitem retomar o treinamento ou usar o modelo mais recente.

O treinamento é necessário para que o fine-tuning ajuste o modelo pré-treinado (BERT) para a tarefa específica de classificação de sentimentos, refinando os pesos para capturar padrões nos dados IMDb. A abstração fornecida pelo Trainer simplifica o processo, garantindo que o treinamento seja eficiente e que o desempenho do modelo seja avaliado de forma contínua.

### 2.11. Avaliação final no conjunto de teste
```python
results = trainer.evaluate(test_data)
print(results)
```
Este trecho realiza a avaliação final do modelo no conjunto de teste, verificando seu desempenho em dados que ele nunca viu durante o treinamento ou validação.

Avaliação no conjunto de teste: 
- A função trainer.evaluate(test_data) é usada para calcular as métricas de desempenho do modelo no conjunto de teste.
- O test_data contém exemplos tokenizados e rotulados, mas que não foram utilizados para ajustar os pesos do modelo ou calibrar hiperparâmetros.

Cálculo das métricas: 
- Durante a avaliação, o modelo realiza inferências sobre os dados de teste, gerando previsões.
- A função de métricas personalizada (compute_metrics) é usada para calcular as principais métricas de desempenho, como acurácia, precisão, recall e F1-score.

Exibição dos resultados:
- O dicionário results contém os valores das métricas calculadas. O comando print(results) exibe esses valores, permitindo avaliar a eficácia do modelo em prever corretamente os sentimentos das resenhas no dataset IMDb.

A avaliação no conjunto de teste fornece uma estimativa imparcial do desempenho final do modelo. Isso é crucial para entender sua capacidade de generalização em dados reais e determinar se ele está pronto para uso prático em aplicações de análise de sentimentos.

### 2.12. Salvar o modelo ajustado (Fine-Tuned)
```python
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")
```
Este trecho salva o modelo ajustado e seu tokenizador em um diretório local, tornando-os reutilizáveis sem a necessidade de repetir o processo de treinamento.

Salvar o modelo:
- O método model.save_pretrained("./fine_tuned_bert") salva o modelo ajustado (com os pesos atualizados após o fine-tuning) no diretório especificado ("./fine_tuned_bert").
- Isso inclui os pesos treinados, a arquitetura do modelo e as configurações (como o número de rótulos).

Salvar o tokenizador:
- O método tokenizer.save_pretrained("./fine_tuned_bert") salva o tokenizador usado no treinamento no mesmo diretório.
- Isso inclui o vocabulário, as regras de tokenização e quaisquer configurações específicas (como truncamento ou padding).

Por que isso é importante?
- Reutilização do modelo: O modelo ajustado pode ser carregado posteriormente para fazer previsões em novos dados sem necessidade de novo treinamento, economizando tempo e recursos computacionais.
- Compatibilidade com a Hugging Face: A combinação de save_pretrained e from_pretrained permite que o modelo e o tokenizador sejam carregados de maneira direta e consistente em qualquer ambiente, incluindo servidores de produção.
- Portabilidade: Salvar o modelo localmente facilita seu compartilhamento com outros desenvolvedores ou sua integração em pipelines de NLP.

### 2.13. Inferência com o modelo ajustado e mapeamento de rótulos
```python
sentiment_analyzer = pipeline("sentiment-analysis", model="./fine_tuned_bert", tokenizer="./fine_tuned_bert")
label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
```
Neste trecho, o modelo ajustado é carregado para realizar inferências em novas entradas de texto, e um mapeamento de rótulos é definido para traduzir as previsões do modelo em valores mais compreensíveis.

Criação do pipeline de inferência:
- A função pipeline("sentiment-analysis", model=..., tokenizer=...) cria um pipeline pré-configurado para a tarefa de análise de sentimentos.
- O modelo e o tokenizador ajustados são carregados a partir do diretório onde foram salvos ("./fine_tuned_bert"). Isso permite que o pipeline use as configurações específicas do modelo treinado.

Mapeamento de rótulos:
- O dicionário label_map mapeia os rótulos preditos pelo modelo (por padrão, LABEL_0 e LABEL_1) para rótulos mais interpretáveis:
    - LABEL_0: Representa sentimentos negativos (NEGATIVE).
    - LABEL_1: Representa sentimentos positivos (POSITIVE).
- Esse mapeamento é importante porque os rótulos padrão (LABEL_0, LABEL_1) podem ser pouco intuitivos, especialmente para usuários finais ou em aplicações práticas.

Por que isso é importante?
- Facilidade de uso: O pipeline simplifica o processo de inferência, abstraindo detalhes como pré-processamento e pós-processamento.
- Interpretação: O mapeamento de rótulos torna as saídas do modelo mais claras, facilitando sua utilização em relatórios ou aplicações voltadas para o usuário final.
- Prontidão para produção: Com o pipeline configurado, o modelo está pronto para ser utilizado diretamente em aplicações para análise de sentimentos, como chatbots, sistemas de feedback ou monitoramento de mídias sociais.

### 2.14. Textos para análise
```python
texts = [
    "This movie was fantastic!",
    "I hated every minute of this film.",
    "The plot was okay, but the acting was superb.",
    "I wouldn't recommend this to anyone.",
    "It was a decent film, not too bad but not great either.",
    "Absolutely amazing! A masterpiece.",
    "Terrible, just terrible. A waste of time.",
    "The visuals were stunning, but the story lacked depth.",
    "One of the best movies I’ve ever seen!",
    "It’s not my kind of movie, but it was well-made.",
]

# Obtenção e formatação das previsões
for text, prediction in zip(texts, sentiment_analyzer(texts)):
    label = label_map[prediction["label"]]
    score = prediction["score"]
    print(f"text: {text}, label: {label}, score: {score:.4f}")
```
Este trecho realiza a inferência sobre uma lista de textos utilizando o modelo ajustado e o pipeline de análise de sentimentos, exibindo os resultados de maneira legível.

Lista de textos para análise:
- A variável texts contém exemplos de frases que representam diferentes opiniões, variando entre positivas, negativas e neutras. Esses textos serão analisados pelo modelo para determinar o sentimento predominante.

Inferência com o pipeline:
- O pipeline sentiment_analyzer processa a lista de textos, retornando uma lista de dicionários. Cada dicionário contém:
    - "label": O rótulo predito pelo modelo (LABEL_0 ou LABEL_1).
    - "score": A confiança associada à previsão, variando de 0 a 1.

Mapeamento e formatação dos resultados:
- Para cada texto, o rótulo ("label") é traduzido para uma forma compreensível usando o label_map (ex.: LABEL_0 -> NEGATIVE).
- O valor de confiança ("score") é formatado com quatro casas decimais, fornecendo uma visão detalhada da confiança do modelo em sua previsão.

Exibição dos resultados:
- Cada texto, junto com o rótulo de sentimento (positivo ou negativo) e o score, é impresso no console, no formato:
```python
text: [texto], label: [sentimento], score: [confiança]
````

Por que isso é importante?
- Análise de sentimentos em lote: Permite analisar rapidamente um conjunto de textos, tornando a solução eficiente para aplicações como monitoramento de redes sociais ou análises de reviews.
- Interpretação do modelo: Exibir o score fornece insights sobre a confiança do modelo, ajudando a avaliar a robustez das previsões.
- Prontidão para produção: Esse formato é ideal para sistemas que precisam processar grandes volumes de texto e apresentar resultados claros para usuários finais.

## 3. Resultados

</div>











