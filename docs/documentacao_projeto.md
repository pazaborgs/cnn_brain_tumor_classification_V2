# Documentação Técnica e Decisões de Projeto

Este documento consolida o guia de arquitetura de código, explicações detalhadas dos scripts que compõem o pipeline MLOps e o registro técnico das decisões de treinamento e fine-tuning adotadas para a Classificação de Tumores Cerebrais em MRI (Versão 2).

---

## 1. Arquitetura do Projeto e Scripts

A refatoração modular dividiu o projeto em componentes especialistas, organizados em pacotes e orquestrados por uma CLI (`main.py`) e uma interface Web (`app.py`).

### 1.1. Orquestração e Entrypoints

#### `main.py`
- **O que faz:** É o orquestrador (CLI) principal da aplicação. Serve como um ponto único para engatilhar as diferentes fases do ciclo de vida MLOps.
- **Entradas:** Argumento via linha de comando `--mode` (`train`, `evaluate`, `upload_hf`, `app`).
- **Saídas:** Inicia sub-rotinas chamando os scripts apropriados (ex: dispara o treinamento, abre o servidor local Streamlit ou faz push pro Hugging Face).
- **Erros Comuns:** 
  - `ValueError`: Caso seja passado um `--mode` não reconhecido (deve ser restrito aos 4 modos suportados).

#### `app.py`
- **O que faz:** Ponto de entrada do dashboard web. Inicializa o Streamlit, configura a estrutura da barra lateral (Sidebar) e orquestra a injeção do CSS e roteamento para as páginas (`diagnostic.py`, `mlops.py`, `comparison.py`).
- **Entradas:** Nenhuma direta (operado via UI). Depende dos binários dos modelos e snapshots (`mlflow_runs_snapshot.csv`).
- **Saídas:** Um servidor web local hospedando o painel de inferência médica e análise MLOps.
- **Erros Comuns:**
  - `FileNotFoundError`: Se tentado rodar sem o `config.yaml` ou se o arquivo `styles.css` foi deletado acidentalmente.

### 1.2. Módulo Core (`src/`)

#### `src/ingest.py`
- **O que faz:** Automatiza a extração (download) do banco de dados oficial diretamente dos servidores do Kaggle e descompacta as ressonâncias magnéticas localmente. Evita re-downloads se a pasta já existir.
- **Entradas:** Chaves da API do Kaggle (vindas do `.env` ou das variáveis de ambiente de deploy).
- **Saídas:** Popula a pasta `data/raw/` com 4 diretórios (`glioma_tumor`, `no_tumor`, etc.) repletos de arquivos `.jpg`.
- **Erros Comuns:**
  - `KaggleApiError`: Disparado quando as credenciais em `KAGGLE_API_TOKEN` são inválidas ou estão ausentes.

#### `src/preprocess.py`
- **O que faz:** Constrói os Datasets de altíssima performance utilizando `tf.data`. Ele particiona os dados (Treino/Val/Teste), aplica redimensionamento (Resize) em tempo real, normaliza os valores dos pixels, e faz o cache/prefetch para que a GPU nunca fique ociosa esperando os dados do HD.
- **Entradas:** Caminho da pasta `data/raw/`, tamanho do batch e dimensão-alvo da imagem (`image_size`).
- **Saídas:** Três objetos `tf.data.Dataset` puros (treino, validação e teste) já batcheados e precarregados em buffer.
- **Erros Comuns:**
  - Falha ao encontrar imagens caso o script de `ingest.py` não tenha sido executado, retornando um Dataset vazio com 0 batches.

#### `src/models.py`
- **O que faz:** Define a arquitetura matemática das redes neurais. Possui funções para construir o modelo autoral (CNN Custom com blocos VGG/ResNet-style) ou importar arquiteturas densas do `tf.keras.applications` transferindo os pesos base do ImageNet.
- **Entradas:** Dicionários de hiperparâmetros (como `learning_rate`, dimensão de entrada e estratégia de fine-tuning / descongelamento).
- **Saídas:** Objetos `tf.keras.Model` prontos e "compilados" (com otimizador e funções de perda anexados).
- **Erros Comuns:**
  - Erro de shape de tensor (`ValueError: Dimensions must be equal`), comum caso o tamanho da imagem fornecida não case com as expectativas (ex: InceptionV3 exige no mínimo 75x75).

#### `src/train_model.py`
- **O que faz:** O script mais pesado do sistema. Pega os Datasets do pré-processamento, instancia os modelos de `models.py`, e realiza as épocas de treinamento (*Gradient Descent*). Utiliza técnicas avançadas como *Early Stopping* para prevenir sobretreino e documenta a evolução da perda via MLflow.
- **Entradas:** `config.yaml` carregado contendo as regras de hiperparâmetros (Paciência, fator de decaimento de LR).
- **Saídas:** Modelos treinados salvos na pasta `artifacts/models/` no formato moderno e comprimido `.keras`.
- **Erros Comuns:**
  - **OOM (Out of Memory)**: Pode crachar a placa de vídeo ou RAM do sistema se o `batch_size` especificado for grande demais (como 64 ou 128 em GPUs modestas).

#### `src/evaluate_model.py`
- **O que faz:** Atua como um "auditor cego". Pega o modelo já finalizado e aplica o conjunto de imagens que ele nunca viu (Teste). Avalia quão bem a máquina generaliza os diagnósticos para o mundo real. Gera gráficos de acurácia x perda e plotagens estatísticas.
- **Entradas:** Caminho do modelo `.keras` existente e conjunto de validação `tf.data`.
- **Saídas:** Tabelas métricas de precisão (F1, Recall) injetadas no MLflow e PNGs da **Matriz de Confusão** injetados na pasta `artifacts/figures/`.
- **Erros Comuns:**
  - `FileNotFoundError` ao tentar avaliar um modelo que sequer terminou de ser treinado.

#### `src/gradcam.py`
- **O que faz:** Motor XAI (Inteligência Artificial Explicável). Através do TensorFlow GradientTape, ele observa retrospectivamente os gradientes que chegaram na última camada de convolução do modelo para entender "quais pixels pesaram mais" na escolha final, construindo um mapa de calor termal.
- **Entradas:** Modelo carregado e uma imagem crua MRI.
- **Saídas:** Uma imagem 2D com as classes ativacionais em tons quentes onde o modelo deu atenção máxima e azul escuro onde ele ignorou.
- **Erros Comuns:**
  - Falha ao identificar o nome da última camada de convolução. Alguns modelos customizados não nomeiam claramente suas saídas (`conv2d_last`), forçando o script a buscar cegamente.

### 1.3. Utilitários MLOps (`src/utils/`)

#### `load_config.py`
- **O que faz:** Utilitário trivial que abre o `config.yaml` e transforma seus metadados num dicionário nativo em Python para o sistema consumir livremente em RAM.
- **Erros Comuns:** `YAMLError` caso falte espaço, aspas ou exista identação errada no arquivo `config.yaml`.

#### `export_mlflow_snapshot.py`
- **O que faz:** Ponte de dados entre o Banco de Dados pesado local e o Dashboard Web hospedado na Nuvem. Busca via API local do MLflow todas as execuções recentes de cada modelo e transcreve suas 14 métricas críticas.
- **Entradas:** Registro SQL/Lite Local em `mlruns/`.
- **Saídas:** O artefato leve `artifacts/metrics/mlflow_runs_snapshot.csv`, que é consumido instantaneamente pelo Streamlit sem sobrecarga.

#### `upload_models_hf.py`
- **O que faz:** Conecta-se aos clusters do repositório remoto **Hugging Face Hub** utilizando a biblioteca nativa `huggingface_hub`. Varre a pasta `artifacts/models`, valida e empurra as atualizações na nuvem para download via Git LFS.
- **Entradas:** O token secreto (Write Access) definido no seu `.env` local (`HF_TOKEN`).
- **Saídas:** Arquivos commitados e visíveis na nuvem no hub central do seu pipeline.
- **Erros Comuns:** Falhas HTTP 401 caso o Token tenha sido revogado ou não possua permissão de gravação (Write).

### 1.4. Frontend Web (`src/ui/`)

#### `tabs/diagnostic.py` e `tabs/mlops.py` (Painel Médico e Analytics)
- **O que fazem:** Sub-páginas encapsuladas. O _diagnostic_ foca em gerar inferências ao vivo nas amostras radiológicas (simulando um ambiente real), enquanto o _mlops_ renderiza os Leaderboards usando a tabela que o snapshot exportou e as imagens de matriz de confusão e arquitetura.
- **Entradas:** Inputs dinâmicos do usuário (botões de clique, caixas de seleção, rádio buttons).
- **Saídas:** Interface renderizada nativamente usando Streamlit UI Components e gráficos analíticos em Altair.
- **Erros Comuns:** Se o modelo estiver rodando na Nuvem e tentar carregar um modelo não publicado, ele exibirá um aviso instruindo o download transparente direto da Hugging Face API.

---

## 2. Decisões de Treinamento, Hiperparâmetros e Fine-Tuning

Registro técnico das decisões aplicadas aos modelos de aprendizado profundo durante os ensaios no banco `sartajbhuvaji/brain-tumor-classification-mri`.

### 2.1. Benchmark Inicial (20 Épocas)

Resultados da execução baseline inicial, onde os extratores de características do *Transfer Learning* ficaram completamente estáticos (pesos do ImageNet bloqueados):

| Arquitetura | Resolução | Acurácia Val. | Perda Val. | Acurácia Teste | F1-Score Macro | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `cnn_custom` | 128x128 | 29.27% | 2.7504 | 18.78% | 0.0791 | Overfitting (Parada na época 8) |
| `inception_v3` | 299x299 | 85.19% | 0.3923 | 66.75% | 0.6531 | Estável na Validação, Queda no Teste |
| `efficientnet_b0` | 224x224 | 80.14% | 0.5913 | 67.77% | 0.6473 | Estável na Validação, Queda no Teste |

### 2.2. Análise Técnica dos Gargalos Identificados no Baseline

1. **Extratores Congelados (`base.trainable = False`):**
   A acurácia de validação do InceptionV3 atingiu ~85%, mas a de teste desabou para ~66%. Isso ocorre porque os pesos pré-treinados em ImageNet não estão extraindo texturas biológicas; necessitam de descongelamento parcial para adaptar os filtros às especificidades biomédicas.
2. **Parada Precoce Agressiva (`patience: 3`):**
   A paciência de 3 épocas interrompeu o treinamento do modelo autoral muito antes da convergência devido a oscilações normais nas rodadas do val_loss.
3. **Taxa de Aprendizado Elevada (`lr = 0.001`):**
   A taxa inicial padronizada gerou variações agressivas e destruiu pesos importantes durante o processo de gradient descent.
4. **Arquitetura Rasa no Modelo Customizado:**
   O modelo convolucional customizado original (V1) possuía apenas uma convolução simples por bloco, induzindo um forte *overfitting* imediato e estagnação crônica em acurácia.

### 2.3. Modificações Aplicadas (Fase II Final)

1. **Reformulação da Arquitetura `cnn_custom` (Blocos Duplos estilo VGG/ResNet):**
   - Inserimos blocos convolucionais duplos profundos (`Conv2D` → `BatchNorm` → `ReLU` → `Conv2D`).
   - Adicionamos a estratégia de inicialização **He Normal**, projetada especialmente para ativar pesos favoráveis à função ReLU.
   - Incrementamos um **Dropout Espacial Progressivo** (`0.15` → `0.25` → `0.35` → `0.40`) ao avançar das camadas de extração brutas para as representações densas.
2. **Descongelamento Seletivo das Camadas (Fine-Tuning):**
   Para transferir o conhecimento sem causar destruição catastrófica:
   - `inception_v3`: Descongelamento das últimas **40** camadas.
   - `efficientnet_b0`: Descongelamento das últimas **30** camadas.
   - `densenet121`: Inclusão na suíte de testes por sua habilidade inata de reutilização visual via block concatenation.
3. **Otimização de Pipeline I/O:**
   Injetamos `prefetch(tf.data.AUTOTUNE)` durante os `image_dataset_from_directory`, fazendo com que a CPU leia as imagens na RAM de fundo simultaneamente ao ciclo em que a GPU as compila.
4. **Hospedagem e Registry Serverless (Hugging Face Hub):**
   Com o modelo estabilizado, optamos por um deployment Serverless no Streamlit hospedando todos os `.keras` binários gigantes via Git LFS e inferindo-os via download estático por demanda.

### 2.4. Segunda Leva de Runs (Expansão Multi-Modelo)

Após a estabilização do pipeline e a aplicação da malha de hiperparâmetros final, uma **segunda leva de execuções** foi conduzida. O objetivo principal foi introduzir a **DenseNet121** ao conjunto de inferência. A DenseNet foi incorporada devido à sua altíssima eficiência paramétrica e à capacidade inata de reutilização de características (feature reuse) por meio da concatenação de blocos convolucionais, o que ajuda na extração de bordas difusas de tumores sem explodir o número de parâmetros.

Resultados consolidados da Segunda Leva (Avaliação no Conjunto de Teste isolado):

| Arquitetura | Resolução | Acurácia Teste | Perda Teste | F1-Score Macro |
| :--- | :---: | :---: | :---: | :---: |
| `cnn_custom` | 128x128 | 26.65% | 1.4646 | 0.1250 |
| `densenet121` | 224x224 | 75.13% | 1.2579 | 0.7205 |
| `inception_v3` | 299x299 | 78.43% | 1.3235 | 0.7666 |
| `efficientnet_b0` | 224x224 | 79.70% | 1.0612 | 0.7700 |

*Nota Analítica:* A `cnn_custom`, por ser uma rede desenvolvida e treinada estritamente do zero (*from scratch*), sofreu limitações graves na taxa de acerto nesta segunda rodada. Sem a bagagem pré-treinada do ImageNet, ela exigiria um banco de dados substancialmente maior e rodadas massivas de *Data Augmentation* para competir, validando empiricamente a escolha primordial por *Transfer Learning* no uso de imagens médicas de pequeno volume.

### 2.5. Histórico e Decisões de Hiperparâmetros (`config.yaml` e Arquitetura)

Durante o desenvolvimento, cometi erros que me ensinaram bastante sobre as particularidades de imagens médicas. A seguir, registro os principais desafios e como cheguei às decisões finais:

#### Problema com Batch Normalization
Inicialmente, na fase de *Fine-tuning*, mantive as camadas de `BatchNormalization` congeladas. Esse é o padrão ensinado na maioria dos tutoriais de Transfer Learning para fotos normais (cachorros, carros). No entanto, quando fui avaliar os modelos (InceptionV3 e EfficientNet), a acurácia despencou. O motivo? As estatísticas visuais (luminosidade e contraste) de uma ressonância magnética (MRI) são radicalmente diferentes das fotos da base ImageNet. As camadas congeladas tentavam normalizar cérebros usando a média de cor de gatos e cachorros. **Decisão:** Passei a descongelar as camadas de BatchNorm no fine-tuning. A rede imediatamente voltou a aprender.

#### Erro de Precisão Mista (FP16)
Ativei o Mixed Precision (cálculos em 16-bits) para treinar mais rápido e economizar memória na GPU. Funcionou bem, mas esbarrei em instabilidade numérica (NaN loss) no modelo `EfficientNet`. Descobri que, ao lidar com a camada de saída (`Softmax`), as probabilidades matemáticas exigem altíssima precisão. **Decisão:** Adicionei explicitamente `dtype="float32"` na última camada Densa para forçar o Softmax a operar em precisão máxima, estabilizando os gradientes sem perder velocidade.

#### Shortcut Learning: Glioma vs. Hipófise
Ao longo dos treinamentos, notei uma confusão frequente do modelo entre Gliomas e Tumores de Hipófise. Isso é intrínseco aos dados: gliomas costumam ter bordas muito difusas e irregulares (se misturando ao cérebro), enquanto os de hipófise ficam numa região anatômica muito específica (Sela Túrcica). 

Ao investigar o mapa de calor (Grad-CAM), percebi um erro grave: o modelo estava sofrendo de **Shortcut Learning** (aprendizado por atalho). Em vez de olhar para a glândula no centro do cérebro, a rede decorou o contorno do crânio nas extremidades laterais das imagens de hipófise (provavelmente um artefato das máquinas de raio-x daquele dataset).

Para curar o modelo desses "atalhos", adotei duas táticas:
1. **Remoção de Bordas (Zoom-In Forçado):** Configurei o `zoom_range` no Keras Data Augmentation para `[-0.15, -0.05]`. Isso obriga a rede a dar um corte de 5% a 15% nas laterais de *todas* as imagens de treino, escondendo o crânio e forçando o foco no tecido interno.
2. **Weight Decay (Regularização L2):** Injetei `kernel_regularizer=l2(0.01)` nas camadas Densas finais. Isso penaliza pesos excessivamente altos, impedindo que a IA confie em apenas dois ou três "pixels fáceis" na ponta da tela. A matemática a obriga a espalhar a atenção por toda a textura, o que ajudou na leitura difusa dos Gliomas.

#### Configurações Finais
Após os ajustes acima, este foi o consenso que entregou resiliência sem *overfitting*:

```yaml
training:
  batch_size: 32
  epochs: 30
  stage2_lr: 0.0001
  optimizer: adam
  loss_function: categorical_crossentropy
  early_stopping_patience: 8
augmentation:
  rotation_range: 15
  zoom_range: [-0.15, -0.05] # Esconde o crânio
  width_shift_range: 0.15    # Balanço lateral agressivo
```
Essas configurações, aliadas ao pipeline do `tf.data`, me permitiram extrair o máximo de um dataset complexo, desafiador e com forte ruído anatômico.
