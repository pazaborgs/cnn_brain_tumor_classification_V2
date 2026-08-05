# 🧠 Classificação de Tumores Cerebrais em MRI via Deep Learning (V2)

[![Live Demo: Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cnnbraintumorclassificationv2-dzjnrtq4axyk3r8eue7eve.streamlit.app/)
[![Hugging Face Hub](https://img.shields.io/badge/Model%20Hub-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/pazaborgs/brain_tumor_classification_V2)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework: Keras 3 / TensorFlow](https://img.shields.io/badge/Framework-Keras%203%20%7C%20TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://keras.io/)
[![MLOps: MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Gerenciador: uv](https://img.shields.io/badge/Gerenciador-uv-DE5FE9?style=flat&logo=astral&logoColor=white)](https://astral.sh/uv)

Projeto de aprendizado profundo focado no diagnóstico de tumores cerebrais a partir de exames de ressonância magnética (MRI). Esta segunda versão (V2) refatora o código acadêmico original em uma estrutura modular orientada a MLOps, integrada ao **Hugging Face Model Hub** e equipada com **IA Explicável (Grad-CAM)** em tempo real.

---

## 🌐 Aplicação Interativa em Nuvem (Live Demo)

Acesse a aplicação web em tempo real hospedada no **Streamlit Cloud**:  
👉 **[https://cnnbraintumorclassificationv2-dzjnrtq4axyk3r8eue7eve.streamlit.app/](https://cnnbraintumorclassificationv2-dzjnrtq4axyk3r8eue7eve.streamlit.app/)**

![Brain Tumor Classification V2 Streamlit Demo](artifacts/figures/git_landing_page.gif)

- **Model Registry em Nuvem:** Os binários dos modelos `.keras` são baixados automaticamente do [Hugging Face Hub](https://huggingface.co/pazaborgs/brain_tumor_classification_V2) sob demanda.
- **Amostras Integradas:** Inclui 12 exames de ressonância magnética de amostra (3 por classe) integrados para teste imediato.
- **IA Explicável (Grad-CAM):** Gera mapas térmicos interativos em tempo real para visualizar a atenção visual de cada modelo.

---

## 📌 Evolução do Projeto (V1 vs V2)

| Componente              | Versão 1 (Notebook Original)                   | Versão 2 (Estrutura Atual)                                                         |
| :---------------------- | :--------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Estrutura de Código** | Células sequenciais no Google Colab (`.ipynb`) | **Módulos Python** (`src/`, `main.py`) com docstrings no padrão Google             |
| **Modelos**             | 2 modelos fixos (`CNN Custom`, `InceptionV3`)  | **Suporte Multi-Modelo** (`CNN Custom`, `InceptionV3`, `EfficientNetB0`, `DenseNet121`) |
| **Parâmetros**          | Variáveis declaradas no código                 | **Arquivo de Configuração YAML** ([config/config.yaml](config/config.yaml))        |
| **MLOps & Registry**    | Métricas exibidas no console                   | **MLflow & Hugging Face Hub** (`pazaborgs/brain_tumor_classification_V2`)          |
| **Artefatos**           | Arquivos na sessão temporária                  | **Diretório de Artefatos & Nuvem** (`artifacts/models/`, Hugging Face)             |
| **IA Explicável (XAI)** | Ausente                                        | **Grad-CAM (GradientTape)** com cálculo focado e mapa térmico `jet`                |
| **Dependências**        | Instalação via `pip` padrão                    | **Gerenciamento com `uv`** ([pyproject.toml](pyproject.toml))                      |
| **Execução**            | Manual por célula                              | **Interface CLI** (`python main.py --mode [train\|evaluate\|upload_hf\|app]`)      |
| **Visualização**        | Nenhuma interface                              | **Aplicação Streamlit** ([app.py](app.py)) com download automático de modelos      |
| **Segurança**           | Token `kaggle.json` no notebook                | **Variáveis de ambiente** ([.env.example](.env.example), [.gitignore](.gitignore)) |

---

## 🗺️ Roadmap e Status de Desenvolvimento

### 🟢 Concluído

- [x] Refatoração do repositório para pacote Python modular (`src/`) com docstrings no padrão Google PT-BR.
- [x] Configuração centralizada via YAML ([config/config.yaml](config/config.yaml)).
- [x] Integração com MLflow para rastreamento unificado de experimentos.
- [x] **Hugging Face Model Registry & Nuvem:** Upload automatizado (`python main.py --mode upload_hf`) e download sob demanda no Streamlit a partir do repositório [`pazaborgs/brain_tumor_classification_V2`](https://huggingface.co/pazaborgs/brain_tumor_classification_V2).
- [x] **Implantação em Nuvem no Streamlit Cloud:** Live Demo acessível em [`https://cnnbraintumorclassificationv2-dzjnrtq4axyk3r8eue7eve.streamlit.app/`](https://cnnbraintumorclassificationv2-dzjnrtq4axyk3r8eue7eve.streamlit.app/).
- [x] **Amostras Integradas no Git:** 12 imagens de amostragem MRI em `data/sample_mri/` para teste imediato.
- [x] **Arquitetura `cnn_custom` Otimizada:** Blocos duplos de convolução no estilo VGG/ResNet, inicialização `he_normal` e dropout espacial progressivo (`0.15` a `0.40`).
- [x] **Otimização de Pipeline I/O:** Pré-carregamento com `prefetch(tf.data.AUTOTUNE)` para eliminação de gargalos na GPU.
- [x] **Integração Grad-CAM no App:** Visualização de mapas de calor de interpretabilidade (IA Explicável) em tempo real com `tf.GradientTape()`.
- [x] Implementação da CLI multi-modo (`main.py --mode [train|evaluate|upload_hf|app]`).
- [x] Dashboard interativo em Streamlit com suporte a inferência ao vivo e seleção amigável de amostras.
- [x] Estratégia de fine-tuning com descongelamento seletivo de camadas superiores (`mixed10`, `top_conv`).
- [x] Versionamento da configuração visual do Streamlit ([.streamlit/config.toml](.streamlit/config.toml)).
- [x] **Design System e UI/UX:** Tipografia moderna (*Outfit*), paleta clínica (Tailwind colors), layout responsivo sem perdas de compressão (Lossless PNG) e hierarquia visual avançada no Leaderboard.
- [x] Inclusão do documento do TCC original ([docs/TCC_brain_tumor_classification.pdf](docs/TCC_brain_tumor_classification.pdf)).

### 🟡 Em Desenvolvimento (Work in Progress - WIP)

- [ ] **Treinamento Final de 30 Épocas:** Execução do fine-tuning completo no Kaggle GPU T4 e atualização dos binários `.keras` no Hugging Face Hub.

---

## 🎓 Contexto Acadêmico (TCC UNIVESP)

Trabalho desenvolvido para obtenção do título de Bacharel em Ciência de Dados pela **UNIVESP** (Universidade Virtual do Estado de São Paulo), 2025.

- **Autores:** Patrick Regis (& Grupo)
- **Orientador:** Prof. Darwish Ahmad Herati
- **Documento do TCC (Monografia em PDF):** [docs/TCC_brain_tumor_classification.pdf](docs/TCC_brain_tumor_classification.pdf)
- **Documentação de Treinamento:** As decisões técnicas e ajustes de hiperparâmetros estão registradas em [docs/decisoes_treinamento.md](docs/decisoes_treinamento.md).

---

## 🩻 Base de Dados

Base de dados: **Brain Tumor Classification (MRI)** do Kaggle.

- **Classes (4):** `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`.
- **Volume:** 3.264 imagens de ressonância magnética (`.jpg`).
- **Link do Kaggle:** [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

---

## 📁 Estrutura do Projeto

```text
cnn_brain_tumor_classification_V2/
├── config/
│   └── config.yaml             # Hiperparâmetros e caminhos do projeto
├── src/
│   ├── utils/
│   │   ├── load_config.py      # Carregamento da configuração YAML
│   │   └── upload_models_hf.py # Upload de modelos para o Hugging Face Hub
│   ├── ingest.py               # Ingestão de dados via Kaggle API
│   ├── preprocess.py           # Preprocessamento e balanceamento de classes
│   ├── models.py               # Construção e compilação das arquiteturas (Keras 3)
│   ├── train_model.py          # Loop de treino e integração com MLflow
│   ├── evaluate_model.py       # Avaliação e geração de matrizes de confusão
│   └── gradcam.py              # Módulo de interpretabilidade (Grad-CAM)
├── docs/
│   ├── TCC_brain_tumor_classification.pdf  # Monografia completa do TCC
│   └── decisoes_treinamento.md             # Documentação técnica de fine-tuning
├── artifacts/
│   ├── models/                 # Binários dos modelos (.keras) - não versionados
│   └── figures/                # Gráficos e matrizes de confusão (.png)
├── data/
│   └── sample_mri/             # 12 imagens de amostra MRI (versionadas no Git)
├── .streamlit/
│   └── config.toml             # Configuração do tema e visual do Streamlit
├── git_landing_page.gif        # Demonstração animada da interface Streamlit
├── main.py                     # CLI e orquestrador do pipeline
├── app.py                      # Dashboard interativo Streamlit
├── pyproject.toml              # Especificação de dependências (uv)
├── .env.example                # Template de variáveis de ambiente
└── .gitignore                  # Exclusões do versionamento Git
```

---

## 🛠️ Como Executar Localmente (Passo a Passo)

### 📋 Pré-requisitos

- **Python 3.10 ou superior** instalado na máquina.
- **Gerenciador `uv`** (recomendado para instalação rápida de dependências):
  - No Windows (PowerShell): `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - No Linux/macOS: `curl -sSf https://astral.sh/uv/install.sh | sh`

---

### Passo 1: Clonar o Repositório e Sincronizar Dependências

```bash
# Clonar o repositório
git clone https://github.com/pazaborgs/cnn_brain_tumor_classification_V2.git
cd cnn_brain_tumor_classification_V2

# Criar ambiente virtual e instalar todas as dependências automaticamente
uv sync
```

---

### Passo 2: Configurar Credenciais da API do Kaggle e Hugging Face

1. Copie o arquivo `.env.example` para `.env`:
   ```bash
   cp .env.example .env
   ```
2. Adicione suas credenciais do **Kaggle API** e **Hugging Face Token (Write)**:
   ```env
   KAGGLE_API_TOKEN=sua_chave_ou_json_do_kaggle_aqui
   HF_TOKEN=seu_token_de_escrita_do_huggingface_aqui
   ```

---

### Passo 3: Executar o Pipeline CLI (`main.py`)

O orquestrador [main.py](main.py) permite executar diferentes etapas do projeto via argumento `--mode`:

#### 🚀 Opção A: Ingestão, Treinamento e Avaliação Completa

```bash
uv run python main.py --mode train
```

#### 📤 Opção B: Publicar Modelos Treinados no Hugging Face Hub

```bash
uv run python main.py --mode upload_hf
```

#### 📊 Opção C: Avaliar Modelos Treinados Existentes

```bash
uv run python main.py --mode evaluate
```

#### 🌐 Opção D: Iniciar o App Streamlit via CLI

```bash
uv run python main.py --mode app
```

---

### Passo 4: Iniciar o Dashboard Interativo (`app.py`)

Você também pode iniciar a aplicação web diretamente via Streamlit:

```bash
uv run streamlit run app.py
```

---

### ⚠️ Nota Legal

Este projeto foi desenvolvido para fins acadêmicos e de pesquisa (TCC UNIVESP). Não substitui laudos médicos nem deve ser utilizado para diagnósticos clínicos sem validação apropriada.
