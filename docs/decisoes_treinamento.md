# Relatório Técnico: Ajuste de Hiperparâmetros, Arquitetura e Estratégia de Fine-Tuning

Registro técnico das decisões arquiteturais, otimizações de pipeline I/O, integração em nuvem e ajustes de hiperparâmetros para o treinamento dos modelos de aprendizado profundo na base de dados de classificação de tumores cerebrais em MRI (`sartajbhuvaji/brain-tumor-classification-mri`).

---

## 1. Benchmark Inicial (20 Épocas)

Resultados da execução baseline com extratores de características estáticos (pesos do ImageNet congelados):

| Arquitetura | Resolução | Acurácia Val. | Perda Val. | Acurácia Teste | F1-Score Macro | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `cnn_custom` | 128x128 | 29.27% | 2.7504 | 18.78% | 0.0791 | Overfitting (Parada na época 8) |
| `inception_v3` | 299x299 | 85.19% | 0.3923 | 66.75% | 0.6531 | Estável na Validação, Queda no Teste |
| `efficientnet_b0` | 224x224 | 80.14% | 0.5913 | 67.77% | 0.6473 | Estável na Validação, Queda no Teste |

---

## 2. Análise Técnica dos Gargalos Identificados

1. **Extratores Congelados (`base.trainable = False`):**
   A acurácia de validação do InceptionV3 atingiu 85.19%, mas a acurácia de teste caiu para 66.75%. Os pesos pré-treinados no ImageNet necessitam de descongelamento parcial para adaptar os filtros às características específicas de tecidos em ressonância magnética.

2. **Parada Precoce Agressiva (`patience: 3`):**
   A paciência de 3 épocas interrompeu o treinamento do modelo customizado antes da convergência durante oscilações normais de validação.

3. **Taxa de Aprendizado Elevada (`lr = 0.001`):**
   A taxa inicial de 0.001 gerou variações bruscas na função de perda durante o fine-tuning.

4. **Arquitetura Rasa no Modelo Customizado:**
   O modelo convolucional customizado baseline possuía apenas uma camada de convolução por bloco sem inicialização especializada, causando forte overfitting e estagnação em acurácia.

---

## 3. Modificações Aplicadas (Fase II)

1. **Reformulação da Arquitetura `cnn_custom` (Blocos Duplos VGG/ResNet-style):**
   - Implementação de **blocos de convolução dupla** (`Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU`).
   - Adição de inicializador **He Normal** (`kernel_initializer="he_normal"`) calibrado para relu.
   - **Dropout Espacial Progressivo** (`0.15` $\rightarrow$ `0.25` $\rightarrow$ `0.35` $\rightarrow$ `0.40`) para preservar bordas em blocos iniciais e regularizar representações densas profundas.

2. **Descongelamento Seletivo das Camadas (Fine-Tuning):**
   - `inception_v3`: Descongelamento das últimas 40 camadas (`base.layers[:-40]`).
   - `efficientnet_b0`: Descongelamento das últimas 30 camadas (`base.layers[:-30]`).

3. **Otimização de Pipeline I/O:**
   - Adição de `prefetch(tf.data.AUTOTUNE)` nas compilações de dataset (`src/preprocess.py`) para eliminação de gargalos no consumo de CPU/GPU.

4. **Hospedagem e Model Registry na Nuvem (Hugging Face Hub):**
   - Integração com o repositório [`pazaborgs/brain_tumor_classification_V2`](https://huggingface.co/pazaborgs/brain_tumor_classification_V2).
   - Suporte a autenticação via `.env` (local) e Kaggle Secrets (`HF_TOKEN`) para publicação e download sob demanda.

---

## 4. Matriz de Hiperparâmetros Final (`config/config.yaml`)

```yaml
training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.0003
  optimizer: adam
  loss_function: categorical_crossentropy
  early_stopping_patience: 8
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5
```

---

## 5. Execução no Kaggle GPU & Publicação

```bash
# 1. Atualizar repositório e rodar treinamento de 30 épocas
!git pull origin main
!python main.py --mode train

# 2. Publicar modelos treinados no Hugging Face Model Hub
!python main.py --mode upload_hf
```
