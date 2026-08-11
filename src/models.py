"""Módulo de Arquitetura e Construção de Modelos Keras 3.

Define os extratores de características (InceptionV3, EfficientNetB0),
camadas de preprocessamento serializáveis e blocos de aumento de dados (Data Augmentation).
Suporta treinamento em 2 estágios: Feature Extraction (backbone congelado) e Fine-tuning.
"""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_preprocessing_layer(model_name: str) -> layers.Layer:
    """Retorna a camada de pré-processamento compatível e serializável para o Keras 3.

    Args:
        model_name (str): Identificador da camada ('rescale', 'inception_v3', 'efficientnet').

    Returns:
        layers.Layer: Camada Keras nativa para normalização dos pixels.

    Raises:
        ValueError: Se o identificador de pré-processamento não for reconhecido.
    """
    if model_name == "rescale":
        return layers.Rescaling(1.0 / 255.0)
    elif model_name == "inception_v3":
        return layers.Rescaling(1.0 / 127.5, offset=-1.0)
    elif model_name == "efficientnet":
        return layers.Activation("linear")
    elif model_name == "densenet":
        return layers.Rescaling(1.0 / 255.0)
    else:
        print(f"[ERRO] Pré-processamento desconhecido: {model_name}")
        raise ValueError(f"Preprocessing desconhecido: {model_name}")


def augmentation_block(aug_config: dict) -> tf.keras.Sequential:
    """Constrói o bloco de aumento de dados (Data Augmentation) em tempo de execução.

    Aplica rotações, zooms, translações e flips aleatórios nas imagens de treino.

    Args:
        aug_config (dict): Dicionário com as configurações de data augmentation.

    Returns:
        tf.keras.Sequential: Sequencial contendo as camadas de transformação Keras.
    """
    aug_layers = [
        layers.RandomRotation(factor=aug_config["rotation_range"] / 360),
        layers.RandomZoom(
            height_factor=aug_config["zoom_range"],
            width_factor=aug_config["zoom_range"],
        ),
        layers.RandomTranslation(
            height_factor=aug_config["height_shift_range"],
            width_factor=aug_config["width_shift_range"],
        ),
    ]

    horizontal = aug_config.get("horizontal_flip", False)
    vertical = aug_config.get("vertical_flip", False)

    if horizontal and vertical:
        aug_layers.append(layers.RandomFlip("horizontal_and_vertical"))
    elif horizontal:
        aug_layers.append(layers.RandomFlip("horizontal"))
    elif vertical:
        aug_layers.append(layers.RandomFlip("vertical"))

    return tf.keras.Sequential(aug_layers, name="augmentation")


# ---------------------------------------------------------------------------
# CNN Custom Backbone (desativada do pipeline ativo)
# ---------------------------------------------------------------------------
# def build_cnn_custom_backbone() -> tf.keras.Model:
#     """Constrói uma arquitetura CNN convolucional otimizada com blocos duplos de convolução.
#
#     Utiliza a Functional API do Keras, permitindo resolução variável e fatiamento
#     nacional para XAI. Inicialização He Normal, Batch Normalization pós-convolução,
#     Dropout espacial progressivo e um mapa convolucional final de alta resolução.
#
#     Returns:
#         tf.keras.Model: Backbone convolucional customizado otimizado.
#     """
#     inputs = tf.keras.layers.Input(shape=(None, None, 3), name="input_layer")
#
#     x = layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal", name="conv2d_1a")(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal", name="conv2d_1b")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Dropout(0.15)(x)
#
#     x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", name="conv2d_2a")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", name="conv2d_2b")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Dropout(0.25)(x)
#
#     x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", name="conv2d_3a")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", name="conv2d_3b")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Dropout(0.35)(x)
#
#     x = layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal", name="last_feature_map")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, 2)(x)
#     x = layers.Dropout(0.40)(x)
#
#     outputs = layers.GlobalAveragePooling2D()(x)
#     return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_custom_backbone")


def backbone(model_name: str) -> tf.keras.layers.Layer:
    """Retorna o backbone extrator de características com todas as camadas congeladas (Stage 1).

    O descongelamento seletivo para fine-tuning (Stage 2) é feito pela função
    ``unfreeze_backbone`` após o Stage 1 de Feature Extraction.

    Args:
        model_name (str): Nome do modelo ('inception_v3', 'efficientnet_b0').

    Returns:
        tf.keras.layers.Layer: Modelo Keras base com pesos ImageNet congelados.

    Raises:
        ValueError: Se o nome do modelo não for reconhecido.
    """
    if model_name == "inception_v3":
        base = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", pooling="avg"
        )
        base.trainable = False
        return base
    elif model_name == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg"
        )
        base.trainable = False
        return base

    # --- Modelos desativados do pipeline ativo ---
    # elif model_name == "cnn_custom":
    #     return build_cnn_custom_backbone()
    # elif model_name == "densenet121":
    #     base = tf.keras.applications.DenseNet121(
    #         include_top=False, weights="imagenet", pooling="avg"
    #     )
    #     base.trainable = False
    #     return base
    else:
        print(f"[ERRO] Backbone desconhecido: {model_name}")
        raise ValueError(f"Modelo desconhecido: {model_name}")


def unfreeze_backbone(model: tf.keras.Model, fine_tune_from: int) -> None:
    """Descongela as últimas N camadas do backbone para fine-tuning (Stage 2).

    Diferente de tarefas em fotos comuns, para imagens médicas (MRI), 
    as camadas de BatchNormalization nas últimas N camadas devem ser 
    descongeladas para aprenderem as novas estatísticas de contraste 
    e luminosidade do cérebro.

    Args:
        model: Modelo Keras completo contendo o backbone como sub-model.
        fine_tune_from: Número de camadas finais do backbone a descongelar.
    """
    for layer in model.layers:
        if hasattr(layer, "layers") and layer.name != "augmentation":
            layer.trainable = True
            for sublayer in layer.layers[:-fine_tune_from]:
                sublayer.trainable = False
            trainable_count = sum(1 for l in layer.layers if l.trainable)
            print(f"[INFO] Backbone '{layer.name}': {trainable_count} camadas descongeladas")
            break


def build_model(config: dict, model_config: dict, num_classes: int) -> tf.keras.Model:
    """Constrói, estrutura e compila o modelo de Aprendizado Profundo.

    Empilha a camada de pré-processamento específica, o bloco de aumento de dados,
    o backbone selecionado e a cabeça de classificação totalmente conectada.
    Usa label smoothing na loss para melhorar generalização.

    Args:
        config (dict): Dicionário de configuração global.
        model_config (dict): Dicionário de configuração da arquitetura específica.
        num_classes (int): Número total de classes de saída.

    Returns:
        tf.keras.Model: Modelo Keras compilado e pronto para treinamento.
    """
    train_cfg = config["training"]
    target_image_size = model_config.get(
        "image_size", config["preprocessing"]["canonical_image_size"]
    )

    print(f"[INFO] Construindo arquitetura do modelo '{model_config['name']}'...")

    inputs = tf.keras.Input(
        shape=(
            target_image_size,
            target_image_size,
            3,
        ),
        name="input_layer",
    )

    x = get_preprocessing_layer(model_config["preprocessing"])(inputs)
    aug = augmentation_block(config["augmentation"])
    x = aug(x)
    x = backbone(model_config["name"])(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="output_layer",
        dtype="float32",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_config["name"],
    )

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    optimizer = tf.keras.optimizers.get(
        {
            "class_name": train_cfg["optimizer"],
            "config": {
                "learning_rate": train_cfg["learning_rate"],
            },
        }
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )

    print(f"[SUCESSO] Modelo '{model.name}' compilado.")
    print(f"[INFO] Parâmetros totais: {model.count_params():,}")
    print(
        f"[INFO] Optimizer: {train_cfg['optimizer']} | Label Smoothing: {label_smoothing}"
    )

    figures_path = Path(config["paths"]["figures"])
    figures_path.mkdir(parents=True, exist_ok=True)

    try:
        tf.keras.utils.plot_model(
            model,
            to_file=figures_path / f"{model_config['name']}.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True,
            dpi=150,
        )
    except Exception as e:
        print(f"[AVISO] Não foi possível gerar a figura da arquitetura: {e}")

    return model


if __name__ == "__main__":
    config = load_config()
    for model_cfg in config["models"]:
        build_model(config, model_cfg, 4)
