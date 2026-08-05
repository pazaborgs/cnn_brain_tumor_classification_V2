"""Módulo de Preprocessamento e Pipeline de Dados de Imagem.

Carregamento, redimensionamento, divisão em batches e balanceamento
de peso de classes para exames de ressonância magnética (MRI).
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Calcula os pesos das classes para equilibrar a função de perda durante o treino.

    Utiliza a estratégia 'balanced' do scikit-learn para compensar o desequilíbrio
    entre as 4 classes de tumores cerebrais no conjunto de dados.

    Args:
        train_dir (Path): Diretorio contendo as subpastas de cada classe.
        class_names (list[str]): Nomes das classes na ordem ordinal do Keras.

    Returns:
        dict[int, float]: Mapeamento do indice da classe para o peso correspondente.
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    counts = [
        sum(
            1
            for f in (train_dir / c).iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        )
        for c in class_names
    ]
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), counts),
    )
    return dict(enumerate(weights))


def get_model_image_size(config: dict, model_name: str | None = None) -> int:
    """Retorna o tamanho das imagens de entrada do modelo (altura, largura).

    Args:
        config (dict): Dicionario de configuracao global.
        model_name (str, optional): Nome da arquitetura do modelo.

    Returns:
        int: Dimensao em pixels para redimensionar as imagens (ex.: 128, 224, 299).
    """
    if not model_name:
        return int(config["preprocessing"]["canonical_image_size"])

    models = config.get("models", [])
    for model_cfg in models:
        if model_cfg.get("name") == model_name:
            return int(
                model_cfg.get("image_size", config["preprocessing"]["canonical_image_size"])
            )

    return int(config["preprocessing"]["canonical_image_size"])


def generate_image_datasets(
    config: dict | None = None,
    model_name: str | None = None,
    image_size: int | None = None,
):
    """Carrega e prepara os datasets de treino, validação e teste a partir dos arquivos MRI.

    Aplica o redimensionamento dinâmico para a resolução alvo do modelo, configura
    batches, atribui codificação categorical para os rótulos e calcula pesos das classes.

    Args:
        config (dict, optional): Dicionario de configuracao global.
        model_name (str, optional): Nome do modelo para buscar resolucao alvo.
        image_size (int, optional): Tamanho de imagem sobrescrito manualmente.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_weights, class_names)

    Raises:
        FileNotFoundError: Se o diretorio de imagens raw nao for encontrado.
    """
    if config is None:
        config = load_config()

    if image_size is None:
        image_size = get_model_image_size(config, model_name)

    print(f"[INFO] Processando e gerando datasets de imagens (Resolução: {image_size}x{image_size})...")

    target_shape = (image_size, image_size)
    images_path = Path(config["paths"]["raw"])
    seed = int(config["project"]["seed"])
    batch_size = int(config["training"]["batch_size"])
    validation_split = float(config["preprocessing"]["validation_split"])

    if not images_path.exists():
        print(f"[ERRO] Diretório de imagens não encontrado em: {images_path}")
        raise FileNotFoundError("Dataset nao encontrado. Execute ingest.py primeiro.")

    train_dir = images_path / "Training"
    test_dir = images_path / "Testing"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=target_shape,
        batch_size=batch_size,
        label_mode="categorical",
        verbose=0,
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=target_shape,
        batch_size=batch_size,
        label_mode="categorical",
        verbose=0,
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=target_shape,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
        verbose=0,
    )

    class_names = train_dataset.class_names
    assert test_dataset.class_names == class_names, (
        "Mismatch de classes entre treino e teste"
    )

    class_weights = compute_class_weights(train_dir, class_names)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("[SUCESSO] Datasets de treino, validação e teste preparados com prefetch AUTOTUNE.")
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Batch Size: {batch_size} | Resolution: {image_size}x{image_size}")
    print(f"[INFO] Origem: {images_path}")

    return train_dataset, val_dataset, test_dataset, class_weights, class_names


if __name__ == "__main__":
    generate_image_datasets()
