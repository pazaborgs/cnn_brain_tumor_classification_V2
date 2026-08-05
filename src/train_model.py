"""Módulo de Treinamento de Modelos e Rastreamento MLflow.

Orquestra o ciclo de vida de treinamento (Callbacks, Fine-tuning, Weight decay),
salva binários .keras em artifacts/models/ e registra métricas no MLflow.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf

from src.evaluate_model import evaluate_model, plot_results
from src.models import build_model
from src.preprocess import generate_image_datasets, get_model_image_size
from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_model(
    config: dict, model_config: dict, epochs: int | None = None
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Executa o treinamento e validação de um modelo Keras, registrando artefatos e métricas no MLflow.

    Prepara os datasets baseados na configuração, constrói a arquitetura especificada,
    define os callbacks (EarlyStopping, ReduceLROnPlateau) e realiza o fit.
    Por fim, salva o modelo em disco e no registry do MLflow.

    Args:
        config (dict): Dicionário contendo as configurações globais do projeto.
        model_config (dict): Dicionário contendo as configurações específicas do modelo.
        epochs (int | None, optional): Número de épocas para treinar. Se None, utiliza o do config.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: Uma tupla contendo o modelo
        Keras treinado e o histórico de treinamento (History).
    """

    train_cfg = config["training"]
    num_epochs = epochs if epochs is not None else train_cfg["epochs"]

    models_path = Path(config["paths"]["models"])
    models_path.mkdir(parents=True, exist_ok=True)

    figures_path = Path(config["paths"]["figures"])
    figures_path.mkdir(parents=True, exist_ok=True)

    target_image_size = get_model_image_size(config, model_config["name"])

    print(f"[INFO] Preparando treinamento do modelo '{model_config['name']}'...")
    print(f"[INFO] Target Image Size: {target_image_size}x{target_image_size}")
    print(f"[INFO] Epochs: {num_epochs} | Batch Size: {train_cfg['batch_size']}")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        class_weights,
        class_names,
    ) = generate_image_datasets(config=config, image_size=target_image_size)

    model = build_model(
        config=config,
        model_config=model_config,
        num_classes=len(class_names),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=train_cfg["reduce_lr_patience"],
            factor=train_cfg["reduce_lr_factor"],
            verbose=1,
        ),
    ]

    experiment_name = config["mlflow"].get(
        "experiment_name", config["mlflow"].get("experiment_path")
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model.name):
        mlflow.log_params({f"train_{k}": v for k, v in train_cfg.items()})
        mlflow.log_params({f"model_{k}": v for k, v in model_config.items()})
        mlflow.log_param("dataset", config["dataset"]["name"])
        mlflow.log_param("seed", config["project"]["seed"])

        print(f"[INFO] Executando fit para {model.name}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
        )

        if "val_loss" in history.history:
            best_epoch_idx = int(np.argmin(history.history["val_loss"]))
            mlflow.log_metric("train_loss", history.history["loss"][best_epoch_idx])
            mlflow.log_metric(
                "train_accuracy", history.history["accuracy"][best_epoch_idx]
            )
            mlflow.log_metric("val_loss", history.history["val_loss"][best_epoch_idx])
            mlflow.log_metric(
                "val_accuracy", history.history["val_accuracy"][best_epoch_idx]
            )
        else:
            mlflow.log_metric("train_loss", history.history["loss"][-1])
            mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])

        model_path = models_path / f"{model.name}.keras"
        model.save(model_path)

        history_fig_path = figures_path / f"{model.name}_history.png"
        history_fig = plot_results(
            history, title_prefix=model.name, save_path=history_fig_path
        )
        plt.close(history_fig)
        if history_fig_path.exists():
            mlflow.log_artifact(history_fig_path)

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            pip_requirements=["tensorflow", "keras", "mlflow"],
        )
        mlflow.log_artifact(model_path)

        figure = figures_path / f"{model.name}.png"
        if figure.exists():
            mlflow.log_artifact(figure)

        best_acc = (
            max(history.history["val_accuracy"])
            if "val_accuracy" in history.history
            else 0.0
        )
        best_loss = (
            min(history.history["val_loss"]) if "val_loss" in history.history else 0.0
        )

        print(f"[SUCESSO] Treinamento do modelo '{model.name}' finalizado com sucesso.")
        print(f"[INFO] Épocas Executadas: {len(history.history['loss'])}")
        print(f"[INFO] Best Val Accuracy: {best_acc:.2%} | Best Val Loss: {best_loss:.4f}")
        print(f"[INFO] Modelo salvo em: {model_path}")

        try:
            evaluate_model(
                config=config,
                model_config=model_config,
                active_run=True,
                test_dataset=test_dataset,
                class_names=class_names,
            )
        except Exception as e:
            print(f"[ERRO] Não foi possível executar avaliação automática: {e}")

    return model, history


if __name__ == "__main__":
    config = load_config()

    for model_cfg in config["models"]:
        train_model(config=config, model_config=model_cfg)
