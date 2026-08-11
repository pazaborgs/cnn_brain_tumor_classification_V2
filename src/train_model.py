"""Módulo de Treinamento de Modelos e Rastreamento MLflow.

Orquestra o ciclo de vida de treinamento em 2 estágios (Feature Extraction + Fine-tuning),
salva binários .keras em artifacts/models/ e registra métricas no MLflow.
Suporta Mixed Precision Training para aceleração em GPUs T4/A100.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf

from src.evaluate_model import evaluate_model, plot_results
from src.models import build_model, unfreeze_backbone
from src.preprocess import generate_image_datasets, get_model_image_size
from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_model(
    config: dict, model_config: dict, epochs: int | None = None
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Executa o treinamento em 2 estágios e registra artefatos/métricas no MLflow.

    Estágio 1 (Feature Extraction): Backbone congelado, treina apenas o head de classificação
    com LR alta para alinhar as features ImageNet às classes de tumor.

    Estágio 2 (Fine-tuning): Descongela as últimas N camadas do backbone com LR baixa
    para ajustar features discriminativas. BN permanece congelada.

    Args:
        config (dict): Dicionário contendo as configurações globais do projeto.
        model_config (dict): Dicionário contendo as configurações específicas do modelo.
        epochs (int | None, optional): Override do total de épocas do Stage 2.

    Returns:
        tuple[tf.keras.Model, tf.keras.callbacks.History]: Modelo treinado e histórico combinado.
    """
    train_cfg = config["training"]

    stage1_epochs = int(model_config.get("stage1_epochs", 10))
    stage2_epochs = epochs if epochs is not None else int(train_cfg["epochs"])
    stage2_lr = float(model_config.get("stage2_lr", 1e-5))
    fine_tune_from = int(model_config.get("fine_tune_from", 40))

    models_path = Path(config["paths"]["models"])
    models_path.mkdir(parents=True, exist_ok=True)

    figures_path = Path(config["paths"]["figures"])
    figures_path.mkdir(parents=True, exist_ok=True)

    target_image_size = get_model_image_size(config, model_config["name"])

    print(f"[INFO] Preparando treinamento do modelo '{model_config['name']}'...")
    print(f"[INFO] Target Image Size: {target_image_size}x{target_image_size}")
    print(f"[INFO] Stage 1: {stage1_epochs} epochs (LR={train_cfg['learning_rate']})")
    print(f"[INFO] Stage 2: {stage2_epochs} epochs (LR={stage2_lr})")

    # Mixed Precision
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed Precision (FP16) ativado.")
    except Exception:
        print("[AVISO] Mixed Precision não disponível, usando FP32.")

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

    experiment_name = config["mlflow"].get(
        "experiment_name", config["mlflow"].get("experiment_path")
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model.name):
        mlflow.log_params({f"train_{k}": v for k, v in train_cfg.items()})
        mlflow.log_params({f"model_{k}": v for k, v in model_config.items()})
        mlflow.log_param("dataset", config["dataset"]["name"])
        mlflow.log_param("seed", config["project"]["seed"])
        mlflow.log_param("stage1_epochs", stage1_epochs)
        mlflow.log_param("stage2_epochs", stage2_epochs)
        mlflow.log_param("stage2_lr", stage2_lr)
        mlflow.log_param("fine_tune_from", fine_tune_from)

        # ===== STAGE 1: Feature Extraction (backbone congelado) =====

        print(f"\n{'=' * 60}")
        print("  STAGE 1 — Feature Extraction (Head-Only)")
        print(f"{'=' * 60}")

        callbacks_s1 = [
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

        history_s1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=stage1_epochs,
            callbacks=callbacks_s1,
            class_weight=class_weights,
        )

        best_s1_acc = max(history_s1.history.get("val_accuracy", [0]))
        print(f"[INFO] Stage 1 concluído — Best Val Accuracy: {best_s1_acc:.2%}")

        # ===== STAGE 2: Fine-tuning (backbone parcialmente descongelado) =====

        print(f"\n{'=' * 60}")
        print(f"  STAGE 2 — Fine-tuning Discriminativo (Top {fine_tune_from} layers)")
        print(f"{'=' * 60}")

        unfreeze_backbone(model, fine_tune_from)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=stage2_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
            ),
            metrics=["accuracy"],
        )

        callbacks_s2 = [
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
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history_s2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=stage2_epochs,
            callbacks=callbacks_s2,
            class_weight=class_weights,
        )

        # Combinar históricos para plots unificados

        combined_history_dict = {}
        for key in history_s1.history:
            combined_history_dict[key] = history_s1.history[
                key
            ] + history_s2.history.get(key, [])

        class CombinedHistory:
            """Container para histórico combinado de 2 estágios."""

            def __init__(self, hist_dict):
                self.history = hist_dict

        combined_history = CombinedHistory(combined_history_dict)

        # Métricas

        if "val_loss" in combined_history.history:
            best_epoch_idx = int(np.argmin(combined_history.history["val_loss"]))
            mlflow.log_metric(
                "train_loss", combined_history.history["loss"][best_epoch_idx]
            )
            mlflow.log_metric(
                "train_accuracy", combined_history.history["accuracy"][best_epoch_idx]
            )
            mlflow.log_metric(
                "val_loss", combined_history.history["val_loss"][best_epoch_idx]
            )
            mlflow.log_metric(
                "val_accuracy", combined_history.history["val_accuracy"][best_epoch_idx]
            )
        else:
            mlflow.log_metric("train_loss", combined_history.history["loss"][-1])
            mlflow.log_metric(
                "train_accuracy", combined_history.history["accuracy"][-1]
            )

        model_path = models_path / f"{model.name}.keras"
        model.save(model_path)

        history_fig_path = figures_path / f"{model.name}_history.png"
        history_fig = plot_results(
            combined_history, title_prefix=model.name, save_path=history_fig_path
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
            max(combined_history.history["val_accuracy"])
            if "val_accuracy" in combined_history.history
            else 0.0
        )
        best_loss = (
            min(combined_history.history["val_loss"])
            if "val_loss" in combined_history.history
            else 0.0
        )

        total_epochs = len(combined_history.history["loss"])
        print(
            f"\n[SUCESSO] Treinamento do modelo '{model.name}' finalizado com sucesso."
        )
        print(
            f"[INFO] Épocas Totais: {total_epochs} (S1: {len(history_s1.history['loss'])} + S2: {len(history_s2.history['loss'])})"
        )
        print(
            f"[INFO] Best Val Accuracy: {best_acc:.2%} | Best Val Loss: {best_loss:.4f}"
        )
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

    # Mixed Precision Reset (para evitar problemas em outros treinos subsequentes)
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    return model, combined_history


if __name__ == "__main__":
    config = load_config()

    for model_cfg in config["models"]:
        train_model(config=config, model_config=model_cfg)
