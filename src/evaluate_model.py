"""Módulo de Avaliação e Geração de Métricas MLOps.

Calcula métricas de desempenho (Acurácia, Perda, Relatório de Classificação),
gera matrizes de confusão e registra artefatos no MLflow.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.preprocess import generate_image_datasets, get_model_image_size
from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sns.set_theme(style="whitegrid", font="monospace")
plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#0F172A",
        "xtick.labelsize": 10,
        "xtick.color": "#334155",
        "ytick.labelsize": 10,
        "ytick.color": "#334155",
        "figure.facecolor": "#F8FAFC",
        "figure.titlesize": 15,
        "text.color": "#0F172A",
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.8,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#CBD5E1",
        "legend.fontsize": 10,
        "savefig.facecolor": "#F8FAFC",
        "savefig.edgecolor": "#F8FAFC",
    }
)


def plot_results(
    history: tf.keras.callbacks.History,
    title_prefix: str = "Modelo",
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera gráficos comparativos da evolução de acurácia e perda durante o treinamento.

    Args:
        history (tf.keras.callbacks.History): Objeto de histórico retornado pelo `model.fit()`.
        title_prefix (str, optional): Prefixo a ser exibido no título do gráfico. Default é "Modelo".
        save_path (Path | None, optional): Caminho opcional para salvar o gráfico como imagem.

    Returns:
        plt.Figure: Objeto Figure do Matplotlib contendo os gráficos lado a lado.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        f"Desempenho no Treinamento - {title_prefix}", fontweight="bold", y=1.02
    )

    ax1.plot(
        epochs,
        acc,
        label="Treino",
        color="#3b82f6",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        val_acc,
        label="Validacao",
        color="#10b981",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    if val_acc:
        best_acc_epoch = int(np.argmax(val_acc)) + 1
        best_acc = max(val_acc)
        ax1.scatter(
            [best_acc_epoch],
            [best_acc],
            color="#059669",
            s=100,
            zorder=5,
            label=f"Melhor Val ({best_acc:.2%})",
        )

    ax1.set_title("Evolucao da Acuracia")
    ax1.set_xlabel("Epocas")
    ax1.set_ylabel("Acuracia")
    ax1.set_ylim([0, 1.02])
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(
        epochs,
        loss,
        label="Treino",
        color="#3b82f6",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs,
        val_loss,
        label="Validacao",
        color="#f43f5e",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    if val_loss:
        best_loss_epoch = int(np.argmin(val_loss)) + 1
        best_loss = min(val_loss)
        ax2.scatter(
            [best_loss_epoch],
            [best_loss],
            color="#e11d48",
            s=100,
            zorder=5,
            label=f"Menor Perda ({best_loss:.4f})",
        )

    ax2.set_title("Evolucao da Perda (Loss)")
    ax2.set_xlabel("Epocas")
    ax2.set_ylabel("Perda Categorical Crossentropy")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Matriz de Confusao",
    save_path: Path | None = None,
) -> plt.Figure:
    """Gera e exibe uma matriz de confusão customizada para o conjunto de teste.

    Args:
        y_true (np.ndarray): Array com os rótulos reais.
        y_pred (np.ndarray): Array com os rótulos previstos pelo modelo.
        class_names (list[str]): Lista com o nome das classes na ordem correta.
        title (str, optional): Título do gráfico. Default é "Matriz de Confusao".
        save_path (Path | None, optional): Caminho opcional para salvar o gráfico.

    Returns:
        plt.Figure: Objeto Figure do Matplotlib contendo a matriz de confusão plotada.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y_true, y_pred)

    project_cmap = LinearSegmentedColormap.from_list(
        "project", ["#F0FDF4", "#A7F3D0", "#34D399", "#059669", "#047857"]
    )

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=project_cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=1.5,
        linecolor="#F8FAFC",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=14, fontweight="bold", color="#0F172A")
    ax.set_xlabel(
        "Classes Preditas pelo Modelo", fontsize=11, labelpad=10, color="#334155"
    )
    ax.set_ylabel(
        "Classes Reais (Ground Truth)", fontsize=11, labelpad=10, color="#334155"
    )
    ax.tick_params(axis="x", rotation=20, colors="#334155")
    ax.tick_params(axis="y", rotation=0, colors="#334155")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def evaluate_model(
    config: dict,
    model_config: dict,
    active_run: bool = False,
    test_dataset: tf.data.Dataset = None,
    class_names: list[str] = None,
) -> dict:
    """Avalia um modelo salvo carregando-o e inferindo sobre o conjunto de teste.

    Calcula perda, acurácia, gera relatório de classificação, cria matriz de confusão
    e registra todos os dados no MLflow.

    Args:
        config (dict): Dicionário contendo as configurações globais do projeto.
        model_config (dict): Dicionário contendo as configurações específicas do modelo.
        active_run (bool, optional): Indica se já existe uma run do MLflow ativa.
        test_dataset (tf.data.Dataset, optional): Dataset pré-carregado.
        class_names (list[str], optional): Lista pré-carregada de nomes das classes.

    Returns:
        dict: Dicionário contendo métricas sumarizadas, relatório de classificação e figura.
    """

    if test_dataset is None or class_names is None:
        target_image_size = get_model_image_size(config, model_config["name"])
        _, _, test_dataset, _, class_names = generate_image_datasets(
            config=config, image_size=target_image_size
        )

    model_name = model_config["name"]
    models_path = Path(config["paths"]["models"])
    figures_path = Path(config["paths"]["figures"])
    model_file = models_path / f"{model_name}.keras"

    if not model_file.exists():
        print(
            f"[ERRO] Modelo {model_file} não encontrado. Execute train_model.py primeiro."
        )
        raise FileNotFoundError(f"Arquivo do modelo {model_file} não encontrado.")

    print(f"[INFO] Avaliando modelo '{model_name}' no conjunto de Teste...")

    custom_objects = {}
    if model_config.get("preprocessing") == "inception_v3":
        custom_objects["preprocess_input"] = (
            tf.keras.applications.inception_v3.preprocess_input
        )

    model = tf.keras.models.load_model(
        model_file, custom_objects=custom_objects, compile=False
    )
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    target_image_size = get_model_image_size(config, model_name)

    _, _, test_dataset, _, class_names = generate_image_datasets(
        config=config, image_size=target_image_size
    )

    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)

    y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)

    predictions = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm_save_path = figures_path / f"{model_name}_confusion_matrix.png"
    cm_fig = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"Matriz de Confusao - {model_name}",
        save_path=cm_save_path,
    )
    plt.close(cm_fig)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Log de Metricas no MLflow
    def _log_metrics_to_mlflow():
        mlflow.log_param("image_size", target_image_size)
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("macro_f1_score", float(report_dict["macro avg"]["f1-score"]))
        mlflow.log_metric(
            "weighted_f1_score", float(report_dict["weighted avg"]["f1-score"])
        )

        for class_name in class_names:
            if class_name in report_dict:
                safe_name = class_name.replace(" ", "_")
                mlflow.log_metric(
                    f"recall_{safe_name}", float(report_dict[class_name]["recall"])
                )
                mlflow.log_metric(
                    f"precision_{safe_name}",
                    float(report_dict[class_name]["precision"]),
                )
                mlflow.log_metric(
                    f"f1_{safe_name}", float(report_dict[class_name]["f1-score"])
                )

        if cm_save_path.exists():
            mlflow.log_artifact(cm_save_path)

    if active_run:
        _log_metrics_to_mlflow()
    else:
        experiment_name = config["mlflow"].get(
            "experiment_name", config["mlflow"].get("experiment_path")
        )
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            _log_metrics_to_mlflow()

    macro_f1 = float(report_dict["macro avg"]["f1-score"])

    print(f"[SUCESSO] Avaliação do modelo '{model_name}' concluída.")
    print(
        f"[INFO] Test Accuracy: {test_acc:.2%} | Test Loss: {test_loss:.4f} | Macro F1: {macro_f1:.4f}"
    )
    print(f"[INFO] Matriz de Confusão salva em: {cm_save_path}")

    return {
        "model_name": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "cm_figure": cm_fig,
    }


if __name__ == "__main__":
    config = load_config()
    for model_cfg in config["models"]:
        try:
            evaluate_model(config, model_cfg)
        except Exception as e:
            print(f"\n[ERRO] Nao foi possivel avaliar {model_cfg['name']}: {e}\n")
