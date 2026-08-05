from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

from src.preprocess import get_model_image_size


def get_mlflow_runs_dataframe():
    """Tenta carregar as runs nativamente via MLflow, com fallback para CSV.

    Returns:
        pd.DataFrame | None: DataFrame limpo e unificado com as runs agrupadas por modelo,
        ou None caso não haja dados disponíveis no banco de dados ou no snapshot.
    """
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
        df_runs = (
            mlflow.search_runs(experiment_ids=all_experiments)
            if all_experiments
            else pd.DataFrame()
        )

        if df_runs.empty:
            raise ValueError("Banco de dados vazio.")

        clean_df = pd.DataFrame()
        if "tags.mlflow.runName" in df_runs.columns:
            clean_df["Model Name"] = df_runs["tags.mlflow.runName"]
        elif "tags.model_name" in df_runs.columns:
            clean_df["Model Name"] = df_runs["tags.model_name"]

        metrics = [
            "metrics.train_accuracy",
            "metrics.train_loss",
            "metrics.val_accuracy",
            "metrics.val_loss",
            "metrics.test_accuracy",
            "metrics.test_loss",
            "metrics.macro_f1_score",
            "metrics.weighted_f1_score",
        ]
        for m in metrics:
            if m in df_runs.columns:
                clean_name = m.replace("metrics.", "").replace("_", " ").title()
                clean_df[clean_name] = df_runs[m]

        params = [
            "params.train_batch_size",
            "params.train_epochs",
            "params.train_learning_rate",
            "params.model_base",
            "params.model_weights",
            "params.dataset",
        ]
        for p in params:
            if p in df_runs.columns:
                clean_name = (
                    p.replace("params.", "")
                    .replace("train_", "")
                    .replace("model_", "")
                    .replace("_", " ")
                    .title()
                )
                clean_df[clean_name] = df_runs[p]

        clean_df["Status"] = df_runs["status"]
        if "end_time" in df_runs.columns:
            clean_df["Data"] = df_runs["end_time"].dt.strftime("%d/%m %H:%M")

    except Exception:
        snapshot_path = Path("artifacts/metrics/mlflow_runs_snapshot.csv")
        if snapshot_path.exists():
            clean_df = pd.read_csv(snapshot_path)
        else:
            return None

    if clean_df is not None and not clean_df.empty:
        clean_df["Model Name"] = (
            clean_df["Model Name"]
            .astype(str)
            .str.replace("_evaluation", "", regex=False)
        )
        clean_df = clean_df.groupby("Model Name", as_index=False).first()

    return clean_df


def render_mlops_tab(config: dict, models_list: list, load_trained_model_fn):
    """Renderiza a Aba 3: Métricas & Relatórios MLOps.

    Args:
        config (dict): Dicionário de configuração global.
        models_list (list): Lista de nomes dos modelos disponíveis.
        load_trained_model_fn (callable): Função de cache para carregar a instância do modelo.
    """
    st.header("Relatórios e Matrizes MLOps")
    st.caption(
        "Visão geral de todas as execuções de treinamento e métricas registradas."
    )
    st.divider()

    st.markdown("#### Histórico de Execuções (Leaderboard)")
    df_runs = get_mlflow_runs_dataframe()
    if df_runs is not None and not df_runs.empty:
        st.dataframe(
            df_runs,
            width="stretch",
            hide_index=True,
            column_config={
                "Train Accuracy": st.column_config.ProgressColumn(
                    "Acurácia (Treino)", format="%.2f", min_value=0, max_value=1
                ),
                "Val Accuracy": st.column_config.ProgressColumn(
                    "Acurácia (Val)", format="%.2f", min_value=0, max_value=1
                ),
                "Test Accuracy": st.column_config.ProgressColumn(
                    "Acurácia (Teste)", format="%.2f", min_value=0, max_value=1
                ),
                "Macro F1 Score": st.column_config.NumberColumn(
                    "Macro F1 (Teste)", format="%.4f"
                ),
                "Weighted F1 Score": st.column_config.NumberColumn(
                    "Weighted F1 (Teste)", format="%.4f"
                ),
                "Test Loss": st.column_config.NumberColumn(
                    "Perda (Teste)", format="%.4f"
                ),
                "Val Loss": st.column_config.NumberColumn("Perda (Val)", format="%.4f"),
            },
        )
    else:
        st.info(
            "Nenhum histórico de execução encontrado (MLflow DB ou Snapshot não disponíveis)."
        )

    st.divider()

    if not models_list:
        st.info("Nenhum modelo disponível para exibir.")
        return

    model_tabs = st.tabs(models_list)

    for tab, model_name in zip(model_tabs, models_list):
        with tab:
            model_target_size = get_model_image_size(config, model_name)
            model_instance = load_trained_model_fn(model_name)

            run_info = None
            if (
                df_runs is not None
                and not df_runs.empty
                and model_name in df_runs["Model Name"].values
            ):
                run_info = df_runs[df_runs["Model Name"] == model_name].iloc[0]

            from src.ui.components.containers import window

            with window(f"painel_mlops — {model_name}", "analytics"):
                # Top metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Resolução", f"{model_target_size}×{model_target_size}")

                if model_instance is not None:
                    col2.metric("Parâmetros", f"{model_instance.count_params():,}")
                else:
                    col2.metric("Parâmetros", "N/A")

                epochs_val = (
                    run_info["Epochs"]
                    if run_info is not None and "Epochs" in run_info
                    else "N/A"
                )
                batch_val = (
                    run_info["Batch Size"]
                    if run_info is not None and "Batch Size" in run_info
                    else "N/A"
                )
                lr_val = (
                    run_info["Learning Rate"]
                    if run_info is not None and "Learning Rate" in run_info
                    else "N/A"
                )

                col3.metric("Épocas", epochs_val)
                col4.metric("Batch Size", batch_val)
                col5.metric("Learning Rate", lr_val)

                st.divider()

                # Imagens melhoradas em layout vertical (grafico historico é muito largo para dividir tela)
                st.markdown("#### Histórico de Treinamento e Evolução de Perda")
                hist_path = Path(f"artifacts/figures/{model_name}_history.png")
                if hist_path.exists():
                    st.image(str(hist_path), width="stretch", output_format="PNG")
                else:
                    st.info(
                        "Execute `python main.py --mode train` para gerar o gráfico de histórico.",
                        icon=":material/info:",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("#### Matriz de Confusão (Dados de Teste)")
                cm_path = Path(f"artifacts/figures/{model_name}_confusion_matrix.png")

                # A matriz é mais quadrada, podemos centralizá-ela numa coluna do meio para não ficar esticada demais
                _, col_cm, _ = st.columns([1, 2, 1])
                with col_cm:
                    if cm_path.exists():
                        st.image(str(cm_path), width="stretch", output_format="PNG")
                    else:
                        st.info(
                            "Execute `python main.py --mode evaluate` para gerar a matriz de confusão.",
                            icon=":material/info:",
                        )

                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander(
                    "Visualizar Arquitetura da Rede Neural (Grafo)", expanded=False
                ):
                    arch_path = Path(f"artifacts/figures/{model_name}.png")
                    if arch_path.exists():
                        st.image(str(arch_path), width="stretch", output_format="PNG")
                    else:
                        st.info(
                            f"O diagrama de arquitetura '{model_name}.png' não foi encontrado em artifacts/figures/.",
                            icon=":material/info:",
                        )
