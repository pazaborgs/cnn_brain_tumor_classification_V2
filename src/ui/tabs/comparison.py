import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.ui.components.containers import window
from src.ui.components.charts import CLASS_NAMES, CLASS_LABELS_PT
from src.preprocess import get_model_image_size

def render_comparison_tab(
    config: dict, 
    models_list: list, 
    sample_path: Path | None,
    run_prediction_fn, 
    get_last_conv_layer_name_fn
):
    """Renderiza a Aba 2: Comparação Direta de Arquiteturas.

    Args:
        config (dict): Dicionário de configuração global.
        models_list (list): Lista de nomes dos modelos disponíveis.
        sample_path (Path | None): Caminho para a imagem de teste (MRI).
        run_prediction_fn (callable): Função responsável pela inferência e heatmap.
        get_last_conv_layer_name_fn (callable): Função para obter a camada base do Grad-CAM.
    """
    if sample_path is None:
        st.info("Selecione uma imagem de amostra no Painel de Controle.", icon=":material/info:")
        return

    compare_models = st.multiselect(
        "Selecione as arquiteturas para comparação direta:",
        options=models_list,
        default=models_list,
        key="compare_models_tab2",
    )

    if not compare_models:
        st.warning("Selecione ao menos uma arquitetura acima.", icon=":material/warning:")
        return

    rows, preds_by_model = [], {}
    for model_name in compare_models:
        model_target_size = get_model_image_size(config, model_name)
        last_conv = get_last_conv_layer_name_fn(config, model_name)
        _, model_preds, model_ms, _ = run_prediction_fn(
            model_name, str(sample_path), model_target_size, last_conv
        )
        if model_preds is not None:
            pred_idx = int(np.argmax(model_preds))
            preds_by_model[model_name] = model_preds
            rows.append(
                {
                    "Arquitetura": model_name,
                    "Laudo Predito": CLASS_LABELS_PT[CLASS_NAMES[pred_idx]],
                    "Nível de Confiança (%)": float(model_preds[pred_idx]) * 100,
                    "Tempo Processamento (ms)": model_ms,
                    "_classe_raw": CLASS_NAMES[pred_idx],
                }
            )

    if not rows:
        st.info("Nenhum modelo `.keras` encontrado em `artifacts/models/` para gerar a comparação.", icon=":material/info:")
        return

    df_compare = pd.DataFrame(rows)
    consensus = df_compare["_classe_raw"].nunique() == 1

    with window("tabela_comparativa_modelos", "science"):
        if consensus:
            st.success(
                f"Consenso entre as {len(rows)} arquiteturas: Todos os modelos indicam o mesmo laudo preliminar.",
                icon=":material/check_circle:"
            )
        else:
            st.warning(
                "Divergência preditiva: As arquiteturas apresentam divergência no diagnóstico preliminar.",
                icon=":material/warning:"
            )

        st.dataframe(
            df_compare.drop(columns="_classe_raw").style.format(
                {
                    "Nível de Confiança (%)": "{:.2f}%",
                    "Tempo Processamento (ms)": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
