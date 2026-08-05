import streamlit as st
import numpy as np
import PIL.Image
from pathlib import Path

from src.ui.components.containers import window
from src.ui.components.charts import probability_chart, CLASS_NAMES, CLASS_LABELS_PT, CLASS_COLORS
from src.preprocess import get_model_image_size


def render_diagnostic_tab(
    config: dict, 
    selected_model_name: str, 
    sample_path: Path | None,
    run_prediction_fn, 
    get_last_conv_layer_name_fn
):
    """Renderiza a Aba 1: Diagnóstico por Imagem e Visualização Grad-CAM.

    Args:
        config (dict): Dicionário de configuração global.
        selected_model_name (str): Nome do modelo Keras ativo.
        sample_path (Path | None): Caminho do arquivo da imagem de teste (MRI).
        run_prediction_fn (callable): Função para inferência e heatmap.
        get_last_conv_layer_name_fn (callable): Função auxiliar para o Grad-CAM.
    """
    if sample_path is None:
        st.info("Nenhuma imagem de amostra disponível.", icon=":material/info:")
        return

    target_image_size = get_model_image_size(config, selected_model_name)
    last_conv_layer = get_last_conv_layer_name_fn(config, selected_model_name)

    with st.spinner("Analisando exame MRI e gerando mapa de calor Grad-CAM..."):
        img_rgb, preds, inference_ms, gradcam_overlay = run_prediction_fn(
            selected_model_name, str(sample_path), target_image_size, last_conv_layer
        )

    if preds is None:
        st.info(
            f"O modelo `{selected_model_name}.keras` não foi encontrado na pasta `artifacts/models/`.\n\n"
            f"Para utilizar este modelo para inferência ao vivo, treine-o localmente via `python main.py --mode train`.",
            icon=":material/info:"
        )
        st.image(PIL.Image.open(sample_path), width="stretch")
        return

    pred_index = int(np.argmax(preds))
    confidence = float(preds[pred_index]) * 100
    predicted_class_en = CLASS_NAMES[pred_index]
    predicted_class_pt = CLASS_LABELS_PT[predicted_class_en]
    color = CLASS_COLORS[predicted_class_en]

    true_class_en = sample_path.parent.name if sample_path else None
    is_correct = (
        (predicted_class_en == true_class_en) if true_class_en in CLASS_NAMES else True
    )

    with window(f"laudo_preliminar — {selected_model_name}", "monitor_heart"):
        st.markdown(
            f'<span class="diag-badge" style="background:{color};">{predicted_class_pt}</span>',
            unsafe_allow_html=True,
        )

        if true_class_en in CLASS_NAMES:
            true_class_pt = CLASS_LABELS_PT[true_class_en]
            if is_correct:
                st.success(
                    f"Diagnóstico Preciso: O modelo previu '{predicted_class_pt}', correspondendo ao rótulo real do exame.",
                    icon=":material/check_circle:"
                )
            else:
                st.error(
                    f"Divergência Diagnóstica (Erro): O modelo previu '{predicted_class_pt}', mas o rótulo real é '{true_class_pt}'.",
                    icon=":material/error:"
                )

        col_m1, col_m2, col_m3 = st.columns(3, gap="medium")
        col_m1.metric("Nível de Confiança", f"{confidence:.2f}%")
        col_m2.metric("Tempo de Processamento", f"{inference_ms:.1f} ms")
        col_m3.metric(
            "Resolução da Imagem", f"{target_image_size}×{target_image_size} px"
        )

    # Reduzindo a largura das colunas das imagens para mascarar a baixa resolução original (espreme para o centro)
    _, col_v1, col_v2, _ = st.columns([0.4, 2, 2, 0.4], gap="medium")
    
    with col_v1:
        with window("exame_mri.png", "image"):
            st.caption("Imagem de Entrada (MRI)")
            st.image(img_rgb, width="stretch", output_format="PNG")

    with col_v2:
        with window("gradcam_heatmap.png", "local_fire_department"):
            st.caption("IA Explicável (Grad-CAM)")
            st.image(gradcam_overlay, width="stretch", output_format="PNG")

    # Gráfico ganha um pouco de respiro usando as mesmas margens
    _, col_chart, _ = st.columns([0.4, 4, 0.4])
    with col_chart:
        with window("distribuicao_probabilidades", "bar_chart"):
            st.caption("Distribuição de Probabilidade")
            st.altair_chart(
                probability_chart(preds, predicted_class_en, height=250), width="stretch"
            )
