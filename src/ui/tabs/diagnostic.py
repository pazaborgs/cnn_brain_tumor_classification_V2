from pathlib import Path

import numpy as np
import PIL.Image
import streamlit as st

from src.preprocess import get_model_image_size
from src.ui.components.charts import (
    CLASS_COLORS,
    CLASS_LABELS_PT,
    CLASS_NAMES,
)
from src.ui.components.containers import window


def render_diagnostic_tab(
    config: dict,
    selected_model_name: str,
    sample_path: Path | None,
    run_prediction_fn,
    get_last_conv_layer_name_fn,
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
            icon=":material/info:",
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

    # --- Seção 1: Badge de Diagnóstico e Status ---
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
                    icon=":material/check_circle:",
                )
            else:
                st.error(
                    f"Divergência Diagnóstica (Erro): O modelo previu '{predicted_class_pt}', mas o rótulo real é '{true_class_pt}'.",
                    icon=":material/error:",
                )

        col_m1, col_m2, col_m3 = st.columns(3, gap="medium")
        col_m1.metric("Nível de Confiança", f"{confidence:.2f}%")
        col_m2.metric("Tempo de Processamento", f"{inference_ms:.1f} ms")
        col_m3.metric(
            "Resolução da Imagem", f"{target_image_size}×{target_image_size} px"
        )

    # --- Seção 2: Análise Visual (Slider + Probabilidades) ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_slider, col_chart = st.columns([1, 1], gap="medium")

    with col_slider:
        with window("analise_gradcam_xai", "compare", height=580):
            from streamlit_image_comparison import image_comparison

            pil_original = PIL.Image.fromarray(img_rgb)
            pil_gradcam = PIL.Image.fromarray(gradcam_overlay)
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            _, c_center, _ = st.columns([1, 8, 1])
            with c_center:
                image_comparison(
                    img1=pil_original,
                    img2=pil_gradcam,
                    label1="MRI",
                    label2="XAI",
                    starting_position=50,
                    show_labels=True,
                    width=420,
                    make_responsive=False,
                    in_memory=True,
                )
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    with col_chart:
        with window("distribuicao_probabilidade", "analytics", height=580):
            _render_probability_panel(preds, predicted_class_en, confidence, color)


def _render_probability_panel(
    preds: np.ndarray,
    predicted_class_en: str,
    confidence: float,
    color: str,
):
    """Renderiza painel compacto de distribuição de probabilidades."""
    sorted_indices = np.argsort(preds)[::-1]

    bars_html = ""
    for idx in sorted_indices:
        cls_en = CLASS_NAMES[idx]
        cls_pt = CLASS_LABELS_PT[cls_en]
        cls_color = CLASS_COLORS[cls_en]
        prob = float(preds[idx]) * 100
        is_top = cls_en == predicted_class_en
        bg_bar = (
            cls_color
            if is_top
            else f"rgba({int(cls_color[1:3], 16)}, {int(cls_color[3:5], 16)}, {int(cls_color[5:7], 16)}, 0.25)"
        )
        bar_opacity = "1" if is_top else "0.8"
        fw = "700" if is_top else "500"
        text_color = "#FFFFFF" if is_top else "#94A3B8"
        bars_html += (
            f'<div style="opacity: {bar_opacity};">'
            f'  <div style="display:flex; justify-content:space-between; margin-bottom: 0.3rem;">'
            f'      <span style="color:{text_color}; font-size: 0.85rem; font-weight:{fw};">{cls_pt}</span>'
            f'      <span style="color:{cls_color}; font-size: 0.85rem; font-weight:700;">{prob:.1f}%</span>'
            f"  </div>"
            f'  <div style="background:#1E293B; height:12px; border-radius:6px; width:100%; overflow:hidden;">'
            f'      <div style="background:{bg_bar}; height:100%; width:{max(prob, 1):.1f}%; border-radius:6px; transition:width 0.5s ease;"></div>'
            f"  </div>"
            f"</div>"
        )

    svg_donut = (
        f'<div style="position:relative; width:120px; height:120px; margin: 0 auto;">'
        f'  <svg viewBox="0 0 36 36" style="width:120px; height:120px; transform:rotate(-90deg); drop-shadow(0px 0px 8px {color}40);">'
        f'      <circle cx="18" cy="18" r="15.9" fill="none" stroke="#1E293B" stroke-width="3"/>'
        f'      <circle cx="18" cy="18" r="15.9" fill="none" stroke="{color}" stroke-width="3" '
        f'          stroke-dasharray="{confidence} {100 - confidence}" stroke-linecap="round"/>'
        f"  </svg>"
        f'  <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);'
        f"      font-family:'JetBrains Mono', monospace; font-weight:800; font-size:1.2rem; color:#F8FAFC;\">"
        f"      {confidence:.0f}%"
        f"  </div>"
        f"</div>"
    )

    full_html = (
        f'<div style="border:1px solid #334155; border-radius:12px; padding:1.8rem 1.5rem; margin: 0.5rem;'
        f'background:linear-gradient(145deg, #0F172A 0%, #1E293B 100%); box-shadow:0 10px 30px rgba(0,0,0,0.3);'
        f'display: flex; flex-direction: column; justify-content: space-between; height: 500px;">'
        f'  <div>'
        f'      <div style="margin-bottom: 2rem;">{svg_donut}</div>'
        f'      <div style="border-top:1px dashed #334155; margin-bottom: 1.5rem;"></div>'
        f'  </div>'
        f'  <div style="display: flex; flex-direction: column; justify-content: space-between; flex: 1;">'
        f'      {bars_html}'
        f'  </div>'
        f"</div>"
    )

    st.markdown(full_html, unsafe_allow_html=True)
