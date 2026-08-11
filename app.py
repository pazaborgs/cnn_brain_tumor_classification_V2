"""
Brain Tumor MRI Classification — Painel Interativo em Streamlit.

Design de alto desempenho com inferência ao vivo e suporte a IA Explicável (Grad-CAM),
arquitetura modularizada (componentes em src/ui) e compatibilidade com Streamlit atualizada.
"""

import os
import time
from pathlib import Path

import numpy as np
import PIL.Image
import streamlit as st

# Ocultar logs verbosos de CPU/GPU do TensorFlow e resolver aviso de imagem longa (DecompressionBombWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
PIL.Image.MAX_IMAGE_PIXELS = None

import tensorflow as tf

from src.gradcam import make_gradcam_heatmap, overlay_gradcam
from src.ui.components.charts import CLASS_LABELS_PT, CLASS_NAMES
from src.ui.components.containers import (
    groupbox_end,
    groupbox_label,
    inject_css_from_file,
    render_app_titlebar,
)
from src.ui.tabs.comparison import render_comparison_tab
from src.ui.tabs.diagnostic import render_diagnostic_tab
from src.ui.tabs.mlops import render_mlops_tab
from src.utils.load_config import load_config

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_app_config() -> dict:
    """Carrega o arquivo de configuração YAML global.

    Returns:
        dict: Dicionário contendo as configurações globais do aplicativo.
    """
    return load_config()


def get_last_conv_layer_name(config: dict, model_name: str) -> str | None:
    """Retorna o nome da camada convolucional alvo cadastrada no config.yaml.

    Args:
        config (dict): Dicionário de configuração da aplicação.
        model_name (str): Nome do modelo a ser verificado.

    Returns:
        str | None: O nome da camada convolucional se especificado, caso contrário None.
    """
    for m in config.get("models", []):
        if m["name"] == model_name:
            return m.get("last_conv_layer")
    return None


@st.cache_resource(show_spinner=True, max_entries=1)
def load_trained_model(model_name: str):
    """Carrega o modelo Keras localmente ou faz download automático do Hugging Face Hub.

    Args:
        model_name (str): Nome do modelo a ser carregado.

    Returns:
        keras.Model | None: A instância do modelo treinado ou None em caso de falha.
    """
    model_path = Path(f"artifacts/models/{model_name}.keras")
    if not model_path.exists():
        config = load_app_config()
        repo_id = config.get("huggingface", {}).get(
            "repo_id", "pazaborgs/brain_tumor_classification_V2"
        )
        try:
            from huggingface_hub import hf_hub_download

            model_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_name}.keras",
                local_dir="artifacts/models",
            )
        except Exception:
            return None

    if not model_path.exists():
        return None

    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image_for_model(pil_img: PIL.Image.Image, target_image_size: int):
    """Prepara a imagem PIL convertendo para RGB uint8 e tensor float32 normalizado.

    Args:
        pil_img (PIL.Image.Image): Imagem crua em formato PIL.
        target_image_size (int): Resolução alvo quadrada para redimensionamento.

    Returns:
        tuple: Tupla contendo o array RGB (np.ndarray) e o tensor final (np.ndarray).
    """
    pil_rgb = pil_img.convert("RGB")
    img_rgb = np.array(pil_rgb)
    pil_resized = pil_rgb.resize((target_image_size, target_image_size))
    img_tensor = np.expand_dims(np.array(pil_resized), axis=0).astype(np.float32)
    return img_rgb, img_tensor


def run_prediction(
    model_name: str,
    image_path_str: str,
    target_image_size: int,
    last_conv_layer_name: str,
):
    """Executa a inferência Keras e gera a sobreposição Grad-CAM precisa em tempo real.

    Args:
        model_name (str): Nome do modelo alvo.
        image_path_str (str): Caminho absoluto da imagem no disco.
        target_image_size (int): Resolução alvo em pixels suportada pelo modelo.
        last_conv_layer_name (str): Nome da camada base para extração do heatmap.

    Returns:
        tuple: Array original, predições brutas (ndarray), tempo em ms (float),
               e a imagem final em Grad-CAM (ndarray).
    """
    pil_img = PIL.Image.open(image_path_str)
    img_rgb, img_tensor = preprocess_image_for_model(pil_img, target_image_size)

    model = load_trained_model(model_name)
    if model is None:
        return img_rgb, None, 0.0, img_rgb

    start = time.time()
    preds = model.predict(img_tensor, verbose=0)[0]
    inference_ms = (time.time() - start) * 1000

    pred_index = int(np.argmax(preds))

    if not last_conv_layer_name:
        print(
            f"[AVISO] last_conv_layer_name ausente para {model_name}. Grad-CAM desabilitado."
        )
        heatmap = None
    else:
        heatmap = make_gradcam_heatmap(
            img_tensor,
            model,
            last_conv_layer_name=last_conv_layer_name,
            pred_index=pred_index,
        )

    gradcam_overlay = overlay_gradcam(img_rgb, heatmap, alpha=0.45, colormap="jet")

    return img_rgb, preds, inference_ms, gradcam_overlay


@st.cache_data(show_spinner=True)
def list_sample_files(config_paths_raw: str) -> dict:
    """Carrega exatamente 3 amostras por classe com rótulos amigáveis para o dropdown.

    Args:
        config_paths_raw (str): Diretório raiz dos dados brutos a ser escaneado.

    Returns:
        dict: Dicionário contendo labels de amostra como chave e seus caminhos absolutos como valor.
    """
    root_dir = Path(__file__).resolve().parent
    search_dirs = [
        root_dir / "data" / "sample_mri",
        root_dir / config_paths_raw / "Testing",
    ]
    formatted_samples = {}

    for c in CLASS_NAMES:
        class_label = CLASS_LABELS_PT[c]
        c_files = []
        for base_dir in search_dirs:
            c_dir = base_dir / c
            if c_dir.exists():
                imgs = sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")))
                c_files.extend(imgs)
                if len(c_files) >= 3:
                    break

        for i, img_path in enumerate(c_files[:3], start=1):
            dropdown_label = f"{class_label} — Amostra #{i}"
            formatted_samples[dropdown_label] = img_path

    return formatted_samples


def main():
    """Inicializa a interface gráfica do Streamlit central."""
    inject_css_from_file()
    config = load_app_config()

    render_app_titlebar()

    models_list = ["inception_v3",]
    default_model_index = (
        models_list.index("inception_v3") if "inception_v3" in models_list else 0
    )

    with st.sidebar:
        st.markdown(
            '<div class="os-titlebar"><div class="os-titlebar-left"><span class="material-symbols-rounded" style="font-size:18px; margin-right:6px; vertical-align:middle;">settings_input_component</span><span style="vertical-align:middle;">Painel de Controle</span></div></div>',
            unsafe_allow_html=True,
        )

        groupbox_label("Modelo Selecionado")
        selected_model_name = st.selectbox(
            "Arquitetura Ativa",
            options=models_list,
            index=default_model_index,
            label_visibility="collapsed",
            help="Alterne entre os diferentes modelos de IA treinados para comparar seus laudos e diagnósticos.",
        )
        groupbox_end()

        groupbox_label("Exame MRI (Amostra)")
        sample_options = list_sample_files(config["paths"]["raw"])
        sample_path = None
        if sample_options:
            selected_sample_key = st.selectbox(
                "Selecione uma imagem de teste",
                options=list(sample_options.keys()),
                label_visibility="collapsed",
                help="Selecione o caso clínico (paciente) que deseja analisar. As abas carregarão o resultado instantaneamente.",
            )
            sample_path = sample_options[selected_sample_key]
        else:
            st.warning("Nenhuma amostra encontrada em data/raw/Testing.")
        groupbox_end()

        st.markdown(
            f'<span class="status-pill">TF {tf.__version__}</span> <span class="status-pill">{len(models_list)} Modelo(s)</span>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(
        [
            ":material/stethoscope: Diagnóstico & Grad-CAM",
            ":material/compare: Comparação de Arquiteturas",
            ":material/analytics: Métricas & Relatórios MLOps",
        ]
    )

    with tab1:
        render_diagnostic_tab(
            config,
            selected_model_name,
            sample_path,
            run_prediction,
            get_last_conv_layer_name,
        )
    with tab2:
        render_comparison_tab(
            config, models_list, sample_path, run_prediction, get_last_conv_layer_name
        )
    with tab3:
        render_mlops_tab(config, models_list, load_trained_model)


if __name__ == "__main__":
    main()
