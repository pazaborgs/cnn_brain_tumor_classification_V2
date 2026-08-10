import streamlit as st


def inject_css_from_file(css_file_path: str = "src/ui/styles.css"):
    """Lê o arquivo CSS externo e o injeta na página do Streamlit.

    Args:
        css_file_path (str): Caminho absoluto ou relativo do arquivo CSS.
    """
    try:
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Arquivo CSS não encontrado em {css_file_path}")


def window(title: str, icon: str = "folder", height: int = None):
    """Cria um container estilizado no formato de janela com cabeçalho limpo.

    Args:
        title (str): Título da janela a ser exibido.
        icon (str): Nome do ícone do Google Material Symbols.
        height (int, opcional): Altura fixa em pixels.

    Returns:
        st.container: O contexto do container do Streamlit.
    """
    kwargs = {"border": True}
    if height is not None:
        kwargs["height"] = height
    container = st.container(**kwargs)
    with container:
        st.markdown(
            f"""
            <div class="os-titlebar">
                <div class="os-titlebar-left">
                    <span class="material-symbols-rounded" style="font-size:18px; margin-right:6px;">{icon}</span>
                    <span>{title}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return container


def groupbox_label(text: str):
    """Renderiza o label estilizado do groupbox na barra lateral (Sidebar).

    Args:
        text (str): Texto de título do groupbox.
    """
    st.markdown(
        f'<div class="sidebar-groupbox"><div class="gb-label">{text}</div>',
        unsafe_allow_html=True,
    )


def groupbox_end():
    """Renderiza o fechamento da tag HTML do groupbox na barra lateral (Sidebar)."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_app_titlebar():
    """Renderiza o cabeçalho principal do App."""
    st.markdown(
        """
        <div class="app-titlebar">
            <div class="app-titlebar-left">
                <span class="material-symbols-rounded" style="font-size: 2.2rem; color: #6EE7B7;">neurology</span>
                <div class="txt">
                    <div class="name">brain-tumor-classifier.app</div>
                    <div class="sub">Classificação MRI · Avaliação Multi-Modelo & Grad-CAM</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
