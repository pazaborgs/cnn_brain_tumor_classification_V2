import altair as alt
import pandas as pd
import numpy as np

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

CLASS_LABELS_PT = {
    "glioma_tumor": "Glioma",
    "meningioma_tumor": "Meningioma",
    "no_tumor": "Sem Tumor",
    "pituitary_tumor": "Tumor de Hipófise",
}

CLASS_COLORS = {
    "glioma_tumor": "#3b82f6",  # Blue
    "meningioma_tumor": "#8b5cf6",  # Purple
    "no_tumor": "#10b981",  # Emerald (Saudável)
    "pituitary_tumor": "#f59e0b",  # Amber
}

def probability_chart(
    preds: np.ndarray, highlight_class: str | None = None, height: int = 180
) -> alt.Chart:
    """Gera o gráfico de probabilidades por classe."""
    df = pd.DataFrame(
        {
            "classe": [CLASS_LABELS_PT[c] for c in CLASS_NAMES],
            "classe_raw": CLASS_NAMES,
            "probabilidade": preds,
        }
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=26)
        .encode(
            x=alt.X(
                "probabilidade:Q",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
                title="Probabilidade (%)",
            ),
            y=alt.Y("classe:N", sort="-x", title=None),
            color=alt.Color(
                "classe_raw:N",
                scale=alt.Scale(
                    domain=CLASS_NAMES, range=[CLASS_COLORS[c] for c in CLASS_NAMES]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("classe:N", title="Classe"),
                alt.Tooltip("probabilidade:Q", format=".2%", title="Probabilidade"),
            ],
            opacity=alt.condition(
                alt.datum.classe_raw == highlight_class, alt.value(1.0), alt.value(0.55)
            )
            if highlight_class
            else alt.value(0.9),
        )
        .properties(
            height=height, padding={"left": 5, "right": 15, "top": 10, "bottom": 5}
        )
        .configure_view(strokeWidth=0, fill="#FFFFFF")
        .configure_axis(labelColor="#0F172A", titleColor="#0F172A", gridColor="#E2E8F0")
    )
    return chart
