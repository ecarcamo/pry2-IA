import altair as alt
import pandas as pd


_ALGO_COLORS = {
    "BFS":    "#00B0FF",
    "DFS":    "#FF6D00",
    "Greedy": "#00E676",
    "A*":     "#FF4081",
}


def _color_scale():
    return alt.Scale(
        domain=list(_ALGO_COLORS.keys()),
        range=list(_ALGO_COLORS.values()),
    )


def bar_chart(df: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Algorithm:N", sort=list(_ALGO_COLORS.keys()), axis=alt.Axis(labelColor="#E8E8F0", titleColor="#E8E8F0")),
            y=alt.Y(f"{metric}:Q", axis=alt.Axis(labelColor="#E8E8F0", titleColor="#E8E8F0")),
            color=alt.Color("Algorithm:N", scale=_color_scale(), legend=None),
            tooltip=["Algorithm:N", alt.Tooltip(f"{metric}:Q", format=".5f" if "Runtime" in metric else "d")],
        )
        .properties(
            title=alt.TitleParams(text=title, color="#E8E8F0", fontSize=13),
            height=260,
        )
        .configure_view(strokeOpacity=0)
        .configure(
            background="#1A1A3E",
            axis=alt.AxisConfig(grid=True, gridColor="#2A2A4A", domainColor="#444"),
        )
    )
    return chart


def multi_bar_chart(df: pd.DataFrame) -> alt.Chart:
    """Gráfico de barras agrupadas para las 3 métricas principales."""
    metrics = ["Path Length", "Nodes Explored"]
    melted = df.melt(
        id_vars="Algorithm",
        value_vars=metrics,
        var_name="Métrica",
        value_name="Valor",
    )

    chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Algorithm:N", axis=alt.Axis(labelColor="#E8E8F0", title="", labelAngle=0)),
            y=alt.Y("Valor:Q", axis=alt.Axis(labelColor="#E8E8F0", titleColor="#E8E8F0")),
            color=alt.Color("Algorithm:N", scale=_color_scale(), legend=None),
            column=alt.Column("Métrica:N", title="", header=alt.Header(labelColor="#E8E8F0", labelFontSize=12)),
            tooltip=["Algorithm:N", "Métrica:N", "Valor:Q"],
        )
        .properties(height=240, width=180)
        .configure_view(stroke="transparent")
        .configure(
            background="#1A1A3E",
            axis=alt.AxisConfig(grid=True, gridColor="#2A2A4A", domainColor="#444"),
        )
    )
    return chart


def runtime_bar_chart(df: pd.DataFrame) -> alt.Chart:
    return bar_chart(df, "Runtime (s)", "⏱ Tiempo de ejecución (s)")


def explored_bar_chart(df: pd.DataFrame) -> alt.Chart:
    return bar_chart(df, "Nodes Explored", "🔍 Nodos explorados")


def path_bar_chart(df: pd.DataFrame) -> alt.Chart:
    return bar_chart(df, "Path Length", "📏 Longitud del camino")
