import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.markdown(
    """
    <style>
    .block-container {
        max-width: 98%;
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# CONFIG
# =========================================================

MAPA_COR = {
    "AÇÃO BARATA": "#F4A261",
    "AÇÃO MÉDIA": "#6EA8BD",
    "AÇÃO FORTE": "#1D4E69",
    "IGNORAR": "#D9D9D9"
}

ORDEM_ESTRATEGIA = ["AÇÃO BARATA", "AÇÃO MÉDIA", "AÇÃO FORTE", "IGNORAR"]


# =========================================================
# LOAD
# =========================================================
@st.cache_data
def carregar_dados(caminho: str):
    if caminho.endswith(".parquet"):
        df = pd.read_parquet(caminho)
    else:
        df = pd.read_csv(caminho)

    if "prob_churn" in df.columns:
        df["prob_churn"] = pd.to_numeric(df["prob_churn"], errors="coerce")
        df["faixa_risco"] = pd.cut(
            df["prob_churn"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
            include_lowest=True
        )

    if {"valor_esperado", "custo"}.issubset(df.columns):
        df["roi_unitario_calc"] = np.where(
            df["custo"] > 0,
            df["valor_esperado"] / df["custo"],
            np.nan
        )

    if "valor_cliente_6m" not in df.columns:
        if "media_valor_6m" in df.columns:
            df["valor_cliente_6m"] = df["media_valor_6m"] * 6
        elif "valor_ultima_fatura" in df.columns:
            df["valor_cliente_6m"] = df["valor_ultima_fatura"] * 6

    if "prioridade_execucao" not in df.columns and "valor_esperado" in df.columns:
        df = df.sort_values("valor_esperado", ascending=False).copy()
        df["prioridade_execucao"] = np.arange(1, len(df) + 1)

    if "motivo_prescricao" not in df.columns:
        condicoes = []

        if {"prob_churn", "valor_esperado"}.issubset(df.columns):
            condicoes.append(
                (df["prob_churn"] >= 0.8) &
                (df["valor_esperado"] >= df["valor_esperado"].median())
            )
        else:
            condicoes.append(pd.Series(False, index=df.index))

        if "freq_atraso_6m" in df.columns:
            condicoes.append(df["freq_atraso_6m"] >= 3)
        else:
            condicoes.append(pd.Series(False, index=df.index))

        if "tempo_relacionamento_meses_corte" in df.columns:
            condicoes.append(df["tempo_relacionamento_meses_corte"] <= 12)
        else:
            condicoes.append(pd.Series(False, index=df.index))

        df["motivo_prescricao"] = np.select(
            condicoes,
            [
                "Alto risco e alto valor",
                "Atraso recorrente",
                "Cliente recente"
            ],
            default="Prioridade econômica"
        )

    if "canal_sugerido" not in df.columns and "estrategia" in df.columns:
        df["canal_sugerido"] = np.select(
            [
                df["estrategia"] == "AÇÃO FORTE",
                df["estrategia"] == "AÇÃO MÉDIA",
                df["estrategia"] == "AÇÃO BARATA",
            ],
            [
                "Atendimento ativo / retenção",
                "Contato direcionado",
                "Canal massivo"
            ],
            default="Sem ação"
        )

    return df

# =========================================================
# HELPERS
# =========================================================
def coluna_existe(df, col):
    return col in df.columns

def formatar_moeda(v):
    return f"R$ {v:,.0f}"

def resumo_estrategia(df):
    if not {"estrategia", "id_cliente_servico", "custo", "valor_esperado"}.issubset(df.columns):
        return pd.DataFrame()

    resumo = df.groupby("estrategia").agg(
        clientes=("id_cliente_servico", "count"),
        custo_total=("custo", "sum"),
        retorno_total=("valor_esperado", "sum")
    ).reset_index()

    resumo["roi"] = np.where(
        resumo["custo_total"] > 0,
        resumo["retorno_total"] / resumo["custo_total"],
        np.nan
    )

    resumo["estrategia"] = pd.Categorical(
        resumo["estrategia"],
        categories=ORDEM_ESTRATEGIA,
        ordered=True
    )
    resumo = resumo.sort_values("estrategia")
    resumo = resumo.dropna(subset=["estrategia"])
    return resumo


def aplicar_filtros(df):
    st.sidebar.title("Filtros")

    st.sidebar.markdown("### Carteira")


    st.sidebar.markdown("### Perfil")
 

    st.sidebar.markdown("### Simulação")


    df_f = df.copy()

    if coluna_existe(df_f, "estrategia"):
        opcoes = sorted(df_f["estrategia"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Estratégia", opcoes, default=opcoes)
        if sel:
            df_f = df_f[df_f["estrategia"].isin(sel)]

    if coluna_existe(df_f, "faixa_risco"):
        opcoes = [x for x in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"] if x in df_f["faixa_risco"].astype(str).unique()]
        sel = st.sidebar.multiselect("Faixa de risco", opcoes, default=opcoes)
        if sel:
            df_f = df_f[df_f["faixa_risco"].astype(str).isin(sel)]

    if coluna_existe(df_f, "bairro"):
        opcoes = sorted(df_f["bairro"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Bairro", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["bairro"].astype(str).isin(sel)]

    if coluna_existe(df_f, "nome_plano"):
        opcoes = sorted(df_f["nome_plano"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Plano", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["nome_plano"].astype(str).isin(sel)]

    if coluna_existe(df_f, "fase_cliente"):
        opcoes = sorted(df_f["fase_cliente"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Fase do cliente", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["fase_cliente"].astype(str).isin(sel)]

    if coluna_existe(df_f, "prob_churn"):
        intervalo = st.sidebar.slider(
            "Probabilidade de churn",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01
        )
        df_f = df_f[df_f["prob_churn"].between(intervalo[0], intervalo[1])]

    if coluna_existe(df_f, "valor_esperado"):
        minimo = st.sidebar.number_input(
            "Valor esperado mínimo",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
        df_f = df_f[df_f["valor_esperado"] >= minimo]

    return df_f


def aplicar_simulacao_budget(df):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulação de orçamento")

    usar_simulacao = st.sidebar.checkbox("Ativar simulação", value=False)


    if not usar_simulacao:
        return df

    if not {"estrategia", "custo", "valor_esperado"}.issubset(df.columns):
        return df

    budget_forte = st.sidebar.number_input("Budget AÇÃO FORTE", min_value=0.0, value=50000.0, step=5000.0)
    budget_media = st.sidebar.number_input("Budget AÇÃO MÉDIA", min_value=0.0, value=70000.0, step=5000.0)
    budget_barata = st.sidebar.number_input("Budget AÇÃO BARATA", min_value=0.0, value=80000.0, step=5000.0)

    budgets = {
        "AÇÃO FORTE": budget_forte,
        "AÇÃO MÉDIA": budget_media,
        "AÇÃO BARATA": budget_barata
    }

    partes = []
    for estrategia, budget in budgets.items():
        parte = df[df["estrategia"] == estrategia].copy()
        if parte.empty:
            continue
        parte = parte.sort_values("valor_esperado", ascending=False).copy()
        parte["custo_acum_simulado"] = parte["custo"].cumsum()
        parte = parte[parte["custo_acum_simulado"] <= budget]
        partes.append(parte)

    ignorar = df[df["estrategia"] == "IGNORAR"].copy()
    if not ignorar.empty:
        partes.append(ignorar)

    if partes:
        return pd.concat(partes, axis=0).copy()

    return df.iloc[0:0].copy()


# =========================================================
# VISUALS
# =========================================================
def render_kpis(df):
    clientes = len(df)
    custo_total = df["custo"].sum() if coluna_existe(df, "custo") else 0
    retorno_total = df["valor_esperado"].sum() if coluna_existe(df, "valor_esperado") else 0
    roi_total = retorno_total / custo_total if custo_total > 0 else 0
    ticket_medio = df["valor_cliente_6m"].mean() if coluna_existe(df, "valor_cliente_6m") else 0

    top_10_pct = max(int(len(df) * 0.10), 1)
    retorno_top10 = (
        df.sort_values("valor_esperado", ascending=False)["valor_esperado"].head(top_10_pct).sum()
        if coluna_existe(df, "valor_esperado") else 0
    )
    perc_top10 = (retorno_top10 / retorno_total * 100) if retorno_total > 0 else 0

    if "estrategia" in df.columns:
        pct_acionados = (df["estrategia"] != "IGNORAR").mean() * 100
    else:
        pct_acionados = 0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Clientes", f"{clientes:,.0f}")
    c2.metric("Custo total", formatar_moeda(custo_total))
    c3.metric("Retorno esperado", formatar_moeda(retorno_total))
    c4.metric("ROI total", f"{roi_total:.2f}")
    c5.metric("Ticket médio 6m", formatar_moeda(ticket_medio))
    c6.metric("Retorno no top 10%", f"{perc_top10:.1f}%")
    c7.metric("% acionados", f"{pct_acionados:.1f}%")

def render_funil_prescritivo(df):
    if not {"estrategia", "prob_churn"}.issubset(df.columns):
        st.info("Funil prescritivo indisponível.")
        return

    total = len(df)
    acionados = (df["estrategia"] != "IGNORAR").sum()
    alto_risco = (df["prob_churn"] >= 0.8).sum()
    acao_forte = (df["estrategia"] == "AÇÃO FORTE").sum()

    funil = pd.DataFrame({
        "etapa": ["Base filtrada", "Acionados", "Alto risco", "Ação forte"],
        "clientes": [total, acionados, alto_risco, acao_forte]
    })

    fig = px.funnel(
        funil,
        x="clientes",
        y="etapa",
        title="Funil Prescritivo"
    )
    st.plotly_chart(fig, width="stretch")

def render_heatmap_risco_estrategia(df):
    if not {"faixa_risco", "estrategia", "id_cliente_servico"}.issubset(df.columns):
        st.info("Heatmap risco x estratégia indisponível.")
        return

    heat = df.pivot_table(
        index="faixa_risco",
        columns="estrategia",
        values="id_cliente_servico",
        aggfunc="count",
        fill_value=0
    )

    heat = heat.reindex(
        index=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        fill_value=0
    )

    cols_presentes = [c for c in ORDEM_ESTRATEGIA if c in heat.columns]
    heat = heat[cols_presentes]

    fig = px.imshow(
        heat,
        text_auto=True,
        aspect="auto",
        title="Distribuição de Clientes por Faixa de Risco e Estratégia"
    )
    st.plotly_chart(fig, width="stretch")

def render_pareto_retorno(df):
    if "valor_esperado" not in df.columns or df["valor_esperado"].sum() <= 0:
        st.info("Pareto de retorno indisponível.")
        return

    pareto = df.sort_values("valor_esperado", ascending=False).copy()
    pareto["ordem"] = np.arange(1, len(pareto) + 1)
    pareto["retorno_acum"] = pareto["valor_esperado"].cumsum()
    pareto["retorno_acum_pct"] = pareto["retorno_acum"] / pareto["valor_esperado"].sum() * 100
    pareto["clientes_pct"] = pareto["ordem"] / len(pareto) * 100

    fig = px.line(
        pareto,
        x="clientes_pct",
        y="retorno_acum_pct",
        title="Concentração do Retorno Esperado (Pareto)"
    )
    fig.update_layout(
        xaxis_title="% dos clientes priorizados",
        yaxis_title="% do retorno acumulado"
    )
    fig.add_hline(y=80, line_dash="dash")
    st.plotly_chart(fig, width="stretch")

def render_scatter_quadrantes(df):
    if not {"prob_churn", "valor_esperado", "estrategia"}.issubset(df.columns):
        st.info("Mapa de quadrantes indisponível.")
        return

    mediana_valor = df["valor_esperado"].median()
    corte_risco = 0.6

    hover = [c for c in [
        "id_cliente_servico",
        "bairro",
        "nome_plano",
        "fase_cliente",
        "motivo_prescricao"
    ] if c in df.columns]

    size_col = "valor_cliente_6m" if "valor_cliente_6m" in df.columns else None

    fig = px.scatter(
        df,
        x="prob_churn",
        y="valor_esperado",
        color="estrategia",
        size=size_col,
        hover_data=hover,
        color_discrete_map=MAPA_COR,
        title="Quadrantes de Priorização"
    )

    fig.add_vline(x=corte_risco, line_dash="dash", line_color="gray")
    fig.add_hline(y=mediana_valor, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Leitura sugerida: canto superior direito = maior prioridade "
        "(alto risco e alto valor esperado)."
    )

def render_resumo_recomendacoes(df):
    st.markdown("### Recomendações automáticas")

    resumo = resumo_estrategia(df)

    estrat_top = "-"
    if not resumo.empty and "roi" in resumo.columns:
        resumo_valido = resumo.dropna(subset=["roi"])
        if not resumo_valido.empty:
            estrat_top = resumo_valido.sort_values("roi", ascending=False).iloc[0]["estrategia"]

    bairro_top = "-"
    if "bairro" in df.columns and not df["bairro"].dropna().empty:
        bairro_top = df["bairro"].astype(str).value_counts().idxmax()

    plano_top = "-"
    if "nome_plano" in df.columns and not df["nome_plano"].dropna().empty:
        plano_top = df["nome_plano"].astype(str).value_counts().idxmax()

    motivo_top = "-"
    if "motivo_prescricao" in df.columns and not df["motivo_prescricao"].dropna().empty:
        motivo_top = df["motivo_prescricao"].astype(str).value_counts().idxmax()

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Estratégia com maior ROI:** {estrat_top}")
        st.write(f"**Bairro com maior concentração:** {bairro_top}")
    with c2:
        st.write(f"**Plano mais recorrente:** {plano_top}")
        st.write(f"**Motivo prescritivo dominante:** {motivo_top}")
def render_aba_executiva(df):
    st.subheader("Visão Executiva")
    render_kpis(df)
    render_resumo_recomendacoes(df)

    resumo = resumo_estrategia(df)
    if resumo.empty:
        st.warning("Resumo por estratégia indisponível.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            resumo,
            x="estrategia",
            y="clientes",
            color="estrategia",
            color_discrete_map=MAPA_COR,
            text="clientes",
            title="Clientes por Estratégia"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            resumo,
            x="estrategia",
            y="retorno_total",
            color="estrategia",
            color_discrete_map=MAPA_COR,
            text="retorno_total",
            title="Retorno Esperado por Estratégia"
        )
        fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure()
        fig.add_bar(x=resumo["estrategia"], y=resumo["custo_total"], name="Custo", marker_color="#6EA8BD")
        fig.add_bar(x=resumo["estrategia"], y=resumo["retorno_total"], name="Retorno", marker_color="#F4A261")
        fig.update_layout(barmode="group", title="Custo vs Retorno")
        st.plotly_chart(fig, width="stretch")

    with col4:
        fig = px.bar(
            resumo,
            x="estrategia",
            y="roi",
            color="estrategia",
            color_discrete_map=MAPA_COR,
            text="roi",
            title="ROI por Estratégia"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    col5, col6 = st.columns(2)

    with col5:
        render_funil_prescritivo(df)

    with col6:
        render_heatmap_risco_estrategia(df)


def render_aba_analitica(df):
    st.subheader("Visão Analítica")

    col1, col2 = st.columns(2)

    with col1:
        render_scatter_quadrantes(df)

    with col2:
        if coluna_existe(df, "prob_churn"):
            fig = px.histogram(
                df,
                x="prob_churn",
                nbins=20,
                color="estrategia" if coluna_existe(df, "estrategia") else None,
                color_discrete_map=MAPA_COR,
                title="Distribuição do Score de Churn"
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna prob_churn não disponível.")

    col3, col4 = st.columns(2)

    with col3:
        if coluna_existe(df, "bairro"):
            top_bairros = (
                df.groupby("bairro")["valor_esperado"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_bairros.sort_values("valor_esperado"),
                x="valor_esperado",
                y="bairro",
                orientation="h",
                text="valor_esperado",
                title="Top 10 Bairros por Retorno Esperado"
            )
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna bairro não disponível.")

    with col4:
        if coluna_existe(df, "nome_plano"):
            top_planos = (
                df.groupby("nome_plano")["valor_esperado"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_planos.sort_values("valor_esperado"),
                x="valor_esperado",
                y="nome_plano",
                orientation="h",
                text="valor_esperado",
                title="Top 10 Planos por Retorno Esperado"
            )
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna nome_plano não disponível.")

    col5, col6 = st.columns(2)

    with col5:
        render_pareto_retorno(df)

    with col6:
        if coluna_existe(df, "fase_cliente"):
            fase = df["fase_cliente"].astype(str).value_counts().reset_index()
            fase.columns = ["fase_cliente", "clientes"]
            fig = px.bar(
                fase,
                x="fase_cliente",
                y="clientes",
                text="clientes",
                title="Fase do Cliente"
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Coluna fase_cliente não disponível.")


def render_aba_operacional(df):
    st.subheader("Visão Operacional")

    if {"id_cliente_servico", "estrategia", "valor_esperado"}.issubset(df.columns):
        top_n = st.slider("Top clientes na tabela rápida", min_value=10, max_value=200, value=50, step=10)

        colunas = [
            c for c in [
                "prioridade_execucao",
                "id_cliente_servico",
                "estrategia",
                "motivo_prescricao",
                "canal_sugerido",
                "prob_churn",
                "valor_esperado",
                "custo",
                "roi_unitario",
                "roi_unitario_calc",
                "valor_cliente_6m",
                "bairro",
                "cidade",
                "regiao",
                "nome_plano",
                "fase_cliente",
                "tempo_relacionamento_meses_corte",
                "freq_atraso_6m",
                "media_atraso_historico_total",
                "dias_desde_ultimo_atraso"
            ] if c in df.columns
        ]

        base = df[colunas].sort_values("valor_esperado", ascending=False).head(top_n)
        st.dataframe(base, width="stretch", hide_index=True)

        csv = df[colunas].sort_values("valor_esperado", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar base operacional filtrada",
            data=csv,
            file_name="base_operacional_filtrada.csv",
            mime="text/csv"
        )
    else:
        st.info("Colunas operacionais mínimas não disponíveis.")

    col1, col2 = st.columns(2)

    with col1:
        if {"estrategia", "valor_esperado"}.issubset(df.columns):
            box = px.box(
                df,
                x="estrategia",
                y="valor_esperado",
                color="estrategia",
                color_discrete_map=MAPA_COR,
                title="Distribuição do Valor Esperado por Estratégia"
            )
            box.update_layout(showlegend=False)
            st.plotly_chart(box, width="stretch")

    with col2:
        if "motivo_prescricao" in df.columns:
            motivos = df["motivo_prescricao"].value_counts().reset_index()
            motivos.columns = ["motivo_prescricao", "clientes"]
            fig = px.bar(
                motivos,
                x="motivo_prescricao",
                y="clientes",
                text="clientes",
                title="Motivos de Prescrição"
            )
            st.plotly_chart(fig, width="stretch")


def render_inconsistencia(df_score, df_erro):
    st.subheader("Inconsistência do Modelo")

    total = len(df_score)
    erros = len(df_erro)

    st.metric("Clientes inconsistentes", f"{erros} ({erros/total:.1%})")

    fig = px.histogram(
        df_score,
        x="prob_churn",
        color=df_score["prob_churn"].isin(df_erro["prob_churn"]),
        title="Distribuição de risco vs inconsistência"
    )

    st.plotly_chart(fig)

def render_aba_qualidade_modelo(df_score, df_erro):
    st.subheader("Qualidade do Modelo")

    total = len(df_score)
    inconsistentes = len(df_erro)
    pct = inconsistentes / total * 100 if total > 0 else 0

    c1, c2 = st.columns(2)
    c1.metric("Clientes avaliados", f"{total:,.0f}")
    c2.metric("Inconsistências", f"{inconsistentes:,.0f} ({pct:.1f}%)")

    if "prob_churn" in df_score.columns:
        fig = px.histogram(
            df_score,
            x="prob_churn",
            nbins=20,
            title="Distribuição do score de churn"
        )
        st.plotly_chart(fig, width="stretch")

    if not df_erro.empty and "prob_churn" in df_erro.columns:
        fig2 = px.histogram(
            df_erro,
            x="prob_churn",
            nbins=20,
            title="Distribuição dos casos inconsistentes"
        )
        st.plotly_chart(fig2, width="stretch")

    cols = [c for c in [
        "id_cliente_servico",
        "prob_churn",
        "id_motivo_cancelamento",
        "bairro",
        "cidade",
        "regiao",
        "nome_plano",
        "cluster_risco"
    ] if c in df_erro.columns]

    if cols:
        st.dataframe(df_erro[cols].head(100), width="stretch", hide_index=True)

# =========================================================
# APP
# =========================================================
CAMINHO_BUDGET = "dados/df_budget.parquet"
CAMINHO_SCORE = "dados/df_score_dashboard.parquet"
CAMINHO_ERRO = "dados/df_erro_modelo.parquet"

st.title("📊 Dashboard Prescritivo de Retenção")
st.caption("Versão profissional em Streamlit — leitura dos arquivos Parquet gerados pelo notebook.")

try:
    df_budget = carregar_dados(CAMINHO_BUDGET)
    df_score = carregar_dados(CAMINHO_SCORE)
    df_erro = carregar_dados(CAMINHO_ERRO)
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()
df_filtrado = aplicar_filtros(df_budget)
df_simulado = aplicar_simulacao_budget(df_filtrado)

if df_simulado.empty:
    st.warning("Nenhum registro encontrado com os filtros/simulação atuais.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Visão Executiva",
    "Visão Analítica",
    "Visão Operacional",
    "Qualidade do Modelo"
])

with tab1:
    render_aba_executiva(df_simulado)

with tab2:
    render_aba_analitica(df_simulado)

with tab3:
    render_aba_operacional(df_simulado)

with tab4:
    render_aba_qualidade_modelo(df_score, df_erro)
