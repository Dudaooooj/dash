import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Dashboard Prescritivo de Retenção", layout="wide")

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

CAMINHO_BUDGET = "dados/df_budget_test.csv"
CAMINHO_SCORE = "dados/df_score_dashboard_test.csv"
CAMINHO_ERRO = "dados/df_erro_modelo.parquet"


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

        if "media_valor_6m" in df.columns:
            condicoes.append(df["media_valor_6m"] >= df["media_valor_6m"].quantile(0.75))
        else:
            condicoes.append(pd.Series(False, index=df.index))

        df["motivo_prescricao"] = np.select(
            condicoes,
            [
                "Alto risco e alto valor",
                "Atraso recorrente",
                "Cliente recente",
                "Cliente de alto valor"
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


def criar_persona_risco(df):
    df = df.copy()

    if "persona_risco" in df.columns:
        return df

    score_alto = df["prob_churn"] >= df["prob_churn"].quantile(0.80) if "prob_churn" in df.columns else pd.Series(False, index=df.index)
    valor_alto = df["valor_cliente_6m"] >= df["valor_cliente_6m"].quantile(0.75) if "valor_cliente_6m" in df.columns else pd.Series(False, index=df.index)
    atraso_alto = df["freq_atraso_6m"] >= 3 if "freq_atraso_6m" in df.columns else pd.Series(False, index=df.index)
    rel_curto = df["tempo_relacionamento_meses_corte"] <= 12 if "tempo_relacionamento_meses_corte" in df.columns else pd.Series(False, index=df.index)
    aceleracao = df["aceleracao_atraso"] > 0 if "aceleracao_atraso" in df.columns else pd.Series(False, index=df.index)
    sem_trafego = df["tem_trafego"] == 0 if "tem_trafego" in df.columns else pd.Series(False, index=df.index)

    condicoes = [
        rel_curto & atraso_alto,
        (~rel_curto) & aceleracao & score_alto,
        valor_alto & score_alto,
        (~valor_alto) & atraso_alto,
        sem_trafego & score_alto,
    ]

    escolhas = [
        "Novo com atraso precoce",
        "Cliente consolidado em deterioração",
        "Alto valor em risco",
        "Baixo valor com atraso recorrente",
        "Possível abandono de uso"
    ]

    df["persona_risco"] = np.select(condicoes, escolhas, default="Outros perfis")
    return df

def adicionar_colunas_validacao_teste(df):
    df = df.copy()

    if "target_churn_0a6m" in df.columns:
        df["target_churn_0a6m"] = (
            pd.to_numeric(df["target_churn_0a6m"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        df["resultado_teste"] = np.where(
            df["target_churn_0a6m"] == 1,
            "Target positivo",
            "Target negativo"
        )

        if "acionar" in df.columns:
            df["acerto_acionado"] = np.where(
                (df["acionar"] == 1) & (df["target_churn_0a6m"] == 1),
                1,
                0
            )

            df["grupo_validacao"] = np.select(
                [
                    (df["acionar"] == 1) & (df["target_churn_0a6m"] == 1),
                    (df["acionar"] == 1) & (df["target_churn_0a6m"] == 0),
                    (df["acionar"] == 0) & (df["target_churn_0a6m"] == 1),
                ],
                [
                    "Acerto priorizado",
                    "Priorizado sem target",
                    "Target não priorizado",
                ],
                default="Demais casos"
            )

    return df

def aplicar_filtros(df):
    st.sidebar.title("Filtros")

    df_f = df.copy()

    if coluna_existe(df_f, "estrategia"):
        opcoes = sorted(df_f["estrategia"].dropna().unique().tolist())
        sel = st.sidebar.multiselect("Estratégia", opcoes, default=opcoes)
        if sel:
            df_f = df_f[df_f["estrategia"].isin(sel)]

    if coluna_existe(df_f, "persona_risco"):
        opcoes = sorted(df_f["persona_risco"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Persona", opcoes, default=opcoes)
        if sel:
            df_f = df_f[df_f["persona_risco"].astype(str).isin(sel)]

    if coluna_existe(df_f, "faixa_risco"):
        opcoes = [x for x in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"] if x in df_f["faixa_risco"].astype(str).unique()]
        sel = st.sidebar.multiselect("Faixa de risco", opcoes, default=opcoes)
        if sel:
            df_f = df_f[df_f["faixa_risco"].astype(str).isin(sel)]

    if coluna_existe(df_f, "cidade"):
        opcoes = sorted(df_f["cidade"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Cidade", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["cidade"].astype(str).isin(sel)]

    if coluna_existe(df_f, "regiao"):
        opcoes = sorted(df_f["regiao"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Região", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["regiao"].astype(str).isin(sel)]

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

    if coluna_existe(df_f, "canal_sugerido"):
        opcoes = sorted(df_f["canal_sugerido"].dropna().astype(str).unique().tolist())
        sel = st.sidebar.multiselect("Canal sugerido", opcoes, default=[])
        if sel:
            df_f = df_f[df_f["canal_sugerido"].astype(str).isin(sel)]

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
# VISUAIS
# =========================================================
def render_kpis(df):
    clientes = len(df)
    custo_total = df["custo"].sum() if coluna_existe(df, "custo") else 0
    retorno_total = df["valor_esperado"].sum() if coluna_existe(df, "valor_esperado") else 0
    roi_total = retorno_total / custo_total if custo_total > 0 else 0
    ticket_medio = df["valor_cliente_6m"].mean() if coluna_existe(df, "valor_cliente_6m") else 0
    pct_acionados = (df["estrategia"] != "IGNORAR").mean() * 100 if "estrategia" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clientes", f"{clientes:,.0f}")
    c2.metric("Custo total", formatar_moeda(custo_total))
    c3.metric("Retorno esperado", formatar_moeda(retorno_total))
    c4.metric("ROI total", f"{roi_total:.2f}")
    c5.metric("% acionados", f"{pct_acionados:.1f}%")


def render_resumo_recomendacoes(df):
    st.markdown("### Leitura executiva")

    resumo = resumo_estrategia(df)

    estrat_top = "-"
    if not resumo.empty and "roi" in resumo.columns:
        resumo_valido = resumo.dropna(subset=["roi"])
        if not resumo_valido.empty:
            estrat_top = resumo_valido.sort_values("roi", ascending=False).iloc[0]["estrategia"]

    bairro_top = df["bairro"].astype(str).value_counts().idxmax() if "bairro" in df.columns and not df["bairro"].dropna().empty else "-"
    cidade_top = df["cidade"].astype(str).value_counts().idxmax() if "cidade" in df.columns and not df["cidade"].dropna().empty else "-"
    persona_top = df["persona_risco"].astype(str).value_counts().idxmax() if "persona_risco" in df.columns and not df["persona_risco"].dropna().empty else "-"
    canal_top = df["canal_sugerido"].astype(str).value_counts().idxmax() if "canal_sugerido" in df.columns and not df["canal_sugerido"].dropna().empty else "-"

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Estratégia com maior ROI:** {estrat_top}")
        st.write(f"**Cidade com maior concentração:** {cidade_top}")
        st.write(f"**Bairro com maior concentração:** {bairro_top}")
    with c2:
        st.write(f"**Persona dominante:** {persona_top}")
        st.write(f"**Canal sugerido dominante:** {canal_top}")


def render_aba_resumo(df):
    st.subheader("Resumo Executivo")
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)


def render_aba_personas(df):
    st.subheader("Personas e Perfis")

    if "persona_risco" not in df.columns:
        st.info("Coluna persona_risco não disponível.")
        return

    metricas = {}

    if "id_cliente_servico" in df.columns:
        metricas["clientes"] = ("id_cliente_servico", "count")

    if "valor_esperado" in df.columns:
        metricas["retorno_total"] = ("valor_esperado", "sum")

    if "roi_unitario_calc" in df.columns:
        metricas["roi_medio"] = ("roi_unitario_calc", "mean")
    elif "roi_unitario" in df.columns:
        metricas["roi_medio"] = ("roi_unitario", "mean")

    if "prob_churn" in df.columns:
        metricas["score_medio"] = ("prob_churn", "mean")

    if not metricas:
        st.warning("Nenhuma métrica disponível para resumir personas.")
        return

    persona = (
        df.groupby("persona_risco")
        .agg(**metricas)
        .reset_index()
    )

    if "retorno_total" in persona.columns:
        persona = persona.sort_values("retorno_total", ascending=False)
    elif "clientes" in persona.columns:
        persona = persona.sort_values("clientes", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        if "clientes" in persona.columns:
            fig = px.bar(
                persona,
                x="persona_risco",
                y="clientes",
                text="clientes",
                title="Clientes por Persona"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "retorno_total" in persona.columns:
            fig = px.bar(
                persona,
                x="persona_risco",
                y="retorno_total",
                text="retorno_total",
                title="Retorno Esperado por Persona"
            )
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        elif "score_medio" in persona.columns:
            fig = px.bar(
                persona,
                x="persona_risco",
                y="score_medio",
                text="score_medio",
                title="Score Médio por Persona"
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    if {"persona_risco", "estrategia", "id_cliente_servico"}.issubset(df.columns):
        heat = df.pivot_table(
            index="persona_risco",
            columns="estrategia",
            values="id_cliente_servico",
            aggfunc="count",
            fill_value=0
        )

        if not heat.empty:
            cols_presentes = [c for c in ORDEM_ESTRATEGIA if c in heat.columns]
            heat = heat[cols_presentes]

            fig = px.imshow(
                heat,
                text_auto=True,
                aspect="auto",
                title="Persona x Estratégia"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.dataframe(persona, use_container_width=True, hide_index=True)


def render_aba_geografia(df):
    st.subheader("Geografia e Carteira")

    col1, col2 = st.columns(2)

    with col1:
        if {"cidade", "valor_esperado"}.issubset(df.columns):
            top_cidades = (
                df.groupby("cidade")["valor_esperado"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_cidades.sort_values("valor_esperado"),
                x="valor_esperado",
                y="cidade",
                orientation="h",
                text="valor_esperado",
                title="Top 10 Cidades por Retorno Esperado"
            )
            fig.update_traces(texttemplate="R$ %{text:,.0f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        elif "cidade" in df.columns:
            top_cidades = (
                df["cidade"]
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )
            top_cidades.columns = ["cidade", "clientes"]

            fig = px.bar(
                top_cidades.sort_values("clientes"),
                x="clientes",
                y="cidade",
                orientation="h",
                text="clientes",
                title="Top 10 Cidades por Volume de Clientes"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Coluna cidade não disponível.")

    with col2:
        if {"bairro", "valor_esperado"}.issubset(df.columns):
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
            st.plotly_chart(fig, use_container_width=True)

        elif "bairro" in df.columns:
            top_bairros = (
                df["bairro"]
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )
            top_bairros.columns = ["bairro", "clientes"]

            fig = px.bar(
                top_bairros.sort_values("clientes"),
                x="clientes",
                y="bairro",
                orientation="h",
                text="clientes",
                title="Top 10 Bairros por Volume de Clientes"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Coluna bairro não disponível.")

    if {"regiao", "estrategia", "id_cliente_servico"}.issubset(df.columns):
        reg = df.pivot_table(
            index="regiao",
            columns="estrategia",
            values="id_cliente_servico",
            aggfunc="count",
            fill_value=0
        )

        if not reg.empty:
            cols_presentes = [c for c in ORDEM_ESTRATEGIA if c in reg.columns]
            reg = reg[cols_presentes]

            fig = px.imshow(
                reg,
                text_auto=True,
                aspect="auto",
                title="Região x Estratégia"
            )
            st.plotly_chart(fig, use_container_width=True)


def render_aba_playbook(df):
    st.subheader("Playbook de Ação")

    col1, col2 = st.columns(2)

    with col1:
        if {"persona_risco", "canal_sugerido", "id_cliente_servico"}.issubset(df.columns):
            canal = (
                df.groupby(["persona_risco", "canal_sugerido"])["id_cliente_servico"]
                .count()
                .reset_index(name="clientes")
            )
            fig = px.bar(
                canal,
                x="persona_risco",
                y="clientes",
                color="canal_sugerido",
                barmode="group",
                title="Canal Sugerido por Persona"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if {"persona_risco", "motivo_prescricao", "id_cliente_servico"}.issubset(df.columns):
            motivo = (
                df.groupby(["persona_risco", "motivo_prescricao"])["id_cliente_servico"]
                .count()
                .reset_index(name="clientes")
            )
            fig = px.bar(
                motivo,
                x="persona_risco",
                y="clientes",
                color="motivo_prescricao",
                barmode="group",
                title="Motivo da Prescrição por Persona"
            )
            st.plotly_chart(fig, use_container_width=True)

    cols = [c for c in [
        "prioridade_execucao",
        "persona_risco",
        "estrategia",
        "canal_sugerido",
        "motivo_prescricao",
        "valor_esperado",
        "roi_unitario_calc",
        "bairro",
        "cidade",
        "nome_plano"
    ] if c in df.columns]

    if cols:
        st.markdown("### Playbook resumido")
        base = (
            df[cols]
            .sort_values("valor_esperado", ascending=False)
            .head(30)
        )
        st.dataframe(base, use_container_width=True, hide_index=True)


def render_aba_operacao(df):
    st.subheader("Operação")

    if {"id_cliente_servico", "estrategia", "valor_esperado"}.issubset(df.columns):
        top_n = st.slider("Top clientes na tabela rápida", min_value=10, max_value=200, value=50, step=10)

        colunas = [
            c for c in [
                "prioridade_execucao",
                "id_cliente_servico",
                "persona_risco",
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
        st.dataframe(base, use_container_width=True, hide_index=True)

        csv = df[colunas].sort_values("valor_esperado", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar base operacional filtrada",
            data=csv,
            file_name="base_operacional_filtrada.csv",
            mime="text/csv"
        )

def render_aba_validacao_target(df_score):
    st.subheader("Validação do Target no Teste")

    if not {"target_churn_0a6m", "acionar"}.issubset(df_score.columns):
        st.info("Colunas de validação do teste não disponíveis.")
        return

    total = len(df_score)
    total_target = int(df_score["target_churn_0a6m"].sum())
    total_acionados = int(df_score["acionar"].sum())
    acertos = int(((df_score["acionar"] == 1) & (df_score["target_churn_0a6m"] == 1)).sum())

    precision_top = acertos / total_acionados if total_acionados > 0 else 0
    recall_top = acertos / total_target if total_target > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes no teste", f"{total:,.0f}")
    c2.metric("Target positivo", f"{total_target:,.0f}")
    c3.metric("Acionados", f"{total_acionados:,.0f}")
    c4.metric("Acertos no top", f"{acertos:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric("Precisão nos acionados", f"{precision_top:.1%}")
    c6.metric("Recall do top", f"{recall_top:.1%}")

    col1, col2 = st.columns(2)

    with col1:
        if "grupo_validacao" in df_score.columns:
            resumo = df_score["grupo_validacao"].value_counts().reset_index()
            resumo.columns = ["grupo_validacao", "clientes"]

            fig = px.bar(
                resumo,
                x="grupo_validacao",
                y="clientes",
                text="clientes",
                title="Clientes por Grupo de Validação"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if {"faixa_risco", "target_churn_0a6m"}.issubset(df_score.columns):
            risco = (
                df_score.groupby("faixa_risco")["target_churn_0a6m"]
                .mean()
                .reset_index(name="taxa_target")
            )

            fig = px.bar(
                risco,
                x="faixa_risco",
                y="taxa_target",
                text="taxa_target",
                title="Taxa Real do Target por Faixa de Risco"
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if {"cidade", "target_churn_0a6m"}.issubset(df_score.columns):
            cidade = (
                df_score[df_score["target_churn_0a6m"] == 1]
                .groupby("cidade")["id_cliente_servico"]
                .count()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="targets_positivos")
            )

            fig = px.bar(
                cidade,
                x="cidade",
                y="targets_positivos",
                text="targets_positivos",
                title="Top Cidades com Target Positivo"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if {"persona_risco", "target_churn_0a6m"}.issubset(df_score.columns):
            persona = (
                df_score.groupby("persona_risco")["target_churn_0a6m"]
                .mean()
                .reset_index(name="taxa_target")
                .sort_values("taxa_target", ascending=False)
            )

            fig = px.bar(
                persona,
                x="persona_risco",
                y="taxa_target",
                text="taxa_target",
                title="Taxa de Target por Persona"
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    cols = [c for c in [
        "id_cliente_servico",
        "prob_churn",
        "acionar",
        "target_churn_0a6m",
        "resultado_teste",
        "grupo_validacao",
        "faixa_risco",
        "cidade",
        "bairro",
        "nome_plano",
        "persona_risco",
    ] if c in df_score.columns]

    if cols:
        st.markdown("### Casos do teste")
        st.dataframe(
            df_score[cols]
            .sort_values(["target_churn_0a6m", "prob_churn"], ascending=[False, False])
            .head(100),
            use_container_width=True,
            hide_index=True
        )
# =========================================================
# APP
# =========================================================
st.title("📊 Dashboard Prescritivo de Retenção")
st.caption("Versão teste reorganizada para responder melhor às perguntas de persona, território e ação.")

try:
    df_budget = carregar_dados(CAMINHO_BUDGET)
    df_score = carregar_dados(CAMINHO_SCORE)
    df_erro = carregar_dados(CAMINHO_ERRO)
except Exception as e:
    st.error(f"Erro ao carregar os arquivos: {e}")
    st.stop()

df_budget = criar_persona_risco(df_budget)
df_score = criar_persona_risco(df_score)
df_budget = adicionar_colunas_validacao_teste(df_score)

df_filtrado = aplicar_filtros(df_budget)
df_simulado = aplicar_simulacao_budget(df_filtrado)

if df_simulado.empty:
    st.warning("Nenhum registro encontrado com os filtros/simulação atuais.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Resumo Executivo",
    "Personas e Perfis",
    "Geografia e Carteira",
    "Playbook de Ação",
    "Operação",
    "Validação do Target no Teste",
    "Qualidade do Modelo"

])

with tab1:
    render_aba_resumo(df_simulado)

with tab2:
    render_aba_personas(df_simulado)

with tab3:
    render_aba_geografia(df_simulado)

with tab4:
    render_aba_playbook(df_simulado)

with tab5:
    render_aba_operacao(df_simulado)
with tab6:
    render_aba_validacao_target(df_score)