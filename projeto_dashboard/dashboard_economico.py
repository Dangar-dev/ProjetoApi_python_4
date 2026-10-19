
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import date
from pathlib import Path


# =========================================================
# CONFIGURAÇÕES
# =========================================================

BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
IPEA_URL = "https://www.ipeadata.gov.br/api/odata4"

PASTA_SAIDA = Path("resultados")
PASTA_SAIDA.mkdir(exist_ok=True)

# Séries de exemplo. Os códigos do IPEA podem ser
# substituídos por outros códigos existentes no catálogo.
INDICADORES = {
    "1": {
        "nome": "Selic meta",
        "fonte": "BCB",
        "codigo": "432",
        "frequencia": "diaria",
    },
    "2": {
        "nome": "IPCA mensal",
        "fonte": "BCB",
        "codigo": "433",
        "frequencia": "mensal",
    },
    "3": {
        "nome": "Dolar comercial - venda",
        "fonte": "BCB",
        "codigo": "1",
        "frequencia": "diaria",
    },
    "4": {
        "nome": "Selic acumulada no mes",
        "fonte": "IPEA",
        "codigo": "BM12_TJOVER12",
        "frequencia": "mensal",
    },
    "5": {
        "nome": "IPCA mensal - IPEA",
        "fonte": "IPEA",
        "codigo": "PRECOS12_IPCAG12",
        "frequencia": "mensal",
    },
    "6": {
        "nome": "IGP-M mensal - IPEA",
        "fonte": "IPEA",
        "codigo": "IGP12_IGPMG12",
        "frequencia": "mensal",
    },
}


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def converter_data(data_texto):
    return pd.Timestamp(data_texto).normalize()


def requisicao_json(url, params=None):
    resposta = requests.get(
        url,
        params=params,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    resposta.raise_for_status()

    try:
        return resposta.json()
    except ValueError as erro:
        raise ValueError(
            "A API nao retornou JSON valido. "
            "Verifique o endpoint e o codigo da serie."
        ) from erro


# =========================================================
# BANCO CENTRAL - SGS
# =========================================================

def buscar_bcb(codigo, inicio, fim):
    url = f"{BCB_URL}.{codigo}/dados"

    params = {
        "formato": "json",
        "dataInicial": inicio.strftime("%d/%m/%Y"),
        "dataFinal": fim.strftime("%d/%m/%Y"),
    }

    dados = requisicao_json(url, params)

    if not isinstance(dados, list) or not dados:
        raise ValueError(
            f"BCB nao retornou dados para a serie {codigo}."
        )

    df = pd.DataFrame(dados)

    df["data"] = pd.to_datetime(
        df["data"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    df["valor"] = pd.to_numeric(
        df["valor"],
        errors="coerce",
    )

    df = df.dropna(subset=["data", "valor"])

    return df[["data", "valor"]].sort_values("data")


# =========================================================
# IPEA - ODATA
# =========================================================

def buscar_ipea(codigo, inicio, fim):
    # Endpoint OData de valores de uma serie.
    url = (
        f"{IPEA_URL}/ValoresSerie"
        f"(SERCODIGO='{codigo}')"
    )

    registros = []

    while url:
        dados = requisicao_json(url)

        if isinstance(dados, dict):
            registros.extend(dados.get("value", []))
            url = dados.get("@odata.nextLink")
        elif isinstance(dados, list):
            registros.extend(dados)
            url = None
        else:
            raise ValueError(
                "Formato inesperado na resposta do IPEA."
            )

    if not registros:
        raise ValueError(
            f"IPEA nao retornou dados para {codigo}. "
            "Confira o codigo da serie."
        )

    df = pd.DataFrame(registros)

    colunas = {"VALDATA", "VALVALOR"}
    if not colunas.issubset(df.columns):
        raise ValueError(
            "A resposta do IPEA nao possui VALDATA e VALVALOR. "
            f"Colunas recebidas: {list(df.columns)}"
        )

    # Datas OData normalmente chegam em formato ISO.
    df["data"] = pd.to_datetime(
        df["VALDATA"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    df["valor"] = pd.to_numeric(
        df["VALVALOR"],
        errors="coerce",
    )

    df = df.dropna(subset=["data", "valor"])

    df = df[
        (df["data"] >= pd.Timestamp(inicio))
        & (df["data"] <= pd.Timestamp(fim))
    ]

    return df[["data", "valor"]].sort_values("data")


# =========================================================
# BUSCA UNIFICADA
# =========================================================

def buscar_indicador(chave, inicio, fim):
    indicador = INDICADORES[chave]

    print(
        f"\nConsultando {indicador['nome']} "
        f"[{indicador['fonte']}]..."
    )

    if indicador["fonte"] == "BCB":
        df = buscar_bcb(
            indicador["codigo"], inicio, fim
        )
    else:
        df = buscar_ipea(
            indicador["codigo"], inicio, fim
        )

    if df.empty:
        raise ValueError(
            "A serie nao possui dados no periodo selecionado."
        )

    df = df.copy()
    df["indicador"] = indicador["nome"]
    df["fonte"] = indicador["fonte"]

    print(f"Registros encontrados: {len(df)}")

    return df


# =========================================================
# CONVERSAO PARA FREQUENCIA MENSAL
# =========================================================

def converter_para_mensal(df):
    df = df.copy()

    # Para series diarias, escolhe a ultima observacao
    # disponivel no mes. Para series mensais, preserva
    # o unico valor do mes quando ha uma observacao.
    df["mes"] = df["data"].dt.to_period("M")

    mensal = (
        df.sort_values("data")
        .groupby(
            ["indicador", "fonte", "mes"],
            as_index=False,
        )
        .last()
    )

    mensal["data"] = mensal["mes"].dt.to_timestamp()

    return mensal[
        ["data", "mes", "indicador", "fonte", "valor"]
    ].sort_values(["data", "indicador"])


# =========================================================
# NORMALIZACAO BASE 100
# =========================================================

def normalizar_base_100(df):
    resultado = df.copy()

    # Primeira observacao positiva de cada serie = base 100.
    resultado["base"] = (
        resultado.groupby("indicador")["valor"]
        .transform("first")
    )

    resultado["base100"] = np.nan

    validos = resultado["base"] > 0

    resultado.loc[validos, "base100"] = (
        resultado.loc[validos, "valor"]
        / resultado.loc[validos, "base"]
        * 100
    )

    return resultado


# =========================================================
# GRAFICO COMPARATIVO
# =========================================================

def gerar_grafico_normalizado(df):
    fig = px.line(
        df,
        x="data",
        y="base100",
        color="indicador",
        markers=False,
        title="Indicadores economicos - base 100",
        labels={
            "data": "Data",
            "base100": "Indice (base 100)",
            "indicador": "Indicador",
        },
    )

    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
    )

    caminho = PASTA_SAIDA / "comparacao_base100.html"
    fig.write_html(str(caminho))

    print(f"\nGrafico salvo em: {caminho.resolve()}")


# =========================================================
# CORRELACAO E DISPERSAO
# =========================================================

def analisar_correlacao(df, nome_x, nome_y):
    x_df = df.loc[
        df["indicador"] == nome_x,
        ["mes", "valor"],
    ].rename(columns={"valor": "x"})

    y_df = df.loc[
        df["indicador"] == nome_y,
        ["mes", "valor"],
    ].rename(columns={"valor": "y"})

    # Combina somente meses presentes nas duas series.
    pares = pd.merge(
        x_df,
        y_df,
        on="mes",
        how="inner",
    ).dropna(subset=["x", "y"])

    if len(pares) < 3:
        print(
            "\nNao ha pelo menos 3 meses em comum "
            "para calcular a correlacao."
        )
        return

    if pares["x"].nunique() < 2 or pares["y"].nunique() < 2:
        print(
            "\nNao e possivel calcular a correlacao: "
            "uma das series nao apresenta variacao."
        )
        return

    correlacao = pares["x"].corr(pares["y"])

    # Reta de regressao linear por minimos quadrados.
    m, b = np.polyfit(
        pares["x"].to_numpy(),
        pares["y"].to_numpy(),
        1,
    )

    x_linha = np.linspace(
        pares["x"].min(),
        pares["x"].max(),
        100,
    )

    y_linha = m * x_linha + b

    print("\n========== CORRELACAO ==========")
    print(f"Indicador X: {nome_x}")
    print(f"Indicador Y: {nome_y}")
    print(f"Meses em comum: {len(pares)}")
    print(f"Correlacao de Pearson: {correlacao:.4f}")

    if correlacao > 0:
        print("Associacao linear positiva.")
    elif correlacao < 0:
        print("Associacao linear negativa.")
    else:
        print("Correlacao linear igual a zero.")

    print("Lembre-se: correlacao nao implica causalidade.")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pares["x"],
            y=pares["y"],
            mode="markers",
            name="Observacoes mensais",
            text=pares["mes"].astype(str),
            hovertemplate=(
                "Mes: %{text}<br>"
                "X: %{x}<br>"
                "Y: %{y}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_linha,
            y=y_linha,
            mode="lines",
            name="Linha de tendencia",
        )
    )

    fig.update_layout(
        title=(
            f"Correlacao: {nome_x} x {nome_y}"
            f" (r = {correlacao:.3f})"
        ),
        xaxis_title=nome_x,
        yaxis_title=nome_y,
        template="plotly_white",
    )

    caminho = PASTA_SAIDA / "correlacao.html"
    fig.write_html(str(caminho))

    print(f"Grafico salvo em: {caminho.resolve()}")

    pares["mes"] = pares["mes"].astype(str)
    pares.to_csv(
        PASTA_SAIDA / "pares_correlacao.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# MENU DO TERMINAL
# =========================================================

def main():
    print("=" * 60)
    print(" DASHBOARD DE INDICADORES ECONOMICOS BRASILEIROS")
    print("=" * 60)

    print("\nIndicadores disponiveis:")

    for chave, indicador in INDICADORES.items():
        print(
            f"{chave}. {indicador['nome']} "
            f"- {indicador['fonte']} "
            f"(codigo: {indicador['codigo']})"
        )

    print("\nDigite ate 5 numeros separados por virgula.")
    print("Exemplo: 1,2,4")

    entrada = input("\nIndicadores: ").strip()

    chaves = list(dict.fromkeys(
        item.strip() for item in entrada.split(",")
        if item.strip()
    ))

    if not chaves or len(chaves) > 5:
        print("Escolha entre 1 e 5 indicadores.")
        return

    invalidos = [
        chave for chave in chaves
        if chave not in INDICADORES
    ]

    if invalidos:
        print(f"Opcoes invalidas: {', '.join(invalidos)}")
        return

    inicio_texto = input(
        "Data inicial (AAAA-MM-DD) [2018-01-01]: "
    ).strip()

    fim_texto = input(
        f"Data final (AAAA-MM-DD) [{date.today()}]: "
    ).strip()

    try:
        inicio = (
            converter_data(inicio_texto)
            if inicio_texto else pd.Timestamp("2018-01-01")
        )
        fim = (
            converter_data(fim_texto)
            if fim_texto else pd.Timestamp(date.today())
        )
    except ValueError:
        print("Data invalida. Use AAAA-MM-DD.")
        return

    if inicio >= fim:
        print("A data inicial precisa ser anterior a final.")
        return

    dados = []
    erros = []

    for chave in chaves:
        try:
            dados.append(
                buscar_indicador(chave, inicio, fim)
            )
        except Exception as erro:
            erros.append(
                f"{INDICADORES[chave]['nome']}: {erro}"
            )

    for erro in erros:
        print(f"\nERRO: {erro}")

    if not dados:
        print("\nNenhuma serie foi carregada com sucesso.")
        return

    bruto = pd.concat(dados, ignore_index=True)

    bruto.to_csv(
        PASTA_SAIDA / "dados_originais.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mensal = converter_para_mensal(bruto)
    normalizado = normalizar_base_100(mensal)

    normalizado.to_csv(
        PASTA_SAIDA / "dados_normalizados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== RESUMO ==========")
    print(
        mensal.groupby("indicador")["valor"]
        .agg(["count", "min", "max", "last"])
        .to_string()
    )

    gerar_grafico_normalizado(normalizado)

    nomes = list(mensal["indicador"].unique())

    if len(nomes) >= 2:
        print("\nIndicadores para correlacao:")

        for i, nome in enumerate(nomes, start=1):
            print(f"{i}. {nome}")

        escolha = input(
            "Digite dois numeros separados por virgula "
            "(Enter para pular): "
        ).strip()

        if escolha:
            try:
                indices = [
                    int(x.strip()) for x in escolha.split(",")
                ]

                if (
                    len(indices) != 2
                    or indices[0] == indices[1]
                    or any(i < 1 or i > len(nomes) for i in indices)
                ):
                    print("Selecao de correlacao invalida.")
                else:
                    analisar_correlacao(
                        mensal,
                        nomes[indices[0] - 1],
                        nomes[indices[1] - 1],
                    )
            except ValueError:
                print("Digite dois numeros validos.")

    print("\nArquivos gerados na pasta resultados/")
    print("Concluido.")


if __name__ == "__main__":
    main()
