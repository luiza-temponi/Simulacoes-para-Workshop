"""
Simulação de Monte Carlo da probabilidade de sucesso de uma busca gulosa
em uma árvore binária com pesos aleatórios nas arestas.

Distribuições disponíveis:
    - "unif": uniforme em [0, 1];
    - "exp": exponencial de taxa lambda, truncada em [0, 1];
    - "normal": normal de média mu e desvio-padrão sigma, truncada em [0, 1].

Para cada profundidade k, considera-se sucesso quando o custo do caminho
escolhido pela heurística gulosa é igual ao menor custo entre todos os
caminhos da raiz até o nível k.

Exemplo:
    python simulacao_gulosa.py normal 1000000
    python simulacao_gulosa.py exp 1000000
    python simulacao_gulosa.py unif 1000000
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri


def criar_amostrador(distribuicao, gerador, taxa=1.0, media=0.5, desvio=0.15):
    """
    Cria uma função que gera pesos aleatórios na faixa [0, 1].

    Parâmetros
    ----------
    distribuicao : str
        Distribuição dos pesos: "unif", "exp" ou "normal".
    gerador : numpy.random.Generator
        Gerador pseudoaleatório utilizado na simulação.
    taxa : float
        Taxa da distribuição exponencial (lambda).
    media : float
        Média da distribuição normal antes do truncamento.
    desvio : float
        Desvio-padrão da distribuição normal antes do truncamento.

    Retorna
    -------
    callable
        Função que recebe uma forma (shape) e devolve uma amostra
        com a forma solicitada.
    """
    if distribuicao == "unif":
        # A distribuição uniforme já está definida em [0, 1].
        return lambda forma: gerador.random(forma)

    if distribuicao == "exp":
        # Normalização da exponencial condicionada ao intervalo [0, 1].
        # F(1) = 1 - exp(-taxa).
        normalizacao = 1.0 - np.exp(-taxa)

        def amostrar_exponencial(forma):
            uniforme = gerador.random(forma)
            return -np.log1p(-uniforme * normalizacao) / taxa

        return amostrar_exponencial

    if distribuicao == "normal":
        # CDF normal nos limites do intervalo de truncamento.
        limite_inferior = ndtr((0.0 - media) / desvio)
        limite_superior = ndtr((1.0 - media) / desvio)

        def amostrar_normal(forma):
            # Transformada inversa da CDF para gerar uma normal
            # condicionada a assumir valores entre 0 e 1.
            uniforme = gerador.random(forma)
            probabilidade = limite_inferior + uniforme * (
                limite_superior - limite_inferior
            )
            return media + desvio * ndtri(probabilidade)

        return amostrar_normal

    raise ValueError(
        "Distribuição desconhecida. Escolha 'unif', 'exp' ou 'normal'."
    )


def simular_probabilidades(
    distribuicao,
    numero_simulacoes,
    profundidade_maxima=9,
    tamanho_lote=20_000,
    semente=1,
    **parametros_distribuicao,
):
    """
    Estima, por Monte Carlo, a probabilidade de sucesso da busca gulosa.

    A simulação é processada em lotes para limitar o consumo de memória.

    Parâmetros
    ----------
    distribuicao : str
        Distribuição dos pesos das arestas.
    numero_simulacoes : int
        Número total de árvores simuladas.
    profundidade_maxima : int
        Maior profundidade analisada.
    tamanho_lote : int
        Número de árvores processadas em cada lote.
    semente : int
        Semente do gerador pseudoaleatório, para reprodutibilidade.
    **parametros_distribuicao
        Parâmetros adicionais enviados a criar_amostrador.

    Retorna
    -------
    numpy.ndarray
        Estimativas da probabilidade de sucesso para as profundidades
        de 1 até profundidade_maxima.
    """
    if numero_simulacoes <= 0:
        raise ValueError("'numero_simulacoes' deve ser positivo.")
    if profundidade_maxima <= 0:
        raise ValueError("'profundidade_maxima' deve ser positiva.")
    if tamanho_lote <= 0:
        raise ValueError("'tamanho_lote' deve ser positivo.")

    gerador = np.random.default_rng(semente)
    amostrar_pesos = criar_amostrador(
        distribuicao, gerador, **parametros_distribuicao
    )

    # acertos[k] conta quantas simulações tiveram sucesso na profundidade k+1.
    acertos = np.zeros(profundidade_maxima, dtype=np.int64)
    simulacoes_concluidas = 0

    while simulacoes_concluidas < numero_simulacoes:
        quantidade_lote = min(
            tamanho_lote, numero_simulacoes - simulacoes_concluidas
        )

        # Custos dos caminhos completos até o nível corrente.
        # Inicialmente, há apenas a raiz, com custo zero.
        custos_caminhos = np.zeros((quantidade_lote, 1))

        # Índice do nó guloso no nível atual, separadamente para cada árvore.
        indice_guloso = np.zeros(quantidade_lote, dtype=np.int64)

        # Custo acumulado do caminho escolhido pela heurística gulosa.
        custo_guloso = np.zeros(quantidade_lote)

        indices_simulacoes = np.arange(quantidade_lote)

        for nivel in range(profundidade_maxima):
            # Para cada nó do nível atual, gera os pesos das duas arestas
            # que levam aos filhos esquerdo e direito.
            pesos = amostrar_pesos(
                (quantidade_lote, custos_caminhos.shape[1], 2)
            )

            # Pesos das duas arestas que saem do nó escolhido pela busca gulosa.
            pesos_no_guloso = pesos[indices_simulacoes, indice_guloso]

            # A heurística gulosa escolhe a aresta de menor peso imediato.
            # Em caso de empate, escolhe o filho direito, como no código original.
            escolhe_direita = (
                pesos_no_guloso[:, 1] < pesos_no_guloso[:, 0]
            ).astype(np.int64)

            # Atualiza o custo e o índice do nó escolhido.
            custo_guloso += pesos_no_guloso[
                indices_simulacoes, escolhe_direita
            ]
            indice_guloso = 2 * indice_guloso + escolhe_direita

            # Calcula os custos acumulados de todos os caminhos até o próximo
            # nível da árvore, somando o peso de cada aresta ao custo do pai.
            custos_caminhos = (
                custos_caminhos[:, :, np.newaxis] + pesos
            ).reshape(quantidade_lote, -1)

            # Sucesso quando o caminho guloso tem custo mínimo global.
            menor_custo = custos_caminhos.min(axis=1)
            acertos[nivel] += np.count_nonzero(
                custo_guloso <= menor_custo + 1e-12
            )

        simulacoes_concluidas += quantidade_lote

    return acertos / numero_simulacoes


def main():
    """Lê os argumentos da linha de comando, executa e salva a simulação."""
    analisador = argparse.ArgumentParser(
        description=(
            "Estima por Monte Carlo a probabilidade de sucesso da busca "
            "gulosa em uma árvore binária."
        )
    )
    analisador.add_argument(
        "distribuicao",
        choices=["unif", "exp", "normal"],
        help="Distribuição dos pesos das arestas.",
    )
    analisador.add_argument(
        "simulacoes",
        type=int,
        help="Número total de simulações.",
    )
    analisador.add_argument(
        "--profundidade",
        type=int,
        default=9,
        help="Profundidade máxima da árvore (padrão: 9).",
    )
    analisador.add_argument(
        "--lote",
        type=int,
        default=20_000,
        help="Árvores processadas por lote (padrão: 20000).",
    )
    analisador.add_argument(
        "--semente",
        type=int,
        default=1,
        help="Semente aleatória para reprodutibilidade (padrão: 1).",
    )
    analisador.add_argument(
        "--taxa",
        type=float,
        default=1.0,
        help="Taxa lambda da exponencial (padrão: 1.0).",
    )
    analisador.add_argument(
        "--media",
        type=float,
        default=0.5,
        help="Média da normal antes do truncamento (padrão: 0.5).",
    )
    analisador.add_argument(
        "--desvio",
        type=float,
        default=0.15,
        help="Desvio-padrão da normal antes do truncamento (padrão: 0.15).",
    )
    argumentos = analisador.parse_args()

    inicio = time.time()

    parametros = {}
    if argumentos.distribuicao == "exp":
        parametros["taxa"] = argumentos.taxa
    elif argumentos.distribuicao == "normal":
        parametros["media"] = argumentos.media
        parametros["desvio"] = argumentos.desvio

    probabilidades = simular_probabilidades(
        distribuicao=argumentos.distribuicao,
        numero_simulacoes=argumentos.simulacoes,
        profundidade_maxima=argumentos.profundidade,
        tamanho_lote=argumentos.lote,
        semente=argumentos.semente,
        **parametros,
    )

    # Erro-padrão binomial aproximado de cada estimativa Monte Carlo.
    erros_padrao = np.sqrt(
        probabilidades * (1.0 - probabilidades) / argumentos.simulacoes
    )

    nome_arquivo = Path(f"mc_{argumentos.distribuicao}.npz")
    np.savez(
        nome_arquivo,
        p=probabilidades,
        se=erros_padrao,
        N=argumentos.simulacoes,
        profundidade=argumentos.profundidade,
        semente=argumentos.semente,
    )

    print(f"Distribuição: {argumentos.distribuicao}")
    print(f"Número de simulações: {argumentos.simulacoes}")
    print(f"Tempo decorrido: {time.time() - inicio:.2f} s")
    print("Probabilidades estimadas:", np.round(probabilidades, 5))
    print("Erros-padrão estimados:", np.round(erros_padrao, 6))
    print(f"Resultados salvos em: {nome_arquivo}")


if __name__ == "__main__":
    main()
