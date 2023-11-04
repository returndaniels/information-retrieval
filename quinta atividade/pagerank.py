import numpy as np
from typing import List


def pagerank(
    A: List[List[int]], beta: float = 0.85, threshold: float = 0.0001
) -> np.ndarray:
    """
    Calcula o algoritmo PageRank para uma dada matriz de adjacência.

    Parâmetros:
    A (List[List[int]]): Matriz de adjacência representando a estrutura do grafo.
    beta (float): Parâmetro de amortecimento, entre 0 e 1. Padrão é 0.85.
    threshold (float): Limiar de convergência. Padrão é 0.0001.

    Retorna:
    np.ndarray: Array NumPy contendo os valores do PageRank para cada página.
    """
    T = len(A)
    initial_value = 1 / T
    PR = np.array([initial_value] * T)

    while True:
        new_PR = np.array([(1 - beta) / T] * T)
        for i in range(T):
            for j in range(T):
                if A[j][i] == 1:
                    new_PR[i] += beta * (PR[j] / sum(A[j]))

        if np.max(np.abs(PR - new_PR)) < threshold:
            break

        PR = new_PR

    return PR


def print_pagerank(A: List[List[int]]) -> None:
    """
    Imprime os valores do PageRank para cada página em uma dada matriz de adjacência.

    Parâmetros:
    A (List[List[int]]): Matriz de adjacência representando a estrutura do grafo.
    """
    result = pagerank(A)
    print("PageRank para cada página:")
    for i, pr in enumerate(result):
        print(f"PR({i + 1}) = {pr:.4f}")


# Exemplo de matriz de adjacências
A = [[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0], [1, 0, 1, 0]]

print_pagerank(A)
