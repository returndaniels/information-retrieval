# PageRank Algorithm

Este é um programa simples em Python para calcular o PageRank de páginas da web com base em uma estrutura de grafo representada por uma matriz de adjacência. O algoritmo implementado segue os passos descritos no livro "Networks, Crowds, and Markets: Reasoning about a Highly Connected World" por David Easley e Jon Kleinberg.

## Requisitos

Certifique-se de ter o Python 3 instalado. Não há dependências externas para este programa.

## Como Usar

1. Clone o repositório ou baixe o arquivo `pagerank.py` diretamente.

2. No arquivo `pagerank.py`, insira sua própria matriz de adjacência para representar a estrutura de links entre as páginas da web.

3. Execute o script Python:

```bash
python pagerank.py
```

4. Os resultados do PageRank para cada página da web serão exibidos no console.

## Como funciona o algoritmo

O algoritmo segue as seguintes etapas:

1. Inicialize o PageRank para cada página com um valor inicial de 1/T, onde T é o número total de páginas.

2. Defina um valor de threshold para verificar a convergência dos valores de PageRank. Neste caso, o threshold é definido como 0.0001.

3. Atualize os valores de PageRank para cada página iterativamente até que os valores convirjam para o threshold especificado.

4. Utilize a fórmula do PageRank para calcular o PageRank de cada página:

PR(a) = (1 - β) / T + β ∑_{i=1}^{n} (PR(p_i) / L_{p_i})


Onde:

- `a` é a página da web para a qual se deseja determinar o PageRank.
- `L(p_i)` é o número de outlinks da página `p_i`.
- `beta` é um parâmetro de amortecimento, definido como 0.85.

## Exemplo de matriz de adjacência

A matriz de adjacência é uma lista de listas em Python, onde cada lista representa os nós para os quais uma determinada página da web aponta. Abaixo está um exemplo de matriz de adjacência:

```python
A = [
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
]
```