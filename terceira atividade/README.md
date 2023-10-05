# Recuperação da Informação com BM25 Similarity

Este é o código da terceira atividade da disciplina de Recuperação da Informação. Nesta atividade, implementamos um mecanismo de recuperação de informações usando a medida de similaridade BM25. O código está dividido em dois arquivos: `index.py` e `main.py`. O `index.py` contém a implementação da classe `Indexer`, que é responsável por indexar os documentos e calcular a similaridade BM25 entre os documentos e uma consulta. O `main.py` contém um exemplo de uso da classe `Indexer` para buscar documentos relacionados a uma consulta.

## `index.py`

Este arquivo contém a implementação da classe `Indexer`, que é responsável por indexar os documentos e calcular a similaridade BM25 entre eles e uma consulta. A classe `Indexer` possui os seguintes métodos:

### `__init__`
- Inicializa um objeto `Indexer` com uma lista de documentos, stopwords, delimitadores e idioma para stemming.

### `flat`
- Transforma uma matriz bidimensional em uma lista unidimensional.

### `tokenize`
- Divide uma string em uma lista de tokens com base em delimitadores.

### `normalize`
- Realiza a normalização de palavras em uma string.

### `stemming`
- Realiza o stemming das palavras em um documento.

### `stopwords_elimination`
- Remove stopwords de um documento.

### `pre_process`
- Realiza o pré-processamento de uma string de texto, incluindo normalização, tokenização, eliminação de stopwords e stemming.

### `bm25_similarity`
- Calcula a similaridade BM25 entre um documento e uma consulta.

### `rank_documents`
- Classifica os documentos da coleção com base na similaridade BM25 com uma consulta.

### `search`
- Pesquisa documentos na coleção com base em uma consulta.

## `main.py`

Este arquivo contém um exemplo de uso da classe `Indexer`. Nele, você pode ver como criar uma instância da classe, indexar documentos e realizar uma busca com a consulta.

## Como usar

Para utilizar o código, siga os seguintes passos:

1. Certifique-se de ter o Python instalado em seu ambiente.

2. Instale a biblioteca NLTK (Natural Language Toolkit) se ainda não estiver instalada. Você pode instalar usando o comando:
   ```
   pip install nltk
   ```

3. Execute o arquivo `main.py`. Isso criará uma instância do `Indexer`, indexará os documentos fornecidos e realizará uma busca com a consulta. O exemplo atual realiza uma busca com a consulta "Parasita oscar 2020" e classifica os documentos com base na similaridade BM25.

4. Você pode personalizar os documentos, stopwords, delimitadores e consulta conforme necessário para testar diferentes consultas e configurações.

### Usando o `main.py`

O arquivo main.py serve para executar consultas na classe `Indexer` e obter resultados. Aqui está um exemplo de como usar o `main.py`:

1. Primeiro, verifique se você já inicializou o ambiente e as dependências conforme mencionado nas seções anteriores deste README.
2. Abra o arquivo `main.py` no seu editor de código ou IDE.
3. No trecho de código em `main.py`, você verá a inicialização de um objeto Indexer. 

Certifique-se de que a variável `documents` contém sua lista de documentos, `stopwords` contém sua lista de palavras-chave a serem removidas e `spliters` contém sua lista de delimitadores.

```python
d = [['Parasita é o grande vencedor do Oscar 2020, com quatro prêmios'],
    # ... outros documentos ...
    ['Setembro chegou! Confira o calendário da temporada 2020/2021 do futebol europeu']]
sw = ['a', 'o', 'e', 'é', 'de', 'do', 'da', 'no', 'na', 'são', 'dos', 'com','como','eles', 'em', 'os', 'ao', 'para', 'pelo']
st = [' ',',','.','!','?',':',';','/']

if __name__ == "__main__":
    search_engine = Indexer(documents=d, stopwords=sw, spliters=st, lang="portuguese")
```

Após inicializar o objeto search_engine, você pode usar o método search para realizar consultas. O exemplo abaixo realiza uma pesquisa pela frase "Parasita oscar 2020" com correspondência exata:

```python
search_engine.search(query='Parasita oscar 2020', exact=True)
```
Execute o arquivo `main.py` no seu terminal ou IDE. Os resultados da pesquisa serão impressos na saída, mostrando quais documentos correspondem à consulta especificada.

Isso é tudo o que você precisa fazer para começar a usar o `main.py` para pesquisar documentos usando a classe Indexer. Certifique-se de adaptar os documentos, stopwords, spliters e idioma conforme necessário para suas necessidades específicas.

## Como Usar `Indexer`

Aqui está um exemplo de como usar a classe Indexer:

```python
import nltk
from nltk.stem import SnowballStemmer

# Inicialize o NLTK (uma única vez por sessão)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')  # Recurso para stemming em português

# Importe a classe Indexer
from indexer import Indexer  # Substitua 'indexer' pelo nome do seu arquivo se necessário

# Crie uma lista de documentos de exemplo
doc_list = [
    ["Este é um exemplo de documento."],
    ["Outro exemplo de documento com mais texto."],
    ["E mais um exemplo para completar."]
]

# Defina uma lista de stopwords
stopwords_list = ["é", "um", "de", "com"]

# Defina uma lista de delimitadores (spliters)
spliters_list = [" ", ".", "!"]

# Crie um objeto Indexer com os documentos, stopwords, spliters e idioma
lang = "portuguese"
indexer = Indexer(documents=doc_list, stopwords=stopwords_list, spliters=spliters_list, lang=lang)

# Realize uma pesquisa por documentos que contenham a palavra "exemplo" (correspondência exata)
result_exact = indexer.search("exemplo", exact=True)
print("Resultados da Pesquisa (Correspondência Exata):")
for doc in result_exact:
    print(doc)

# Realize uma pesquisa por documentos que contenham pelo menos uma palavra da consulta
result_partial = indexer.search("texto mais", exact=False)
print("\nResultados da Pesquisa (Correspondência Parcial):")
for doc in result_partial:
    print(doc)
```

Este exemplo ilustra como inicializar um objeto `Indexer`, realizar pesquisas com correspondência exata ou parcial e imprimir os resultados.

Lembre-se de adaptar os documentos, `stopwords`, `spliters` e idioma conforme necessário para suas necessidades específicas.

Você pode estender e personalizar o código para atender às suas necessidades específicas de recuperação de informações.