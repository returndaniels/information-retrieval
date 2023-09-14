# Indexer

O `Indexer` é uma classe Python que facilita a indexação e busca de documentos de texto. Ele oferece recursos para normalização de texto, divisão em tokens, remoção de stopwords, stemming e pesquisa de documentos com base em consultas.

## Instalação

Para usar a classe `Indexer`, você precisará ter o Python instalado no seu sistema. Se você ainda não o tem, você pode baixá-lo e instalá-lo a partir do [site oficial do Python.]([https://exemplo.com/](https://www.python.org/downloads/)https://www.python.org/downloads/)

Além disso, o Indexer depende da biblioteca NLTK (Natural Language Toolkit) para o stemming e outras operações de processamento de linguagem natural. Você pode instalar o NLTK usando o pip:

```bash
pip install nltk
```

Depois de instalar o NLTK, você precisará fazer o download dos recursos específicos para o stemming, dependendo do idioma que deseja usar. Por exemplo, para o idioma português, você pode fazer o download dos recursos da seguinte maneira:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')  # Recurso para stemming em português
```

## Usando o `main.py`

Você pode usar o arquivo main.py para executar consultas na classe `Indexer` e obter resultados. Aqui está um exemplo de como usar o `main.py`:

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
