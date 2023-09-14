# Indexer

O Indexer é uma classe Python que facilita a indexação e busca de documentos de texto. Ele oferece recursos para normalização de texto, divisão em tokens, remoção de stopwords, stemming e pesquisa de documentos com base em consultas.

## Instalação

Para usar a classe Indexer, você precisará ter o Python instalado no seu sistema. Se você ainda não o tem, você pode baixá-lo e instalá-lo a partir do [site oficial do Python.]([https://exemplo.com/](https://www.python.org/downloads/)https://www.python.org/downloads/)

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

## Como Usar

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

Este exemplo ilustra como inicializar um objeto Indexer, realizar pesquisas com correspondência exata ou parcial e imprimir os resultados.

Lembre-se de adaptar os documentos, stopwords, spliters e idioma conforme necessário para suas necessidades específicas.
