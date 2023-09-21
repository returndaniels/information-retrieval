import nltk
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

class Indexer():
  def __init__(self, documents: list=[], stopwords: list=[], spliters: list=[], lang: str='portuguese'):
    """
    Inicializa um objeto Indexer para indexar documentos.

    Args:
        documents (list, optional): Uma lista de documentos a serem indexados.
        stopwords (list, optional): Uma lista de palavras-chave (stopwords) a serem removidas durante o processamento.
        spliters (list, optional): Uma lista de delimitadores usados para dividir os documentos em tokens.
        lang (str, optional): O idioma a ser usado para o stemming (derivada de palavras).

    Example:
        >>> doc_list = [["Este é um exemplo de documento."]]
        >>> stopwords_list = ["é", "um", "de"]
        >>> splitters_list = [" ", ".", "!"]
        >>> lang = "portuguese"
        >>> indexer = Indexer(documents=doc_list, stopwords=stopwords_list, spliters=splitters_list, lang=lang)
    """
    self.documents = documents
    self.spliters = spliters
    self.lang = lang

    self.stopwords = []
    for stopword in stopwords:
      self.stopwords.append(self.normalize(stopword))

  def tokenize(self, s: str):
    """
    Divide uma string em uma lista de tokens com base em delimitadores.

    Args:
        s (str): A string a ser dividida em tokens.

    Returns:
        list: Uma lista de tokens resultantes após a divisão da string.

    Example:
        >>> text = "Esta é uma frase de exemplo, com delimitadores."
        >>> self.spliters = [" ", ",", "."]
        >>> result = self.tokenize(text)
        >>> print(result)
        ["Esta", "é", "uma", "frase", "de", "exemplo", "com", "delimitadores"]
    """
    result = [s]
    for spliter in self.spliters:
        temp_result = []
        for item in result:
            temp_result.extend(item.split(spliter))
        result = temp_result

    return [item for item in result if item]

  def normalize(self, s: str):
    """
    Realiza a normalização das palavras em uma string.

    Args:
        s (str): Uma string para ser normalizada.

    Returns:
        str: Uma nova string de normalizada.

    Example:
        >>> text = "Esta String Vai ser Normalizada"
        >>> result = self.normalize(text)
        >>> print(result)
        "esta string vai ser normalizada"
    """
    return s.lower()

  def stemming(self, document: list):
    """
    Realiza o stemming das palavras em um documento.

    Args:
        document (list): Uma lista de palavras que compõem o documento.

    Returns:
        list: Uma nova lista de palavras que representa o documento após o stemming.

    Example:
        >>> document = ["correr", "correndo", "corria"]
        >>> self.lang = "portuguese"
        >>> result = self.stemming(document)
        >>> print(result)
        ["corr", "corr", "corr"]
    """
    stemmer = SnowballStemmer(self.lang)
    return [stemmer.stem(word) for word in document]

  def stopwords_elimination(self, document: list):
    """
    Remove palavras-chave (stopwords) de um documento.

    Args:
        document (list): Uma lista de palavras que compõem o documento.

    Returns:
        list: Uma nova lista de palavras que representa o documento após a remoção das stopwords.

    Example:
        >>> document = ["Este", "é", "um", "exemplo", "com", "algumas", "stopwords"]
        >>> stopwords = ["é", "um", "com", "algumas"]
        >>> result = self.stopwords_elimination(document, stopwords)
        >>> print(result)
        ["Este", "exemplo", "stopwords"]
    """
    return [word for word in document if all(word != sw for sw in self.stopwords)]

  def search(self, query: str, exact: bool=False):
    """
    Pesquisa documentos na coleção com base em uma consulta.

    Args:
        query (str): A consulta que define os termos de pesquisa.
        exact (bool, optional): Se True, a pesquisa requer uma correspondência exata
            de todos os termos da consulta nos documentos. Se False, a pesquisa
            considerará qualquer documento que contenha pelo menos um termo da consulta.

    Returns:
        list: Uma lista de documentos que correspondem à consulta. Cada documento é representado
        como uma única string, onde as palavras são separadas por espaços.

    Example:
        Para pesquisar documentos que contenham pelo menos uma palavra da consulta:
        >>> result = self.search("exemplo de pesquisa", exact=False)

        Para pesquisar documentos que correspondam exatamente à consulta:
        >>> result = self.search("exemplo de pesquisa", exact=True)
    """
    
    normalized_query = self.normalize(query)
    tokenized_query = self.tokenize(normalized_query)
    stemmed_query = self.stemming(tokenized_query)

    documents = []
    for doc in self.documents:
      for stantment in doc:
        normalized_stantment = self.normalize(stantment)
        tokenized_stantment = self.tokenize(normalized_stantment)
        clean_stantment = self.stopwords_elimination(tokenized_stantment)
        stemmed_document = self.stemming(clean_stantment)
        if not(exact) and any(word in stemmed_query for word in stemmed_document):
          documents.append([' '.join(doc)])
        elif exact and all(q in stemmed_document for q in stemmed_query):
          documents.append([' '.join(doc)])

    # Calcular o TF-IDF dos documentos e da consulta
    tfidf_vectorizer = TfidfVectorizer()
    corpus = [' '.join(doc) for doc in self.documents]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    query_tfidf = tfidf_vectorizer.transform([normalized_query])

    # Calcular os scores de similaridade TF-IDF
    similarity_scores = (tfidf_matrix * query_tfidf.T).toarray()

    # Ordenar os documentos por relevância (pontuação de similaridade TF-IDF)
    ranked_documents = [(doc, score) for doc, score in zip(self.documents, similarity_scores)]
    ranked_documents.sort(key=lambda x: x[1], reverse=True)

    return ranked_documents
