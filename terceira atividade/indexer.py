import math
from nltk.stem import SnowballStemmer


class Indexer:
    def __init__(
        self,
        documents: list = [],
        stopwords: list = [],
        spliters: list = [],
        lang: str = "portuguese",
    ):
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

    def flat(self, arr: list):
        """
        Transforma uma matriz bidimensional em uma lista unidimensional.

        Args:
                arr (list of list): A matriz bidimensional a ser achatada.

        Returns:
                list: Uma lista unidimensional contendo todos os elementos da matriz.

        Example:
                matrix = [
                        [1, 2, 3],
                        [12, 22, 32],
                ]
                arr = flat(matrix)
                # Resultado: [1, 2, 3, 12, 22, 32]
        """
        return [item for row in arr for item in row]

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

    def pre_process(self, s: str):
        """
        Realiza o pré-processamento de uma string de texto.

        Isso inclui as seguintes etapas:
        1. Normalização: Converte todas as palavras para letras minúsculas.
        2. Tokenização: Divide a string em uma lista de tokens com base em delimitadores.
        3. Eliminação de Stopwords: Remove palavras-chave (stopwords) da lista de tokens.
        4. Stemming: Realiza o stemming das palavras restantes.

        Args:
            s (str): A string de texto a ser pré-processada.

        Returns:
            list: Uma lista de palavras que representa o documento após o pré-processamento.

        Example:
            >>> text = "Esta é uma frase de exemplo, com delimitadores."
            >>> self.spliters = [" ", ",", "."]
            >>> self.lang = "portuguese"
            >>> self.stopwords = ["é", "um", "com"]
            >>> result = self.pre_process(text)
            >>> print(result)
            ["esta", "frase", "exemplo", "delimitadores"]
        """
        normalized = self.normalize(s)
        tokenized = self.tokenize(normalized)
        clean = self.stopwords_elimination(tokenized)
        stemmed_document = self.stemming(clean)
        return stemmed_document

    def bm25_similarity(self, doc: list, query: str, avg_doclen: int, k1=1.0, b=0.75):
        """
        Calcula a similaridade BM25 entre um documento e uma consulta.

        A similaridade BM25 é uma medida de relevância entre um documento e uma consulta
        com base no modelo BM25. Este modelo leva em consideração o número de ocorrências
        de termos no documento e na consulta, bem como a frequência inversa do termo (IDF)
        e a média do comprimento dos documentos na coleção.

        Args:
            doc (list): Uma lista de palavras que representa o documento.
            query (str): A consulta para a qual deseja-se calcular a similaridade.
            avg_doclen (int): A média do comprimento dos documentos na coleção.
            k1 (float): Um parâmetro de ajuste para controlar a influência da frequência do termo (padrão: 1.0).
            b (float): Um parâmetro de ajuste para controlar a influência do comprimento do documento (padrão: 0.75).

        Returns:
            float: O valor da similaridade BM25 entre o documento e a consulta.

        Example:
            >>> document = ["esta", "é", "uma", "frase", "de", "exemplo"]
            >>> query = "frase de exemplo"
            >>> avg_doclen = 5  # Média do comprimento dos documentos na coleção
            >>> result = self.bm25_similarity(document, query, avg_doclen)
            >>> print(result)
            0.123456789  # Valor de similaridade BM25 entre o documento e a consulta
        """
        N = len(self.documents)
        doc_score = 0.0
        processed_query = self.pre_process(query)

        for term in processed_query:
            f_ij = 0
            for stantment in doc:
                processed_document = self.pre_process(stantment)
                if term not in processed_document:
                    continue
                f_ij += processed_document.count(term)

            n_i = 0
            for doc_list in self.documents:
                for d in doc_list:
                    if term in self.pre_process(d):
                        n_i += 1
                        break
            idf = math.log((N - n_i + 0.5) / (n_i + 0.5))
            numerator = (k1 + 1) * f_ij
            doc_length = sum(len(self.pre_process(d)) for d in doc)
            denominator = f_ij + k1 * (1 - b + b * doc_length / avg_doclen)
            doc_score += idf * numerator / denominator

        return doc_score

    def rank_documents(self, query, k1=1.5, b=0.75):
        """
        Classifica os documentos da coleção com base na similaridade BM25 com uma consulta.

        Este método calcula a similaridade BM25 entre cada documento na coleção e uma consulta
        especificada. Em seguida, classifica os documentos com base em seus escores de similaridade
        em ordem decrescente.

        Args:
            query (str): A consulta para a qual deseja-se classificar os documentos.
            k1 (float): Um parâmetro de ajuste para controlar a influência da frequência do termo (padrão: 1.5).
            b (float): Um parâmetro de ajuste para controlar a influência do comprimento do documento (padrão: 0.75).

        Returns:
            list: Uma lista de tuplas contendo o índice do documento e seu escore de similaridade BM25
                em relação à consulta, classificados em ordem decrescente de relevância.

        Example:
            >>> query = "palavras-chave importantes"
            >>> k1 = 1.2
            >>> b = 0.8
            >>> # Lista de documentos classificados por relevância
            >>> ranked_documents = self.rank_documents(query, k1, b)
        """
        flat_documents = self.flat(self.documents)
        sum_length = sum(len(self.pre_process(doc)) for doc in flat_documents)
        avg_doclen = sum_length / len(self.documents)

        document_scores = []
        for doc in self.documents:
            score = self.bm25_similarity(doc, query, avg_doclen, k1, b)
            document_scores.append((doc, score))

        document_scores.sort(key=lambda x: x[1], reverse=True)
        return document_scores

    def search(self, query: str, exact: bool = False):
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
        processed_query = self.pre_process(query)
        documents = []
        for doc in self.documents:
            for stantment in doc:
                processed_document = self.pre_process(stantment)
                if not (exact) and any(
                    word in processed_query for word in processed_document
                ):
                    documents.append([" ".join(doc)])
                elif exact and all(q in processed_document for q in processed_query):
                    documents.append([" ".join(doc)])

        return documents
