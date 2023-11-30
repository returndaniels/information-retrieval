import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
import datetime

from typing import Callable, Any, Tuple


def log(start_time: datetime.datetime, msg: str):
    """
    Registra uma mensagem de log com o tempo decorrido desde o início do programa.

    Args:
    - start_time (datetime): Tempo de início do programa.
    - msg (str): Mensagem a ser registrada no log.
    """
    time = datetime.datetime.now()
    duration = time - start_time
    print(f"[\033[91m{duration}\033[0m] {msg}")


def log_step(
    start_time: datetime.datetime,
    step_message: str,
    func: Callable[..., Any],
    *args: Tuple[Any],
    **kwargs: Any,
):
    """
    Registra uma mensagem de log para o início e o término de uma etapa do processo.

    Args:
    - start_time (datetime): Tempo de início do programa.
    - step_message (str): Mensagem descritiva da etapa.
    - func (function): Função a ser executada na etapa.
    - *args, **kwargs: Argumentos e palavras-chave para a função.

    Returns:
    - result: Resultado da execução da função.
    """
    log(start_time, f"Iniciando {step_message}...")
    result = func(*args, **kwargs)
    log(start_time, f"{step_message} foi concluído.")
    return result


def get_stopwords(stopwords_path: str, lang: str):
    """
    Obtém as stopwords para um idioma específico a partir de um arquivo.

    Args:
    - stopwords_path (str): Caminho para o arquivo de stopwords.
    - lang (str): Idioma das stopwords a serem carregadas.

    Returns:
    - set: Conjunto de stopwords para o idioma especificado.
    """
    df = pd.read_fwf(stopwords_path, header=None)
    custom_stopwords = df.values.tolist()
    custom_stopwords = [s[0] for s in custom_stopwords]
    stop_words = set(stopwords.words(lang))
    stop_words.update(custom_stopwords)
    return stop_words


def preprocess_documents(data: pd.DataFrame):
    """
    Realiza o pré-processamento dos documentos removendo pontuações, stopwords e palavras com apenas uma letra.

    Args:
    - data (pd.DataFrame): Lista de strings representando os documentos.

    Returns:
    - pd.DataFrame: Documentos pré-processados.
    """
    tokenizer = TweetTokenizer()
    data = data.apply(lambda x: " ".join(tokenizer.tokenize(x.lower())))
    return data
