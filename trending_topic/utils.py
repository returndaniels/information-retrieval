import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
import datetime

from typing import Callable, Any, Tuple

import csv
import os


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


def log_error(start_time: datetime.datetime, step_message: str, error: Exception):
    """
    Registra uma mensagem de erro com o tempo decorrido desde o início do programa.

    Args:
    - start_time (datetime): Tempo de início do programa.
    - step_message (str): Mensagem descritiva da etapa onde ocorreu o erro.
    - error (Exception): Exceção capturada.

    Raises:
    - Exception: Relança a exceção para encerrar o programa.
    """
    time = datetime.datetime.now()
    duration = time - start_time
    error_msg = f"Erro durante {step_message}: {error}"
    print(f"[\033[91m{duration}\033[0m] \033[91m{error_msg}\033[0m")


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
    try:
        result = func(*args, **kwargs)
        log(start_time, f"{step_message} foi concluído.")
        return result
    except Exception as e:
        log_error(start_time, step_message, e)
        log(start_time, "Encerrando programa.")
        exit(1)


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


def write_output(
    scored_terms: dict, output_dir: str, output_filename: str, fieldnames: list
):
    """
    Escreve os termos classificados em um arquivo CSV.

    Args:
    - scored_terms (dict): Dicionário contendo os termos e seus scores TF-IDF normalizados.
    - output_dir (str): Caminho para o diretório onde o arquivo CSV será salvo.
    - output_filename (str): Nome base do arquivo CSV a ser salvo.
    - fieldnames (list): Lista dos campos para o cabeçalho do arquivo CSV.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    existing_files = [
        filename
        for filename in os.listdir(output_dir)
        if filename.startswith(output_filename)
    ]
    last_number = max(
        [int(file.split("_")[-1].split(".")[0]) for file in existing_files], default=0
    )
    next_number = last_number + 1
    output_file = os.path.join(output_dir, f"{output_filename}{next_number:02d}.csv")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for term, values in scored_terms.items():
            row = {fieldname: values.get(fieldname, "") for fieldname in fieldnames}
            row["term"] = term
            writer.writerow(row)
