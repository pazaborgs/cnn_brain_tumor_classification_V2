"""Módulo de Carregamento de Configurações Globais.

Fornece utilitários para ler e decodificar o arquivo de configuração YAML
garantindo portabilidade de caminhos em qualquer sistema operacional.
"""

from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carrega e decodifica o arquivo YAML de configuração global do projeto.

    Resolve o caminho absoluto do arquivo a partir da raiz do repositório,
    garantindo portabilidade independentemente do diretório de execução.

    Args:
        config_path: Caminho relativo do arquivo YAML de configuração.

    Returns:
        Dicionário contendo os parâmetros de projeto, caminhos, treino e modelos.

    Raises:
        FileNotFoundError: Se o arquivo YAML de configuração não for localizado.
    """
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[2]
    full_path = project_root / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado em: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_url = config.get("dataset", {}).get("url", "")
    if dataset_url:
        kaggle_input_dir = Path("/kaggle/input") / dataset_url.split("/")[-1]
        if kaggle_input_dir.exists():
            if "paths" not in config:
                config["paths"] = {}
            config["paths"]["raw"] = str(kaggle_input_dir)

    return config
