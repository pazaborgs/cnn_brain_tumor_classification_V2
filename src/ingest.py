"""Módulo de Ingestão de Dados MRI do Kaggle.

Gerencia a verificação automatizada do dataset, detecção do ambiente nativo do Kaggle
e download/extração de arquivos via Kaggle API Token.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(".secrets/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def ingest_images() -> None:
    """Realiza a ingestão e extração do dataset de exames MRI do Kaggle.

    Verifica diretórios locais brutos e ambientes nativos de notebooks Kaggle
    antes de efetuar chamadas de autenticação e download na API do Kaggle.

    Raises:
        RuntimeError: Caso ocorra falha de autenticação ou extração do dataset.
    """
    config = load_config()

    dataset_url = config["dataset"]["url"]
    images_path = Path(config["paths"]["raw"])
    images_path.mkdir(parents=True, exist_ok=True)

    train_dir = images_path / "Training"
    test_dir = images_path / "Testing"

    print("[INFO] Iniciando processo de ingestão de dados...")

    if train_dir.exists() and test_dir.exists() and any(train_dir.iterdir()):
        print("[SUCESSO] Dataset já existe no diretório bruto local.")
        print(f"[INFO] Caminho: {images_path}")
        return

    kaggle_input_dir = Path("/kaggle/input") / dataset_url.split("/")[-1]
    if kaggle_input_dir.exists():
        print("[SUCESSO] Detectado ambiente nativo Kaggle Notebooks.")
        print(f"[INFO] Caminho: {kaggle_input_dir}")
        return

    try:
        from kaggle import api

        api_token = os.getenv("KAGGLE_API_TOKEN")

        if api_token:
            print("[INFO] Autenticando via KAGGLE_API_TOKEN...")
        else:
            print("[INFO] Autenticando via credenciais padrão...")

        api.authenticate()
        print(f"[INFO] Baixando dataset '{dataset_url}' do Kaggle...")
        api.dataset_download_files(dataset_url, path=images_path, unzip=True)
        print("[SUCESSO] Dataset baixado e descompactado com sucesso.")
        print(f"[INFO] Destino: {images_path}")

    except Exception as e:
        print(f"[ERRO] Falha ao autenticar ou baixar dataset via Kaggle API: {e}")
        raise RuntimeError(f"Falha na ingestão do dataset: {e}") from e


if __name__ == "__main__":
    ingest_images()
