"""Utilitário de Publicação de Modelos no Hugging Face Hub.

Faz o upload dos modelos .keras salvos em artifacts/models/ para o repositório do Hugging Face.
Suporta autenticação local via arquivo .env ou remota via Kaggle Secrets (HF_TOKEN).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.utils.load_config import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(".secrets/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def get_huggingface_token() -> str | None:
    """Recupera o token de acesso do Hugging Face do ambiente local ou Kaggle Secrets.

    Returns:
        String com o token de escrita do Hugging Face ou None se não localizado.
    """
    # Tentar ler variável de ambiente local (HF_TOKEN ou HUGGINGFACE_TOKEN)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        return hf_token

    # Tentar ler dos Secrets do Kaggle Notebooks (kaggle_secrets)

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        if hf_token:
            return hf_token
    except Exception:
        pass

    return None


def upload_models_to_huggingface(repo_id: str | None = None) -> None:
    """Publica todos os binários .keras de artifacts/models/ no Hugging Face Hub.

    Args:
        repo_id: Identificador opcional do repositório no Hugging Face (ex: 'username/repo').

    Raises:
        ValueError: Se o token HF_TOKEN não for localizado no ambiente ou Kaggle Secrets.
        FileNotFoundError: Se o diretório artifacts/models/ não existir.
    """
    config = load_config()

    if repo_id is None:
        repo_id = config.get("huggingface", {}).get(
            "repo_id", "pazaborgs/brain_tumor_classification_V2"
        )

    token = get_huggingface_token()
    if not token:
        print("[ERRO] Token do Hugging Face (HF_TOKEN) não encontrado.")
        print("       -> No ambiente local: adicione 'HF_TOKEN=seu_token' no arquivo .env")
        print("       -> No Kaggle Notebook: adicione 'HF_TOKEN' em Add-ons -> Secrets")
        raise ValueError(
            "HF_TOKEN ausente. Configure a variável de ambiente ou Kaggle Secret."
        )

    models_dir = Path(config["paths"]["models"])
    if not models_dir.exists():
        print(f"[ERRO] Diretório de modelos não encontrado em: {models_dir}")
        raise FileNotFoundError(
            f"Diretório {models_dir} não existe. Execute o treinamento primeiro."
        )

    model_files = list(models_dir.glob("*.keras"))
    if not model_files:
        print(f"[AVISO] Nenhum modelo .keras encontrado em {models_dir} para upload.")
        return

    print(f"[INFO] Iniciando publicação no Hugging Face Hub (Repositório: '{repo_id}')...")
    api = HfApi(token=token)

    for model_path in model_files:
        filename = model_path.name
        print(f"[INFO] Enviando '{filename}' para Hugging Face...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"[SUCESSO] '{filename}' publicado!")
        except Exception as e:
            print(f"[ERRO] Falha ao enviar '{filename}': {e}")

    print(f"[SUCESSO] Processo finalizado. Repositório: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload_models_to_huggingface()
