"""Orquestrador Principal do Pipeline de Classificação de Tumores Cerebrais em MRI.

Suporta modos de execução via linha de comando (CLI):
  --mode train     : Ingestão, Treinamento e Avaliação Unificada
  --mode evaluate  : Avalia os modelos salvos em artifacts/models/ no conjunto de teste
  --mode upload_hf : Publica os modelos salvos em artifacts/models/ no Hugging Face Hub
  --mode app       : Inicia o painel interativo Streamlit
"""

import argparse
import os
import subprocess
import sys

from src.evaluate_model import evaluate_model
from src.ingest import ingest_images
from src.train_model import train_model
from src.utils.load_config import load_config
from src.utils.upload_models_hf import upload_models_to_huggingface

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    """Ponto de entrada principal do orquestrador CLI.
    
    Analisa os argumentos da linha de comando e direciona a execução 
    para o módulo correspondente do pipeline.
    """
    parser = argparse.ArgumentParser(
        description="CNN Brain Tumor MRI Classification Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "upload_hf", "app"],
        default="evaluate",
        help="Modo de execução: 'train' (treina e avalia), 'evaluate' (avalia modelos), 'upload_hf' (publica no HF), 'app' (inicia Streamlit)",
    )
    args = parser.parse_args()

    print("========================================================================")
    print(f"[INFO] PIPELINE CNN BRAIN TUMOR CLASSIFICATION V2 — MODO: [{args.mode.upper()}]")
    print("========================================================================\n")

    config = load_config()

    if args.mode == "train":
        ingest_images()
        for model_cfg in config["models"]:
            train_model(config=config, model_config=model_cfg)

    elif args.mode == "evaluate":
        print("[INFO] Modo de avaliação local. Carregando modelos em artifacts/models/...")
        failures = []
        for model_cfg in config["models"]:
            try:
                evaluate_model(config=config, model_config=model_cfg)
            except Exception as e:
                failures.append((model_cfg["name"], e))
                print(f"[ERRO] Falha ao avaliar {model_cfg['name']}: {e}")

        if failures:
            print(f"[ERRO] {len(failures)} modelo(s) falharam na avaliação.")
            sys.exit(1)

    elif args.mode == "upload_hf":
        print("[INFO] Modo de publicação no Hugging Face Hub...")
        upload_models_to_huggingface()

    elif args.mode == "app":
        print("[INFO] Iniciando interface Streamlit (app.py)...")
        app_file = "app.py"
        if not os.path.exists(app_file):
            print("[ERRO] Arquivo app.py não encontrado na raiz do projeto.")
            return
        subprocess.run(["streamlit", "run", app_file])

    print("========================================================================")
    print("[SUCESSO] EXECUÇÃO FINALIZADA.")
    print("========================================================================\n")


if __name__ == "__main__":
    main()
