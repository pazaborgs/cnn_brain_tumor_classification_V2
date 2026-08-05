import sys
from pathlib import Path
from unittest.mock import patch

# Imports de src

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gradcam import make_gradcam_heatmap
from src.ingest import ingest_images
from src.models import build_model
from src.preprocess import generate_image_datasets
from src.train_model import train_model
from src.utils.load_config import load_config


def print_step(name, status, error=""):
    """Imprime de forma visual no terminal o sucesso ou falha de um passo."""
    color = "\033[92m" if status else "\033[91m"
    icon = "✅" if status else "❌"
    reset = "\033[0m"
    print(f"{color}{icon} {name}{reset}")
    if error:
        print(f"    └─ ERRO: {error}")


def run_visual_tests():
    print("\n" + "=" * 70)
    print("INICIANDO TESTES VISUAIS")
    print("=" * 70 + "\n")

    config = None

    # 1. LOAD CONFIG
    try:
        config = load_config()
        print_step("Módulo load_config.py carregou as configurações", True)
    except Exception as e:
        print_step("Módulo load_config.py", False, str(e))
        return

    # 2. INGEST
    try:
        # Só testa se a função executa sem erros (vai verificar a pasta e pular o download se existir)
        ingest_images()
        print_step("Módulo ingest.py verificou/processou o dataset", True)
    except Exception as e:
        print_step("Módulo ingest.py", False, str(e))

    # 3. PREPROCESS
    try:
        train_ds, val_ds, test_ds, weights, names = generate_image_datasets(
            config=config, model_name="cnn_custom", image_size=128
        )
        print_step(
            f"Módulo preprocess.py carregou datasets ({len(names)} classes)", True
        )
    except Exception as e:
        print_step("Módulo preprocess.py", False, str(e))
        return

    # 4. MODELS
    model = None
    try:
        model_cfg = next(m for m in config["models"] if m["name"] == "cnn_custom")
        model = build_model(config, model_cfg, len(names))
        print_step(
            "Módulo models.py construiu a arquitetura cnn_custom e plotou os grafos",
            True,
        )
    except Exception as e:
        print_step("Módulo models.py", False, str(e))
        return

    # 5. TRAIN MODEL & EVALUATE MODEL (com mock)
    try:
        print(
            "    [!] Simulando dataset reduzido (1 batch) para verificar se o treinamento inicia..."
        )

        # Patching o generate_image_datasets no escopo do train_model
        # Isso força o train_model a receber apenas 1 batch de dados, treinando instantaneamente

        with patch("src.train_model.generate_image_datasets") as mock_gen:
            small_train = train_ds.take(1)
            small_val = val_ds.take(1)
            small_test = test_ds.take(1)

            mock_gen.return_value = (small_train, small_val, small_test, weights, names)

            train_model(config, model_cfg, epochs=1)

        print_step(
            "Módulo train_model.py e evaluate_model.py completaram 1 step com MLflow",
            True,
        )
    except Exception as e:
        print_step("Módulo train_model.py ou evaluate_model.py", False, str(e))

    # 6. GRADCAM
    try:
        if model:
            # Pega o primeiro batch e a primeira imagem desse batch para testar o heatmap
            sample_batch = next(iter(train_ds))
            sample_img = sample_batch[0][0:1]  # shape (1, 128, 128, 3)

            heatmap = make_gradcam_heatmap(sample_img, model, "last_feature_map")

            if heatmap is not None and len(heatmap.shape) == 2:
                print_step(
                    f"Módulo gradcam.py gerou mapa de calor com sucesso (shape {heatmap.shape})",
                    True,
                )
            else:
                print_step(
                    "Módulo gradcam.py",
                    False,
                    f"Heatmap gerou shape inesperado: {heatmap.shape if heatmap is not None else 'None'}",
                )
    except Exception as e:
        print_step("Módulo gradcam.py", False, str(e))

    print("\n" + "=" * 70)
    print("[SUCESSO] TESTES FINALIZADOS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_visual_tests()
