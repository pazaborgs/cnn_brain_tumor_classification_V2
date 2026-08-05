"""Módulo de IA Explicável (XAI) — Grad-CAM (Gradient-weighted Class Activation Mapping).

Fornece explicações visuais para as decisões dos modelos de Aprendizado Profundo,
calculando os gradientes da pontuação de classe em relação às camadas convolucionais.
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf


def prepare_image(
    img_path: str, target_shape: tuple[int, int] = (299, 299)
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega e pré-processa uma imagem para inferência e geração do Grad-CAM.

    Args:
        img_path: Caminho do arquivo da imagem de entrada.
        target_shape: Resolução espacial alvo (largura, altura) esperada pelo modelo.

    Returns:
        Uma tupla contendo:
            - img_rgb: Matriz da imagem RGB original (uint8, formato [H, W, 3]).
            - img_tensor: Tensor preparado para entrada do modelo (float32, formato [1, H, W, 3]).
    """
    pil_img = PIL.Image.open(img_path).convert("RGB")
    img_rgb = np.array(pil_img)
    pil_resized = pil_img.resize(target_shape)
    img_tensor = np.expand_dims(np.array(pil_resized), axis=0).astype(np.float32)
    return img_rgb, img_tensor


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray | None:
    """Gera o mapa de calor 2D normalizado do Grad-CAM para a classe alvo.

    Executa a propagação via GradientTape isolando a atenção visual baseada
    na última camada convolucional fornecida explicitamente pelas configurações.

    Args:
        img_array: Tensor de entrada pré-processado (formato [1, H, W, C]).
        model: Instância do modelo de classificação Keras treinado.
        last_conv_layer_name: Nome explícito obrigatório da camada convolucional alvo.
        pred_index: Índice opcional da classe alvo. O padrão é a classe de maior probabilidade.

    Returns:
        Matriz bidimensional float32 normalizada entre [0, 1], ou None se a camada não for encontrada.
    """
    if model is None or not last_conv_layer_name:
        print("[AVISO] Grad-CAM: Modelo ou last_conv_layer_name ausente.")
        return None

    try:
        backbone = None
        target_layer = None

        for layer in model.layers:
            if hasattr(layer, "layers") and layer.name != "augmentation":
                backbone = layer
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    pass

        if target_layer is None:
            backbone = model
            try:
                target_layer = model.get_layer(last_conv_layer_name)
            except ValueError:
                pass

        if target_layer is None:
            print(f"[ERRO] Grad-CAM: Camada alvo '{last_conv_layer_name}' não encontrada na arquitetura.")
            return None

        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        try:
            backbone_grad_model = tf.keras.models.Model(
                inputs=backbone.inputs,
                outputs=[target_layer.output, backbone.output],
            )
            with tf.GradientTape() as tape:
                x_prep = img_tensor
                for l in model.layers:
                    if l == backbone:
                        break
                    if l.name not in ("input_layer", "augmentation"):
                        x_prep = l(x_prep)

                conv_outputs, backbone_pooled = backbone_grad_model(x_prep)
                tape.watch(conv_outputs)

                x = backbone_pooled
                backbone_idx = model.layers.index(backbone)
                for l in model.layers[backbone_idx + 1 :]:
                    x = l(x)
                preds = x

                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                score = preds[:, pred_index]

            grads = tape.gradient(score, conv_outputs)
        except Exception as e:
            print(f"[ERRO] Grad-CAM: Falha ao calcular gradientes: {e}")
            grads = None
            conv_outputs = None

        if grads is None or conv_outputs is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()
    except Exception:
        return None


def overlay_gradcam(
    img_rgb: np.ndarray,
    heatmap: np.ndarray | None = None,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Superpõe o mapa de calor 2D do Grad-CAM sobre a imagem RGB original.

    Args:
        img_rgb: Matriz da imagem RGB original (uint8, formato [H, W, 3]).
        heatmap: Matriz de ativação normalizada 2D float32 no intervalo [0, 1].
        alpha: Peso de transparência para a sobreposição do mapa térmico em [0, 1].
        colormap: Identificador do mapa de cores do Matplotlib (ex: 'jet', 'hot').

    Returns:
        Matriz da imagem RGB com a sobreposição do mapa de calor (uint8, formato [H, W, 3]).
    """
    h, w = img_rgb.shape[:2]

    if heatmap is None or np.all(heatmap == 0):
        return img_rgb

    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(colormap)
    cmap_colors = cmap(np.arange(256))[:, :3]
    colored_heatmap = cmap_colors[heatmap_uint8]

    heatmap_pil = PIL.Image.fromarray(np.uint8(255 * colored_heatmap))
    heatmap_pil_resized = heatmap_pil.resize((w, h))
    heatmap_resized_array = np.array(heatmap_pil_resized).astype(np.float32)

    img_array_float = img_rgb.astype(np.float32)
    superimposed = heatmap_resized_array * alpha + img_array_float * (1.0 - alpha)

    return np.uint8(np.clip(superimposed, 0, 255))
