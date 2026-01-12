from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F

# ==========================
# Configuração de device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando device: {device}")

SCALE_CM = 32/22

# ==========================
# Carregar modelo de profundidade MiDaS DPT_Large
# ==========================
def load_midas(model_type="DPT_Large"):
    """
    Carrega MiDaS via torch.hub e o transform correto.
    model_type: 'DPT_Large', 'DPT_Hybrid' ou 'MiDaS_small'
    """
    print(f"[INFO] Carregando MiDaS ({model_type}) via torch.hub...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)  # baixa na 1ª vez
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

# ==========================
# Rodar profundidade só no crop da box
# ==========================
def run_depth_on_box(img_bgr, box_xyxy, depth_model, depth_transform):
    """
    img_bgr: frame original em BGR (uint8)
    box_xyxy: [x1, y1, x2, y2] na escala do frame original
    depth_model: MiDaS já carregado
    depth_transform: transform do MiDaS (dpt_transform ou small_transform)

    Retorna:
        depth_mean: float (profundidade média relativa do objeto)
        depth_map: np.array [H_crop, W_crop] com o mapa de profundidade upsampled
    """
    x1, y1, x2, y2 = map(int, box_xyxy)

    h, w, _ = img_bgr.shape
    # Garante limites dentro da imagem
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop_bgr = img_bgr[y1:y2, x1:x2, :]
    if crop_bgr.size == 0:
        return None, None

    # BGR -> RGB (MiDaS trabalha em RGB)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Transform do MiDaS (normaliza, redimensiona etc.)
    input_batch = depth_transform(crop_rgb).to(device)  # [1,3,h,w]

    with torch.no_grad():
        prediction = depth_model(input_batch)  # [1, h', w'] ou [1,1,h',w']

    # Garante shape [1,1,h',w']
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)  # [1,1,h',w']

    # Faz upsample para o tamanho do crop original
    pred_resized = F.interpolate(
        prediction,
        size=crop_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    )  # [1,1,H_crop,W_crop]

    depth_map = pred_resized.squeeze().cpu().numpy()  # [H_crop, W_crop]

    # Profundidade média do objeto (valor relativo)
    depth_mean = float(depth_map.mean())

    return depth_mean, depth_map

# ==========================
# Processar vídeo: YOLO + depth só nas boxes
# ==========================
def process_video(
    input_path,
    output_path,
    yolo_weights,
    conf_thresh=0.5,
    model_type="DPT_Large"
):
    # 1) Carrega YOLO
    print(f"[INFO] Carregando YOLO com pesos: {yolo_weights}")
    det_model = YOLO(yolo_weights)

    # 2) Carrega MiDaS
    depth_model, depth_transform = load_midas(model_type=model_type)

    # 3) Abre vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Vídeo: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # 4) Configura VideoWriter para salvar o resultado
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ou 'XVID' pra .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # acabou o vídeo

        frame_idx += 1

        # 5) Roda YOLO no frame (em BGR mesmo, Ultralytics trata)
        results = det_model(frame, verbose=False)[0]
        boxes = results.boxes

        # 6) Para cada box, roda depth só no crop
        for box_tensor, cls_tensor, conf_tensor in zip(boxes.xyxy, boxes.cls, boxes.conf):
            conf = float(conf_tensor.item())
            if conf < conf_thresh:
                continue

            box_xyxy = box_tensor.cpu().numpy().tolist()
            class_id = int(cls_tensor.item())

            depth_mean, depth_map = run_depth_on_box(
                frame,
                box_xyxy,
                depth_model,
                depth_transform
            )

            if depth_mean is None:
                continue

            # Desenha box e texto no frame
            x1, y1, x2, y2 = map(int, box_xyxy)

            # Azul escuro em BGR (você pode ajustar se quiser mais escuro/claro)
            box_color = (128, 0, 0)   # (B, G, R)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            text = f"id:{class_id} conf:{conf:.2f} d:{SCALE_CM*depth_mean:.2f}cm"

            # Fonte maior e mais grossa
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_color = (128, 0, 0)  # mesmo azul escuro, ou mude se quiser

            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 10)),  # sobe um pouco mais o texto
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA
            )


        # 7) Escreve frame anotado no vídeo de saída
        out.write(frame)

        if frame_idx % 10 == 0:
            print(f"[INFO] Frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"[INFO] Vídeo salvo em: {output_path}")


# ==========================
# Main com argparse
# ==========================
def main():
    parser = argparse.ArgumentParser(description="YOLO + MiDaS (profundidade nas boxes) em vídeo")
    parser.add_argument("--input", "-i", required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", "-o", required=True, help="Caminho do vídeo de saída (ex: saida.mp4)")
    parser.add_argument("--yolo-weights", "-w", default="yolo11n_2d_best_v1.pt", help="Caminho dos pesos do YOLO11n fine-tunado")
    parser.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança das detecções")
    parser.add_argument(
        "--midas-type",
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="Versão do MiDaS (DPT_Large = mais preciso, mais lento)"
    )

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        yolo_weights=args.yolo_weights,
        conf_thresh=args.conf,
        model_type=args.midas_type
    )

if __name__ == "__main__":
    main()
