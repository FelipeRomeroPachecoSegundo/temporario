from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    model = YOLO(r"runs/detect/train4/weights/best.pt")
    src = r"./video_teste/video_teste2.mp4"

    # --- Pegamos FPS e resolução só uma vez ---
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Erro ao abrir vídeo de entrada.")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ---- AJUSTE DE TAMANHO (opcional) ----
    scale = 0.5  # 50% da resolução original
    out_w, out_h = int(width * scale), int(height * scale)

    # Codec MP4 (mais comprimido e compatível)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("video_teste_yolo2.mp4", fourcc, fps, (out_w, out_h))

    # --- O próprio YOLO faz o streaming de frames (stream=True) ---
    results_generator = model.predict(
        source=src,
        imgsz=(640, 360),
        conf=0.25,
        device="0",        # se não tiver GPU, pode trocar para "cpu"
        stream=True,       # NÃO carrega tudo na memória
        verbose=False,
        show=False         # não abre janela
    )

    for r in results_generator:
        # Desenha as boxes
        annotated = r.plot()

        # Redimensiona
        if scale != 1.0:
            annotated = cv2.resize(annotated, (out_w, out_h))

        # Salva frame no vídeo de saída
        out.write(annotated)

    out.release()
    print("Vídeo salvo como video_teste_yolo.mp4")
