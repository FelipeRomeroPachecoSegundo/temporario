import cv2
from ultralytics import YOLO
from collections import Counter

# ===================== CONFIGURAÇÕES ===================== #
MODEL_PATH = "runs/detect/train/weights/best.pt"   # caminho do seu modelo YOLO11x treinado
VIDEO_PATH = "video_teste/video_teste2.mp4"                       # caminho do vídeo de entrada
OUTPUT_PATH = "video_saida_rastreado.mp4"            # caminho do vídeo de saída (salvo)
DISAPPEAR_TIME_SEC = 2.0                             # tempo em segundos para considerar que o objeto sumiu
CONF_THRESHOLD = 0.3                                  # limiar de confiança para detecções
SHOW_CLASSES = None                                  # None = todas as classes, ou lista de IDs ex: [0] para só uma classe

# tolerância em pixels no eixo X para considerar que é o mesmo objeto
MAX_MATCH_DX = 80.0
# ========================================================= #


def get_mode(values):
    """
    Retorna a moda (valor mais frequente) de uma lista, ou None se estiver vazia.
    Aqui usamos os valores como float mesmo.
    """
    if not values:
        return None
    counter = Counter(values)
    mode_value, _ = counter.most_common(1)[0]
    return mode_value


def main():
    # Carrega modelo
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {VIDEO_PATH}")
        return

    # FPS do vídeo de entrada
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    # Dimensões do vídeo de entrada
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cria o VideoWriter para salvar o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec para .mp4
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    max_missing_frames = int(DISAPPEAR_TIME_SEC * fps)
    frame_idx = 0

    # tracks:
    # {track_id: {
    #     "center": (cx, y_ref),
    #     "y_ref": float,
    #     "last_seen_frame": int,
    #     "y_values": [float]
    # }}
    tracks = {}
    next_track_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break

        frame_idx += 1

        # ---------------- INFERÊNCIA YOLO ---------------- #
        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        # Lista de detecções: (cx, cy, x1, y1, x2, y2, conf, cls)
        detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < CONF_THRESHOLD:
                    continue
                if SHOW_CLASSES is not None and cls not in SHOW_CLASSES:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append((cx, cy, x1, y1, x2, y2, conf, cls))

        # ------------- ASSOCIAÇÃO DETECÇÃO ↔ TRACK ------------- #
        assigned_dets = set()
        det_to_track = {}  # idx_det -> track_id

        for track_id, track in tracks.items():
            prev_cx, prev_y_const = track["center"]

            best_idx = None
            best_dx = None

            for i, det in enumerate(detections):
                if i in assigned_dets:
                    continue
                cx_det, cy_det = det[0], det[1]

                dx = abs(cx_det - prev_cx)  # diferença só em X

                if best_dx is None or dx < best_dx:
                    best_dx = dx
                    best_idx = i

            # Se a diferença em X for pequena, considera a mesma track
            if best_idx is not None and best_dx <= MAX_MATCH_DX:
                assigned_dets.add(best_idx)
                det_to_track[best_idx] = track_id

        # ------------- CRIA NOVOS TRACKS (OBJETOS NOVOS) ------------- #
        new_dets_indices = [i for i in range(len(detections)) if i not in assigned_dets]
        new_dets_indices.sort(key=lambda i: detections[i][1])  # ordenar por cy (y do centro)

        for i in new_dets_indices:
            cx, cy, x1, y1, x2, y2, conf, cls = detections[i]
            track_id = next_track_id
            next_track_id += 1

            y_ref = cy  # y "constante" que vamos usar para desenhar

            tracks[track_id] = {
                "center": (cx, y_ref),
                "y_ref": y_ref,
                "last_seen_frame": frame_idx,
                "y_values": [cy],
            }
            det_to_track[i] = track_id

            print(
                f"[FRAME {frame_idx:05d}] Novo objeto ID {track_id} "
                f"→ centro inicial: ({cx:.1f}, {y_ref:.1f})"
            )

        # ------------- ATUALIZA TRACKS EXISTENTES ------------- #
        for i, det in enumerate(detections):
            if i not in det_to_track:
                continue

            cx_det, cy_det, x1, y1, x2, y2, conf, cls = det
            track_id = det_to_track[i]
            track = tracks[track_id]

            y_ref = track["y_ref"]  # mantém y constante para desenhar
            track["center"] = (cx_det, y_ref)
            track["last_seen_frame"] = frame_idx
            track["y_values"].append(cy_det)

            print(
                f"[FRAME {frame_idx:05d}] ID {track_id} "
                f"→ centro: ({cx_det:.1f}, {cy_det:.1f}), n_amostras_y = {len(track['y_values'])}"
            )

            # Desenha bbox e info no frame (bbox real, ponto com y_ref)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx_det), int(y_ref)), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(x1), int(y1) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # ------------- VERIFICA QUEM SAIU DE CENA ------------- #
        tracks_to_delete = []
        for track_id, track in list(tracks.items()):
            frames_missing = frame_idx - track["last_seen_frame"]
            if frames_missing > max_missing_frames:
                mode_y = get_mode(track["y_values"])
                print(
                    f"[FRAME {frame_idx:05d}] Objeto ID {track_id} SAIU DE CENA "
                    f"(> {DISAPPEAR_TIME_SEC}s sem ver).\n"
                    f"  → Moda final de y = {mode_y:.3f}\n"
                )
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            del tracks[track_id]

        # ------------- SALVA FRAME NO VÍDEO ------------- #
        out.write(frame)  # grava o frame já com desenhos

        # ------------- MOSTRA VÍDEO ------------- #
        cv2.imshow("YOLO11x - Rastreamento de materiais (multi-objetos)", frame)

        # tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # libera o arquivo de vídeo
    cv2.destroyAllWindows()
    print(f"Vídeo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
