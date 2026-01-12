# train_wallet.py
from ultralytics import YOLO

# Escolha o tamanho do modelo:
# 'yolov8n.pt' (nano) é mais rápido e bom para começar; depois você pode tentar 'yolov8s.pt'.
MODEL_WEIGHTS = "yolo11n.pt"

# Desabilitando augmentations (o Ultralytics faz algumas por padrão; aqui zeramos as principais)
AUG_OFF = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
    perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
)

if __name__ == "__main__":
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data="data_metal.yaml",
        epochs=300,          # com 113 imagens, 50–150 épocas costuma ajudar
        imgsz=(640,360),
        batch=-1,            # ajuste conforme sua GPU/CPU
        device="0",            # 0 para primeira GPU; use "cpu" se não tiver GPU
        workers=0,
        patience=50,         # early stopping
        lr0=0.01,            # taxa de aprendizado inicial padrão
        weight_decay=0.0005,
        **AUG_OFF
    )

    # Avaliação final (mAP etc.) no conjunto de validação
    model.val(data="data_metal.yaml", imgsz=640)
