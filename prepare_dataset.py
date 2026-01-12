# prepare_dataset.py
import os, shutil, random
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW_IMAGES = BASE / "dataset_raw" / "images"
RAW_LABELS = BASE / "dataset_raw" / "labels"

OUT = BASE / "dataset_recicle"
IM_TRAIN = OUT / "images" / "train"
IM_VAL   = OUT / "images" / "val"
LB_TRAIN = OUT / "labels" / "train"
LB_VAL   = OUT / "labels" / "val"

def main():
    assert RAW_IMAGES.exists(), f"Pasta não encontrada: {RAW_IMAGES}"
    assert RAW_LABELS.exists(), f"Pasta não encontrada: {RAW_LABELS}"

    # cria estrutura
    for p in [IM_TRAIN, IM_VAL, LB_TRAIN, LB_VAL]:
        p.mkdir(parents=True, exist_ok=True)

    # lista imagens jpg/jpeg/png
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in sorted(RAW_IMAGES.iterdir()) if p.suffix.lower() in exts]

    # filtra apenas as que têm label com mesmo nome (ou cria label vazio se quiser negativos explícitos)
    paired = []
    missing = []
    for img in imgs:
        lbl = RAW_LABELS / (img.stem + ".txt")
        if lbl.exists():
            paired.append((img, lbl))
        else:
            # Se quiser considerar negativos, descomente as 2 linhas abaixo:
            # lbl.touch(exist_ok=True)
            # paired.append((img, lbl))
            missing.append(img.name)

    if missing:
        print(f"Atenção: {len(missing)} imagens sem label correspondente (ex.: {missing[:5]}).")
        print("Rotule essas imagens ou habilite 'negativos' no script (ver comentário).")

    if not paired:
        raise SystemExit("Nenhuma imagem com label correspondente. Rotule antes de prosseguir.")

    # split 80/20
    random.seed(42)
    random.shuffle(paired)
    n = len(paired)
    n_train = int(0.9 * n)
    train_set = paired[:n_train]
    val_set   = paired[n_train:]

    # copia
    def cp(pairs, im_dst, lb_dst):
        for im, lb in pairs:
            shutil.copy2(im, im_dst / im.name)
            shutil.copy2(lb, lb_dst / lb.name)

    cp(train_set, IM_TRAIN, LB_TRAIN)
    cp(val_set,   IM_VAL,   LB_VAL)

    print(f"Pronto! Treino: {len(train_set)} | Val: {len(val_set)}")
    print(f"Estrutura criada em: {OUT}")

if __name__ == "__main__":
    main()
