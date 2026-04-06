import os
from pathlib import Path
import random
import shutil

def split_data(root: Path, val_frac: float = 0.2, force: bool = False) -> None:
    src = root / "full_train"
    dst_train_img = root / "train" / "images"
    dst_train_lbl = root / "train" / "labels"
    dst_val_img = root / "val" / "images"
    dst_val_lbl = root / "val" / "labels"

    if not force and (dst_train_img.exists() and dst_train_lbl.exists() and dst_val_img.exists() and dst_val_lbl.exists()):
        print("Data already split. Use force=True to split again.")
        return
    else:
        shutil.rmtree(dst_train_img.parent, ignore_errors=True)
        shutil.rmtree(dst_val_img.parent, ignore_errors=True)

    for p in (dst_train_img, dst_train_lbl, dst_val_img, dst_val_lbl):
        p.mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in (src / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    random.seed(42)
    random.shuffle(stems)

    n_val = int(len(stems) * val_frac)
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    def relocate(stem: str, split: str) -> None:
        img_dir, lbl_dir = (dst_train_img, dst_train_lbl) if split == "train" else (dst_val_img, dst_val_lbl)
        img = next((src / "images" / f"{stem}{suf}" for suf in (".jpg", ".jpeg", ".png", ".bmp", ".webp") if (src / "images" / f"{stem}{suf}").exists()), None)
        lbl = src / "labels" / f"{stem}.txt"
        if img is None or not lbl.is_file():
            raise FileNotFoundError(f"missing pair for {stem}")
        shutil.move(str(img), img_dir / img.name)   # use shutil.copy2 to keep full_train
        shutil.move(str(lbl), lbl_dir / lbl.name)

    for s in train_stems:
        relocate(s, "train")
    for s in val_stems:
        relocate(s, "val")