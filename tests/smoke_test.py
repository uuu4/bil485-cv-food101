"""Duman testi: Food-101 indirmeden, sahte küçük bir veri setiyle tüm akışı CPU'da sınar.

    python tests/smoke_test.py

Sınananlar: bölme, iki modelin birkaç adım eğitimi, metrikler, checkpoint kaydet/yükle (fp16),
tahmin, Grad-CAM, EXIF yönü düzeltme, HEIC okuma, dosya adından etiket, rapor üretimi, Gradio arayüzü.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import food_cv as fc  # noqa: E402

CLASSES = ["french_fries", "pizza", "steak"]
COLORS = {"french_fries": (230, 190, 40), "pizza": (200, 60, 40), "steak": (100, 50, 30)}


def make_fake_dataset(root: Path, per_class: int):
    files, labels = [], []
    rng = np.random.default_rng(0)
    for ci, c in enumerate(CLASSES):
        for k in range(per_class):
            arr = np.clip(np.array(COLORS[c]) + rng.normal(0, 25, (96, 128, 3)), 0, 255).astype(np.uint8)
            p = root / c / f"{k}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p, quality=90)
            files.append(str(p))
            labels.append(ci)
    return files, labels


def main():
    fc.seed_everything(0)
    device = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp())
    files, labels = make_fake_dataset(tmp / "data", per_class=20)

    # bölme
    tr_idx, va_idx = fc.stratified_split(labels, 5, seed=0)
    assert len(va_idx) == 15 and len(tr_idx) == 45 and not set(tr_idx) & set(va_idx)
    assert sorted(np.bincount([labels[i] for i in va_idx]).tolist()) == [5, 5, 5]
    sub = fc.subsample_per_class(tr_idx, labels, 3, seed=0)
    assert len(sub) == 9

    results = {"project_title": "test", "repository": "https://example.org/repo", "seed": 0,
               "environment": fc.environment_info(), "split": {"train": 45, "val": 15, "test": 15, "val_per_class": 5},
               "models": {}}
    for cfg in [fc.TrainConfig(name="smallcnn", arch="smallcnn", img_size=64, epochs=2, batch_size=8, lr=2e-3,
                               backbone_lr_mult=1.0, pretrained=False, dropout=0.3),
                fc.TrainConfig(name="resnet50", arch="resnet50", img_size=64, epochs=1, batch_size=8, pretrained=False)]:
        model = fc.build_model(cfg.arch, len(CLASSES), pretrained=False, dropout=cfg.dropout)
        tr = fc.ImageListDataset([files[i] for i in tr_idx], [labels[i] for i in tr_idx], fc.train_transform(cfg.img_size))
        va = fc.ImageListDataset([files[i] for i in va_idx], [labels[i] for i in va_idx], fc.eval_transform(cfg.img_size))
        trl = fc.make_loader(tr, cfg.batch_size, True, 0, drop_last=True)
        val = fc.make_loader(va, 16, False, 0)
        history, best_epoch, ttime = fc.fit(model, trl, val, cfg, device, max_train_batches=3, log_every=2)
        assert len(history) == cfg.epochs and 1 <= best_epoch <= cfg.epochs
        assert history[-1]["lr_head"] < cfg.lr, "kosinüs takvimi lr'yi düşürmeli"

        out = fc.evaluate(model, val, device, return_outputs=True)
        assert out["logits"].shape == (15, 3) and out["n"] == 15
        m = fc.classification_metrics(out["logits"], out["labels"], CLASSES)
        assert abs(m["accuracy"] - out["acc"]) < 1e-9 and m["top5_accuracy"] == 1.0  # 3 sınıfta top-5 her zaman doğru
        assert np.asarray(m["confusion_matrix"]).sum() == 15

        ck = fc.save_checkpoint(model, cfg, CLASSES, tmp / "ck" / f"{cfg.name}_food101.pt",
                                extra={"history": history, "best_epoch": best_epoch, "train_time_s": ttime,
                                       "best_val_acc": max(h["val_acc"] for h in history)})
        model2, meta = fc.load_checkpoint(ck)
        assert meta["config"] == cfg and meta["classes"] == CLASSES and meta["extra"]["best_epoch"] == best_epoch
        out2 = fc.evaluate(model2, val, device, return_outputs=True)
        assert (out2["logits"].argmax(1) == out["logits"].argmax(1)).mean() >= 0.9, "fp16 checkpoint tahminleri değiştirmemeli"

        img = fc.load_image(files[0])
        preds = fc.predict_image(model2, img, CLASSES, cfg.img_size, device)
        assert len(preds) == 3 and abs(sum(p for _, p in preds) - 1) < 1e-4
        cam, view, ci = fc.grad_cam(model2, img, cfg.img_size, device)
        assert cam.shape == (cfg.img_size, cfg.img_size) and 0 <= cam.min() and cam.max() <= 1.0 + 1e-6
        assert view.size == (cfg.img_size, cfg.img_size) and ci == CLASSES.index(preds[0][0])
        assert fc.overlay_cam(view, cam).size == view.size
        assert all(p.grad is None for p in model2.parameters()), "Grad-CAM gradyan bırakmamalı"
        ms = fc.time_inference(model2, device, cfg.img_size, n_warmup=1, n_runs=3)
        assert ms > 0

        results["models"][cfg.name] = {
            "architecture": cfg.arch, "config": fc.asdict(cfg), "params_m": fc.count_parameters(model) / 1e6,
            "history": history, "best_epoch": best_epoch, "best_val_acc": max(h["val_acc"] for h in history),
            "train_time_s": ttime, "test": m, "inference_ms": {"gpu_ms": None, "cpu_ms": ms},
        }

    # telefon fotoğrafları: EXIF yönü, HEIC, dosya adından etiket
    photos = tmp / "my_photos"
    photos.mkdir()
    tall = np.zeros((120, 60, 3), np.uint8)
    tall[:20] = 255  # üst şerit beyaz
    exif = Image.Exif()
    exif[0x0112] = 6  # 90 derece döndürülmüş kaydedilmiş
    Image.fromarray(tall).save(photos / "pizza.jpg", exif=exif)
    fixed = fc.load_image(photos / "pizza.jpg")
    assert fixed.size == (120, 60), f"EXIF yönü düzeltilmedi: {fixed.size}"
    assert fc.HEIC_OK, "pillow-heif kurulu değil"
    Image.fromarray(np.full((80, 80, 3), 120, np.uint8)).save(photos / "Mercimek Corbasi-2.heic")
    assert fc.load_image(photos / "Mercimek Corbasi-2.heic").size == (80, 80)
    assert fc.label_from_filename("french_fries-2.HEIC") == "french_fries"
    assert fc.label_from_filename("Mercimek Corbasi-2.heic") == "mercimek_corbasi"
    rows = fc.evaluate_my_photos(model2, photos, CLASSES, 64, device)
    assert [r["file"] for r in rows] == ["Mercimek Corbasi-2.heic", "pizza.jpg"]
    conv_dir = tmp / "conv"
    conv_dir.mkdir()
    Image.fromarray(np.full((50, 70, 3), 90, np.uint8)).save(conv_dir / "baklava.HEIC")
    conv = fc.convert_heic_to_jpeg(conv_dir)
    assert [p.name for p in conv] == ["baklava.jpg"] and [p.name for p in fc.list_images(conv_dir)] == ["baklava.jpg"]
    assert Image.open(conv[0]).size == (70, 50)
    assert rows[1]["in_food101"] and not rows[0]["in_food101"] and rows[0]["correct_top1"] is False

    # rapor
    results["selected_model"] = "resnet50"
    results["my_photos"] = rows
    fc.save_json(results, tmp / "results.json")
    r = json.loads((tmp / "results.json").read_text(encoding="utf-8"))
    table = fc.report_table(r)
    assert len(table) == 16 and table[0][0] == "Project Title" and table[-1] == ("Repository", "https://example.org/repo")
    rep = fc.write_report(r, tmp / "REPORT.md").read_text(encoding="utf-8")
    assert "| Best Performance |" in rep and "Mercimek corbasi" in rep and "sınıf dışı" in rep

    # Gradio arayüzü: kurulur ve tahmin fonksiyonu çalışır
    import app

    demo = app.create_demo(tmp / "ck" / "resnet50_food101.pt", device=device, examples_dir=photos)
    labels_out, overlay, msg = demo.fn(str(photos / "pizza.jpg"))
    assert len(labels_out) == 3 and overlay.size == (64, 64) and "En olası sınıf" in msg
    assert demo.fn(None)[0] == {}

    print("\nDUMAN TESTİ: TÜMÜ GEÇTİ")
    print(rep)


if __name__ == "__main__":
    main()
