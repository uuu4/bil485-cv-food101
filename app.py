"""Food-101 yemek sınıflandırıcı: Gradio web uygulaması.

Yerelde:   python app.py                      (http://127.0.0.1:7860)
Paylaşım:  python app.py --share              (geçici herkese açık bağlantı)
Colab:     from app import create_demo; create_demo().launch(share=True)
"""
from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import gradio as gr
import torch

from food_cv import get_device, grad_cam, list_images, load_checkpoint, load_image, overlay_cam, predict_image, pretty, tr

DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "resnet50_food101.pt"
LOW_CONFIDENCE = 0.40


def create_demo(ckpt_path: str | Path = DEFAULT_CKPT, device: torch.device | None = None,
                examples_dir: str | Path | None = Path(__file__).parent / "my_photos") -> gr.Blocks:
    device = device or get_device()
    model, meta = load_checkpoint(ckpt_path, device=device)
    classes, img_size = meta["classes"], meta["config"].img_size

    def infer(path):
        if path is None:
            return {}, None, "Bir yemek fotoğrafı yükleyin ya da kamerayla çekin."
        img = load_image(path)
        t0 = time.perf_counter()
        preds = predict_image(model, img, classes, img_size, device, topk=5)
        ms = (time.perf_counter() - t0) * 1000
        cam, view, _ = grad_cam(model, img, img_size, device, class_idx=classes.index(preds[0][0]))
        top_name, top_p = preds[0]
        msg = f"**En olası sınıf:** {pretty(top_name)} (%{tr(100 * top_p, '.1f')})  \nÇıkarım süresi: {ms:.0f} ms ({device.type})"
        if top_p < LOW_CONFIDENCE:
            msg += ("  \n⚠️ Model emin değil. Fotoğraf Food-101'in 101 sınıfından birine ait olmayabilir "
                    "ya da birden fazla yemek içeriyor olabilir.")
        return {pretty(c): p for c, p in preds}, overlay_cam(view, cam), msg

    examples = [[str(p)] for p in list_images(examples_dir)] if examples_dir else []
    kwargs = dict(
        fn=infer,
        inputs=gr.Image(type="filepath", label="Yemek fotoğrafı", sources=["upload", "webcam", "clipboard"]),
        outputs=[
            gr.Label(num_top_classes=5, label="Tahmin (ilk 5)"),
            gr.Image(label="Grad-CAM: modelin karar verirken baktığı bölge"),
            gr.Markdown(),
        ],
        title="Food-101 Yemek Tanıma (ResNet-50)",
        description=("Bir yemek fotoğrafı yükleyin; model 101 yemek sınıfı arasından en olası 5 tanesini "
                     "ve karar verirken baktığı bölgeyi gösterir. BIL485 Derin Öğrenme, Ödev 2."),
        examples=examples or None,
    )
    params = inspect.signature(gr.Interface).parameters
    if "flagging_mode" in params:
        kwargs["flagging_mode"] = "never"
    elif "allow_flagging" in params:
        kwargs["allow_flagging"] = "never"
    return gr.Interface(**kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    create_demo(args.ckpt).launch(share=args.share, server_port=args.port)
