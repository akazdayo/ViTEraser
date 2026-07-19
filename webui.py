import argparse
import gc
import threading
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch

from inference import build_viteraser, infer_image


DEFAULT_CHECKPOINTS = {
    "tiny": Path("pretrained/viteraser_tiny.pth"),
    "small": Path("pretrained/viteraser_small.pth"),
    "base": Path("pretrained/viteraser_base.pth"),
}


class ModelCache:
    def __init__(self):
        self.model = None
        self.key = None
        self.lock = threading.Lock()

    def get(self, model_name, checkpoint, device):
        checkpoint = Path(checkpoint).expanduser().resolve()
        key = (model_name, checkpoint, device)
        with self.lock:
            if self.key != key:
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"重みが見つかりません: {checkpoint}")
                self.model = None
                self.key = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                self.model = build_viteraser(
                    SimpleNamespace(
                        model=model_name,
                        checkpoint=checkpoint,
                        device=device,
                    )
                )
                self.key = key
            return self.model


MODEL_CACHE = ModelCache()


def default_checkpoint(model_name):
    return str(DEFAULT_CHECKPOINTS[model_name])


def erase_text(
    image,
    model_name,
    checkpoint,
    input_size,
    high_quality,
    overlap,
    preserve_threshold,
):
    if image is None:
        raise gr.Error("画像をアップロードしてください。")
    if input_size % 32 != 0:
        raise gr.Error("入力サイズは32の倍数にしてください。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = MODEL_CACHE.get(model_name, checkpoint, device)
        result = infer_image(
            model,
            image,
            int(input_size),
            device,
            int(overlap),
            float(preserve_threshold),
            tiled=high_quality,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    device_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    mode = "高画質タイル" if high_quality else "高速リサイズ"
    status = f"完了 — {model_name.title()} / {device_name} / {mode} / {int(input_size)}px"
    return result, status


def create_demo():
    with gr.Blocks(title="ViTEraser WebUI") as demo:
        gr.Markdown(
            "# ViTEraser\n"
            "画像をアップロードして、シーン内の文字を除去します。出力画像は元の解像度で保存できます。"
        )

        with gr.Row():
            input_image = gr.Image(
                type="pil",
                image_mode="RGB",
                label="入力画像",
                sources=["upload", "clipboard"],
                height=520,
            )
            output_image = gr.Image(
                type="pil",
                image_mode="RGB",
                format="png",
                label="文字除去結果",
                interactive=False,
                height=520,
            )

        with gr.Accordion("推論設定", open=False):
            model_name = gr.Dropdown(
                choices=["tiny", "small", "base"],
                value="base",
                label="モデル",
            )
            checkpoint = gr.Textbox(
                value=str(DEFAULT_CHECKPOINTS["base"]),
                label="チェックポイント",
            )
            input_size = gr.Slider(
                minimum=256,
                maximum=512,
                value=512,
                step=32,
                label="タイル／モデル入力サイズ",
            )
            high_quality = gr.Checkbox(
                value=True,
                label="高画質タイル処理（元解像度を維持）",
            )
            overlap = gr.Slider(
                minimum=0,
                maximum=224,
                value=128,
                step=32,
                label="タイルの重なり",
            )
            preserve_threshold = gr.Slider(
                minimum=0,
                maximum=32,
                value=8,
                step=1,
                label="元画像の保護強度（高いほど非文字領域を維持）",
            )

        run_button = gr.Button("文字を除去", variant="primary")
        status = gr.Textbox(label="状態", interactive=False)

        model_name.change(
            fn=default_checkpoint,
            inputs=model_name,
            outputs=checkpoint,
            queue=False,
        )
        run_button.click(
            fn=erase_text,
            inputs=[
                input_image,
                model_name,
                checkpoint,
                input_size,
                high_quality,
                overlap,
                preserve_threshold,
            ],
            outputs=[output_image, status],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the ViTEraser Gradio WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
