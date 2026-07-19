import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from models import build_model
from utils.parser import get_args_parser


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

MODEL_CONFIGS = {
    "tiny": {
        "swin_enc_embed_dim": 96,
        "swin_enc_depths": [2, 2, 6, 2],
        "swin_enc_num_heads": [3, 6, 12, 24],
        "swin_enc_window_size": 16,
        "swin_dec_depths": [2, 6, 2, 2, 2],
        "swin_dec_num_heads": [24, 12, 6, 3, 2],
        "swin_dec_window_size": 16,
    },
    "small": {
        "swin_enc_embed_dim": 96,
        "swin_enc_depths": [2, 2, 18, 2],
        "swin_enc_num_heads": [3, 6, 12, 24],
        "swin_enc_window_size": 16,
        "swin_dec_depths": [2, 18, 2, 2, 2],
        "swin_dec_num_heads": [24, 12, 6, 3, 2],
        "swin_dec_window_size": 8,
    },
    "base": {
        "swin_enc_embed_dim": 128,
        "swin_enc_depths": [2, 2, 18, 2],
        "swin_enc_num_heads": [4, 8, 16, 32],
        "swin_enc_window_size": 8,
        "swin_dec_depths": [2, 18, 2, 2, 2],
        "swin_dec_num_heads": [32, 16, 8, 4, 2],
        "swin_dec_window_size": 8,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove scene text from an image or every image in a directory."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="ViTEraser checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output image or directory")
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="tiny", help="Checkpoint scale")
    parser.add_argument("--size", type=int, default=512, help="Square tile/model input size")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles")
    parser.add_argument(
        "--preserve-threshold",
        type=float,
        default=8.0,
        help="Keep original pixels changed by less than this value (0-255)",
    )
    parser.add_argument("--fast", action="store_true", help="Resize the whole image instead of tiling")
    parser.add_argument("--device", default="cuda", help="PyTorch device, for example cuda or cpu")
    return parser.parse_args()


def collect_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    images = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")
    return images


def output_path_for(image_path, input_path, output_path):
    if input_path.is_file() and output_path.suffix.lower() in IMAGE_EXTENSIONS:
        return output_path
    relative_path = image_path.name if input_path.is_file() else image_path.relative_to(input_path)
    return (output_path / relative_path).with_suffix(".png")


def build_viteraser(args):
    model_args = get_args_parser().parse_args([])
    model_args.eval = True
    model_args.distributed = False
    model_args.gpu = 0
    model_args.device = args.device
    model_args.pred_mask = False
    model_args.intermediate_erase = False
    model_args.pretrained_model = str(args.checkpoint)
    for name, value in MODEL_CONFIGS[args.model].items():
        setattr(model_args, name, value)

    model, _ = build_model(model_args)
    model.eval()
    return model


def _tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    end = length - tile_size
    overlap = tile_size - stride
    positions = [0]
    while end - positions[-1] > stride + overlap / 4:
        positions.append(positions[-1] + stride)
    if positions[-1] != end:
        positions.append(end)
    return positions


def _protect_original(original, prediction, threshold):
    if threshold <= 0:
        return prediction

    delta = (prediction - original).abs().mean(dim=0, keepdim=True)
    threshold = threshold / 255.0
    mask = ((delta - threshold) / max(threshold, 1e-6)).clamp(0, 1)
    mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=21, stride=1, padding=10).squeeze(0)
    mask = TF.gaussian_blur(mask, kernel_size=[31, 31], sigma=[7.0, 7.0]).clamp(0, 1)
    return original * (1 - mask) + prediction * mask


def infer_image_fast(model, image, size=512, device="cuda", preserve_threshold=8.0):
    image = image.convert("RGB")
    original_size = image.size
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    original = TF.to_tensor(resized)

    with torch.inference_mode():
        output = model(original.unsqueeze(0).to(device))[-1].squeeze(0).clamp(0, 1).cpu()

    output = _protect_original(original, output, preserve_threshold)
    return TF.to_pil_image(output).resize(original_size, Image.Resampling.BICUBIC)


def infer_image(
    model,
    image,
    size=512,
    device="cuda",
    overlap=128,
    preserve_threshold=8.0,
    tiled=True,
):
    if not tiled:
        return infer_image_fast(model, image, size, device, preserve_threshold)
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be at least 0 and smaller than the tile size")

    image = image.convert("RGB")
    original = TF.to_tensor(image)
    _, height, width = original.shape
    padded_height = max(height, size)
    padded_width = max(width, size)
    padded = F.pad(original, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    stride = size - overlap
    y_positions = _tile_positions(padded_height, size, stride)
    x_positions = _tile_positions(padded_width, size, stride)
    window_1d = torch.hann_window(size, periodic=False).clamp_min(0.05)
    window = torch.outer(window_1d, window_1d).unsqueeze(0)
    accumulated = torch.zeros_like(padded)
    weights = torch.zeros((1, padded_height, padded_width), dtype=padded.dtype)

    for y in y_positions:
        for x in x_positions:
            tile = padded[:, y:y + size, x:x + size]
            with torch.inference_mode():
                prediction = model(tile.unsqueeze(0).to(device))[-1].squeeze(0).clamp(0, 1).cpu()
            prediction = _protect_original(tile, prediction, preserve_threshold)
            accumulated[:, y:y + size, x:x + size] += prediction * window
            weights[:, y:y + size, x:x + size] += window

    output = (accumulated / weights.clamp_min(1e-6))[:, :height, :width].clamp(0, 1)
    return TF.to_pil_image(output)


def main():
    args = parse_args()
    if args.size <= 0 or args.size % 32 != 0:
        raise ValueError("--size must be a positive multiple of 32")
    if args.overlap < 0 or args.overlap >= args.size:
        raise ValueError("--overlap must be at least 0 and smaller than --size")
    if not 0 <= args.preserve_threshold <= 255:
        raise ValueError("--preserve-threshold must be between 0 and 255")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    image_paths = collect_images(args.input)
    model = build_viteraser(args)

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            result = infer_image(
                model,
                image,
                args.size,
                args.device,
                args.overlap,
                args.preserve_threshold,
                tiled=not args.fast,
            )
        save_path = output_path_for(image_path, args.input, args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(save_path)
        print(f"[{index}/{len(image_paths)}] {image_path} -> {save_path}")


if __name__ == "__main__":
    main()
