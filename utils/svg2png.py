import argparse
import glob
import os
import shutil
import subprocess
import sys
from typing import List, Optional

# Global flags set after a single dependency detection so we don't
# spam 100+ error lines when dependencies are missing.
_CAIROSVG_AVAILABLE: bool = False
_INKSCAPE_PATH: Optional[str] = None
_CAIROSVG_IMPORT_ERROR: Optional[str] = None  # 保存导入失败的详细原因


def list_svgs(svg_dir: str) -> List[str]:
    pattern = os.path.join(svg_dir, "*.svg")
    files = glob.glob(pattern)
    files += glob.glob(os.path.join(svg_dir, "*.SVG"))
    return sorted(files)


def ensure_outdir(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)


def convert_with_cairosvg(svg_path: str, png_path: str, scale: Optional[float] = None) -> None:
    if not _CAIROSVG_AVAILABLE:
        raise RuntimeError("CairoSVG is not available.")
    # Import locally after detection to avoid global hard failure.
    import cairosvg  # type: ignore
    kwargs = {"url": svg_path, "write_to": png_path}
    if scale is not None and scale > 0:
        kwargs["scale"] = scale
    cairosvg.svg2png(**kwargs)


def _which_inkscape() -> Optional[str]:
    # On Windows, inkscape is commonly available as inkscape.exe or inkscape.com
    for candidate in ("inkscape", "inkscape.exe", "inkscape.com"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def convert_with_inkscape(svg_path: str, png_path: str, scale: Optional[float] = None) -> None:
    if not _INKSCAPE_PATH:
        raise RuntimeError("Inkscape is not available.")
    cmd = [_INKSCAPE_PATH, svg_path, "-o", png_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inkscape failed to convert '{svg_path}' -> '{png_path}'.") from e


def convert_svg_to_png(svg_path: str, png_path: str, scale: Optional[float]) -> None:
    if _CAIROSVG_AVAILABLE:
        convert_with_cairosvg(svg_path, png_path, scale)
    elif _INKSCAPE_PATH:
        convert_with_inkscape(svg_path, png_path, scale)
    else:
        raise RuntimeError(
            "No converter available. Install CairoSVG (`uv add cairosvg`) or install Inkscape and put it on PATH."
        )


def _detect_dependencies() -> None:
    global _CAIROSVG_AVAILABLE, _INKSCAPE_PATH, _CAIROSVG_IMPORT_ERROR
    # 尝试导入 CairoSVG，只在真正 ImportError 时标记缺失；其他异常保留错误信息方便排查
    try:
        import cairosvg  # type: ignore  # noqa: F401
        _CAIROSVG_AVAILABLE = True
    except ImportError as exc:
        _CAIROSVG_AVAILABLE = False
        _CAIROSVG_IMPORT_ERROR = f"ImportError: {exc}"
    except Exception as exc:  # 可能是缺少本地 cairo DLL 等
        _CAIROSVG_AVAILABLE = False
        _CAIROSVG_IMPORT_ERROR = f"其它异常: {type(exc).__name__}: {exc}"

    _INKSCAPE_PATH = _which_inkscape()
    if not _CAIROSVG_AVAILABLE and not _INKSCAPE_PATH:
        detail = _CAIROSVG_IMPORT_ERROR or "未知原因"
        raise RuntimeError(
            "未检测到 CairoSVG 或 Inkscape。CairoSVG 导入失败原因: " + detail +
            "。请使用 `uv add cairosvg` 安装，或安装 Inkscape 并加入 PATH。"
        )


def main(args: argparse.Namespace) -> None:
    svg_dir = args.svg_dir
    save_dir = args.save_dir
    scale = args.scale

    ensure_outdir(save_dir)

    # Detect dependencies once up-front to avoid repeated noisy failures.
    try:
        _detect_dependencies()
    except RuntimeError as dep_err:
        print(f"依赖检测失败: {dep_err}", file=sys.stderr)
        return

    svgs = list_svgs(svg_dir)
    if not svgs:
        print(f"No SVG files found in '{svg_dir}'. Nothing to do.")
        return

    total = len(svgs)
    print(f"Found {total} SVG file(s) in '{svg_dir}'. Converting to '{save_dir}'...")

    converted = 0
    for i, svg_path in enumerate(svgs, start=1):
        base = os.path.splitext(os.path.basename(svg_path))[0]
        png_path = os.path.join(save_dir, f"{base}.png")
        try:
            convert_svg_to_png(svg_path, png_path, scale)
            converted += 1
            print(f"[{i}/{total}] ✓ {os.path.basename(svg_path)} -> {os.path.basename(png_path)}")
        except Exception as e:
            print(f"[{i}/{total}] ✗ Failed to convert '{svg_path}': {e}", file=sys.stderr)

    print(f"Done. Converted {converted}/{total} file(s). Output: '{save_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert SVG files to PNG.")
    parser.add_argument("--svg_dir", default="./svg", help="Directory containing .svg files")
    parser.add_argument("--save_dir", default="./png", help="Directory to save .png files")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional scale factor (e.g., 2 for 2x). Requires CairoSVG.",
    )

    params = parser.parse_args()
    main(params)