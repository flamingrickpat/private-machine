import cv2, numpy as np, base64, io
from PIL import Image

def _to_b64(img_uint8, fmt="PNG", quality=95):
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        Image.fromarray(img_uint8).save(buf, format="JPEG", quality=quality)
    else:
        Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def exr_to_png_jpg_b64(path_exr, png_out="out.png", jpg_out="out.jpg", jpg_quality=95):
    # Read EXR as float32 (BGR[A] in OpenCV)
    exr = cv2.imread(path_exr, cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise FileNotFoundError(f"Could not read: {path_exr}")

    # Split color and (optional) alpha
    has_alpha = exr.ndim == 3 and exr.shape[2] == 4
    bgr = exr[..., :3] if has_alpha else exr
    alpha = exr[..., 3] if has_alpha else None

    # Simple Reinhard tonemap from HDR->SDR (works on linear EXR)
    ldr = bgr / (1.0 + bgr)
    ldr = np.clip(ldr, 0.0, 1.0)
    rgb8 = cv2.cvtColor((ldr * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Keep alpha for PNG if present (JPEG has no alpha)
    if alpha is not None:
        a8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        png_img = np.dstack([rgb8, a8])  # RGBA
    else:
        png_img = rgb8

    # Write files (optional)
    Image.fromarray(png_img).save(png_out, format="PNG")
    Image.fromarray(rgb8).save(jpg_out, format="JPEG", quality=jpg_quality)

    # Base64 strings
    b64_png = _to_b64(png_img, fmt="PNG")
    b64_jpg = _to_b64(rgb8, fmt="JPEG", quality=jpg_quality)
    return b64_png, b64_jpg