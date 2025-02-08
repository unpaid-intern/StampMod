import os
import sys
import io
import random
import threading
import tempfile
import json
import time
import math
import numba
from numba import njit, prange
from pathlib import Path
import socket
from filelock import FileLock, Timeout
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import cv2 
# Image processing libraries
from PIL import Image, ImageSequence, ImageGrab, ImageQt, ImageFilter, UnidentifiedImageError, ImageDraw
import numpy as np
import shutil
# Scikit-learn and SciPy utilities
from sklearn.cluster import KMeans
from joblib import parallel_backend
# PySide6 (Qt framework)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSlider, QComboBox,
    QProgressBar, QMessageBox, QStackedWidget, QLineEdit, QSizePolicy,
    QFormLayout, QGridLayout, QSpacerItem, QFrame, QStackedLayout, QScrollArea
)
from PySide6.QtGui import (
    QPixmap, QMovie, QIcon, QPainter, QCursor, QImage, QPen, QKeySequence, QShortcut
)
from PySide6.QtCore import (
    Qt, Signal, QObject, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QSize, QThread, Slot, QRect, QBuffer, QIODevice
)



def get_base_path() -> Path:
    if getattr(sys, 'frozen', False):

        base_path = Path(sys.executable).parent
    else:

        base_path = Path(__file__).parent


    return base_path.parent

def exe_path_fs(relative_path: str) -> Path:
    base_path = get_base_path()
    return (base_path / relative_path).resolve()

def exe_path_str(relative_path: str) -> str:

    return exe_path_fs(relative_path).as_posix()

def get_appdata_dir() -> Path:
    """
    Get the system-specific AppData/Local directory for storing application data.

    Returns:
        Path: The path to the application-specific directory in AppData/Local.
    """
    if os.name == "nt":  # Windows
        appdata_base = Path(os.getenv("LOCALAPPDATA"))
    else:
        appdata_base = Path.home() / ".local" / "share"
    
    appdata_dir = appdata_base / "webfishing_stamps_mod"
    appdata_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists
    return appdata_dir

def get_config_path() -> Path:
    """
    Get the path to the configuration file for PurplePuppy Stamps.

    Returns:
        Path: The full path to the configuration file.
    """
    # Start with the base path of the executable or script
    base_path = get_base_path()

    # Navigate up until we reach 'GDWeave'
    while base_path.name in ["mods", "PurplePuppy-Stamps"]:
        base_path = base_path.parent

    # Ensure the resolved base path is correct
    if base_path.name != "GDWeave":
        raise ValueError(f"Base path resolution error: {base_path} is not GDWeave.")

    # Navigate to the sibling 'configs' directory
    config_dir = (base_path / "configs").resolve()

    # Ensure the configs directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Define the specific configuration file name
    config_file = config_dir / "PurplePuppy.Stamps.json"

    return config_file

IPC_HOST = '127.0.0.1'
IPC_PORT = 65432
LOCK_FILE = Path(tempfile.gettempdir()) / 'imagePawcessor.lock'
LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
app_lock = None

processing_method_registry = {}
use_lab = False
has_chalks = False
chalks_colors = False
first = False
brightness = 0.5


def get_clipboard_image_via_pyside6():
    """
    Attempt to retrieve an image from the clipboard using PySide6.
    Returns a PIL.Image object if successful, otherwise None.
    """
    try:
        app_created = False
        if not QApplication.instance():
            # If no existing QApplication, create one
            app = QApplication(sys.argv)
            app_created = True
        else:
            app = QApplication.instance()

        clipboard = app.clipboard()
        mime_data = clipboard.mimeData()

        # 1) If the clipboard has URLs (file paths)
        if mime_data.hasUrls():
            for url in mime_data.urls():
                local_path = url.toLocalFile()
                if local_path and os.path.isfile(local_path):
                    return Image.open(local_path)

        # 2) If the clipboard has an image (raw pixel data)
        if mime_data.hasImage():
            # We can retrieve the QPixmap from the clipboard
            pixmap = clipboard.pixmap()
            if not pixmap.isNull():
                # Convert QPixmap to PNG bytes in memory
                buffer = QBuffer()
                buffer.open(QIODevice.WriteOnly)
                pixmap.save(buffer, "PNG")  # or "BMP", "JPEG", etc.
                qt_bytes = buffer.data()
                # Convert those bytes to a PIL image
                return Image.open(io.BytesIO(qt_bytes))

        if app_created:
            app.quit()

    except Exception as e:
        print("PySide6-based clipboard grab failed:", e)

    return None


def get_clipboard_image_fallback():
    """
    Fallback to Pillow's ImageGrab.grabclipboard().
    On Linux, this may require xclip or wl-paste if there's no running X server
    or if Wayland environment is missing the necessary support.
    """
    try:
        clipboard_content = ImageGrab.grabclipboard()
        if isinstance(clipboard_content, list):
            # Possibly file paths
            image_files = [f for f in clipboard_content if os.path.isfile(f)]
            if image_files:
                return Image.open(image_files[0])
            else:
                return None
        # Could be a PIL Image directly
        return clipboard_content

    except Exception as e:
        print("ImageGrab fallback failed:", e)
        return None


def get_clipboard_image():
    """
    Main helper to retrieve a PIL Image from the system clipboard.
    1) Try PySide6 for a cross-platform approach not requiring xclip/wl-paste.
    2) Fallback to Pillow's ImageGrab.grabclipboard().
    """
    img = get_clipboard_image_via_pyside6()
    if img is not None:
        return img

    return get_clipboard_image_fallback()

def register_processing_method(name, default_params=None, description=""):
    """
    Decorator to register a processing method.

    Parameters:
    - name (str): Name of the processing method.
    - default_params (dict): Default parameters for the processing method.
    - description (str): A brief description of the processing method.
    """
    def decorator(func):
        func.default_params = default_params or {}
        func.description = description
        processing_method_registry[name] = func
        return func
    return decorator

def prepare_image(img):
    """
    Converts image to RGBA if not already, and returns a writable copy of the image.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    # Make a writable copy of the image
    img = img.copy()
    return img



###############################################################################
#                   Utility and Helper Functions                              #
###############################################################################

def calculate_luminance(rgb):
    """
    Calculate luminance of an RGB color using standard weights.
    """
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def extract_color_key_brightness_range(color_key_array):
    """
    Compute the min and max brightness from the provided color keys.
    Returns: (min_brightness, max_brightness).
    """
    color_key_luminances = []
    for color_key in color_key_array:
        # Extract R, G, B from hex (without '#')
        hex_code = color_key['hex'].lstrip('#')
        rgb_tuple = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
        color_key_luminances.append(calculate_luminance(rgb_tuple))

    min_brightness = min(color_key_luminances)
    max_brightness = max(color_key_luminances)
    return min_brightness, max_brightness


def get_opaque_mask_and_rgb(image, alpha_threshold):
    """
    Separate alpha channel (if present) and create a mask for 'opaque' pixels.
    Returns:
        has_alpha (bool),
        alpha_channel (np.ndarray or None),
        rgb_image (np.ndarray),
        opaque_mask (np.ndarray of bool).
    """
    has_alpha = (image.shape[2] == 4)
    if has_alpha:
        alpha_channel = image[:, :, 3]
        rgb_image = image[:, :, :3]
        # Create a mask for pixels above alpha_threshold
        opaque_mask = alpha_channel >= alpha_threshold
    else:
        alpha_channel = None
        rgb_image = image
        # If no alpha channel, consider all opaque
        opaque_mask = np.ones(rgb_image.shape[:2], dtype=bool)
    
    return has_alpha, alpha_channel, rgb_image, opaque_mask


def global_contrast_stretch(rgb_image_uint8, opaque_mask, contrast_percentiles):
    """
    Apply global contrast stretching to each color channel using the specified percentiles.
    """
    lower_pct, upper_pct = contrast_percentiles
    for c in range(3):
        channel = rgb_image_uint8[:, :, c]
        # Compute percentiles only on opaque pixels
        min_val = np.percentile(channel[opaque_mask], lower_pct)
        max_val = np.percentile(channel[opaque_mask], upper_pct)
        if max_val - min_val < 1e-5:  # avoid near-zero division
            continue
        stretched = (channel - min_val) * (255.0 / (max_val - min_val))
        channel = np.clip(stretched, 0, 255).astype(np.uint8)
        rgb_image_uint8[:, :, c] = channel
    return rgb_image_uint8


def apply_gamma_correction(rgb_image_uint8, gamma):
    """
    Apply gamma correction if gamma != 1.0.
    """
    if abs(gamma - 1.0) > 1e-5:
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 
                          for i in range(256)]).astype("uint8")
        rgb_image_uint8 = cv2.LUT(rgb_image_uint8, table)
    return rgb_image_uint8

def apply_unsharp_mask(
    image,
    unsharp_strength=1.0,
    unsharp_radius=1.0,
    edge_threshold=5
):
    """
    Apply unsharp mask only on the luminance channel (Lab)
    and only where edges exceed `edge_threshold`.
    """
    if unsharp_strength <= 0:
        return image

    # Convert BGR → Lab
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # Blur just the L channel
    blurred_L = cv2.GaussianBlur(L, (0, 0), unsharp_radius)

    # Compute a mask of where differences exceed edge_threshold
    diff = cv2.absdiff(L, blurred_L)
    _, edge_mask = cv2.threshold(diff, edge_threshold, 255, cv2.THRESH_BINARY)

    # Sharpen L channel
    L_sharp = cv2.addWeighted(L, 1.0 + unsharp_strength,
                              blurred_L, -unsharp_strength, 0)

    # Combine original L where there's no significant edge
    L_combined = np.where(edge_mask > 0, L_sharp, L).astype(np.uint8)

    # Re-merge into Lab and convert back to BGR
    lab_sharpened = cv2.merge([L_combined, A, B])
    sharpened_bgr = cv2.cvtColor(lab_sharpened, cv2.COLOR_Lab2BGR)

    return sharpened_bgr


def apply_clahe(
    rgb_image_uint8,
    clahe_clip_limit=3.0,
    clahe_grid_size=8,
    gamma=0.9,
    color_boost=1.8,
    range_min=10,
    range_max=245,
):
    """
    Aggressive color + contrast transformation to:
      - Heavily increase color saturation
      - Reduce pure black/white areas
      - Retain local contrast by applying CLAHE on L-channel in Lab space

    :param rgb_image_uint8: 3-channel image in RGB order, dtype=uint8
    :param clahe_clip_limit: Clip limit for CLAHE
    :param clahe_grid_size: TileGridSize for CLAHE
    :param color_boost: Factor to multiply the a/b channels (saturation increase)
    :param range_min: Minimum L-channel value after rescaling (to avoid pure black)
    :param range_max: Maximum L-channel value after rescaling (to avoid pure white)
    :param gamma: Gamma correction factor (<1 => brighten midtones, >1 => darken)
    :return: Strongly color-boosted and contrast-adjusted image, dtype=uint8
    """

    # 1) Convert from RGB to Lab
    lab_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 2) Apply CLAHE on L channel
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_grid_size, clahe_grid_size)
    )
    l_eq = clahe.apply(l_channel)

    # 3) Force L channel away from pure 0 or 255 by rescaling:
    #    - First get min and max in the L-channel after CLAHE
    L_min, L_max = float(l_eq.min()), float(l_eq.max())
    if L_max > L_min:  # avoid division-by-zero
        # clamp to actual min/max
        l_clamped = np.clip(l_eq, L_min, L_max).astype(np.float32)
        # scale to [range_min .. range_max]
        scale = (range_max - range_min) / (L_max - L_min)
        l_rescaled = range_min + (l_clamped - L_min) * scale
        l_eq = np.clip(l_rescaled, 0, 255).astype(np.uint8)
    else:
        # if the L channel is flat (rare), just keep it as is
        l_eq = l_eq.astype(np.uint8)

    # 4) Gamma correction on L to further avoid large dark areas (gamma<1 => brighten)
    #    Build a LUT for [0..255].
    if abs(gamma - 1.0) > 1e-3:
        inv_gamma = 1.0 / gamma
        lut = np.array([
            ( (i / 255.0) ** inv_gamma ) * 255.0 for i in range(256)
        ]).astype("uint8")
        l_eq = cv2.LUT(l_eq, lut)

    # 5) Strongly boost colors in a/b channels:
    #    - Shift them around 128 (the neutral point in Lab)
    #    - Multiply to amplify saturation
    #    - Shift back, and clamp to valid [0..255]
    a_float = a_channel.astype(np.float32) - 128.0
    b_float = b_channel.astype(np.float32) - 128.0

    a_boosted = np.clip(a_float * color_boost, -128, 127) + 128.0
    b_boosted = np.clip(b_float * color_boost, -128, 127) + 128.0

    a_boosted = np.clip(a_boosted, 0, 255).astype(np.uint8)
    b_boosted = np.clip(b_boosted, 0, 255).astype(np.uint8)

    # 6) Merge back into Lab space
    lab_merged = cv2.merge((l_eq, a_boosted, b_boosted))

    # 7) Convert Lab back to RGB
    output_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

    return output_rgb




def restore_non_opaque_pixels(rgb_image_uint8, original_rgb, opaque_mask):
    """
    Restore the original RGB values for non-opaque pixels 
    in case transformations altered them.
    """
    rgb_image_uint8[~opaque_mask] = original_rgb[~opaque_mask]
    return rgb_image_uint8


###############################################################################
#                   Automatic Brightness Functions     #
###############################################################################

def auto_brightness_rgb(rgb_image_uint8, opaque_mask):
    """
    A minimal 'auto brightness' approach in RGB:
      - Measures the average luminance (using 0.299,0.587,0.114) of opaque pixels.
      - Shifts the image so that the average is near 128.
      - Applies an asymmetric correction:
          • Dark images are brightened up to +50.
          • Bright images are darkened only up to -25.
      - This pre-adjustment is intended for later matching to a 6-color palette.
    """
    float_img = rgb_image_uint8.astype(np.float32)
    # Compute luminance over opaque pixels.
    lum = (0.299 * float_img[opaque_mask, 0] +
           0.587 * float_img[opaque_mask, 1] +
           0.114 * float_img[opaque_mask, 2])
    avg_lum = np.mean(lum) if len(lum) > 0 else 128.0
    target_lum = 128.0
    diff = target_lum - avg_lum

    # Apply asymmetric clipping: allow stronger brightening than darkening.
    if diff < 0:
        diff = np.clip(diff, -25, 0)
    else:
        diff = np.clip(diff, 0, 50)

    float_img[opaque_mask] += diff
    out = np.clip(float_img, 0, 255).astype(np.uint8)
    return out


def auto_brightness_lab(rgb_image_uint8, opaque_mask):
    """
    A minimal 'auto brightness' approach in LAB:
      - Converts the image to Lab, measures the average L value on opaque pixels.
      - Shifts L toward 128 with asymmetric correction:
          • Dark images are brightened up to +50.
          • Bright images are darkened only up to -25.
      - Converts back to RGB.
    """
    lab = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_opaque = l_channel[opaque_mask]
    avg_l = np.mean(l_opaque) if l_opaque.size > 0 else 128.0
    target_l = 128.0
    diff = target_l - avg_l

    if diff < 0:
        diff = np.clip(diff, -25, 0)
    else:
        diff = np.clip(diff, 0, 50)

    l_channel[opaque_mask] += diff
    l_channel = np.clip(l_channel, 0, 255)

    merged = cv2.merge((l_channel, a_channel, b_channel)).astype(np.uint8)
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return out


###############################################################################
#           Brightness & Range Adjustments (with minimal color-cast fix)      #
###############################################################################

def adjust_brightness_and_range_rgb(rgb_image_uint8, opaque_mask, user_brightness, 
                                    min_brightness, max_brightness):
    """
    Adjust brightness and range in RGB.
    
    Note: The 'user_brightness' parameter is ignored.
    
    This function automatically adjusts the brightness of opaque areas so that
    their average luminance (computed in Lab) is shifted toward mid-range (128).
    The adjustment is asymmetric (darker images are brightened more than bright
    images are darkened). Finally, a channel-by-channel range alignment is applied 
    (mapping each channel's 1st to 99th percentile into [min_brightness, max_brightness]).
    """
    float_img = rgb_image_uint8.astype(np.float32)
    # Convert to Lab to work with luminance.
    lab = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Automatic brightness shift (user_brightness is ignored)
    l_opaque = l_channel[opaque_mask]
    avg_l = np.mean(l_opaque) if l_opaque.size > 0 else 128.0
    target_l = 128.0
    diff = target_l - avg_l
    if diff < 0:
        diff = np.clip(diff, -25, 0)
    else:
        diff = np.clip(diff, 0, 50)
    l_channel[opaque_mask] = np.clip(l_channel[opaque_mask] + diff, 0, 255)

    # Convert back to RGB for range alignment.
    merged_lab = cv2.merge((l_channel, a_channel, b_channel))
    out_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    # Channel-by-channel range alignment on opaque pixels.
    for c in range(3):
        channel_data = out_img[..., c][opaque_mask]
        c_min = np.percentile(channel_data, 1)
        c_max = np.percentile(channel_data, 99)
        denom = (c_max - c_min) if (c_max > c_min) else 1e-5
        range_ratio = (max_brightness - min_brightness) / denom

        scaled = (channel_data - c_min) * range_ratio + min_brightness
        out_img[..., c][opaque_mask] = np.clip(scaled, 0, 255)

    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    return out_img


def adjust_brightness_and_range_lab(rgb_image_uint8, opaque_mask, user_brightness, 
                                    min_brightness, max_brightness):
    """
    Adjust brightness and range in LAB.
    
    Note: The 'user_brightness' parameter is ignored.
    
    This function automatically adjusts the L channel (brightness) of opaque areas
    to shift the average toward 128 with an asymmetric correction 
    (darker images get brightened more than bright images get darkened). Then,
    the L channel is range-aligned so that its 1st to 99th percentiles map into
    [min_brightness, max_brightness]. Finally, the image is converted back to RGB.
    """
    lab_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Automatic brightness shift (ignoring user_brightness)
    l_opaque = l_channel[opaque_mask]
    avg_l = np.mean(l_opaque) if l_opaque.size > 0 else 128.0
    target_l = 128.0
    diff = target_l - avg_l
    if diff < 0:
        diff = np.clip(diff, -25, 0)
    else:
        diff = np.clip(diff, 0, 50)
    l_channel[opaque_mask] = np.clip(l_channel[opaque_mask] + diff, 0, 255)

    # Range alignment: map [1%ile, 99%ile] of the L channel to [min_brightness, max_brightness]
    l_opaque = l_channel[opaque_mask]
    current_min = np.percentile(l_opaque, 1)
    current_max = np.percentile(l_opaque, 99)
    denom = (current_max - current_min) if (current_max > current_min) else 1e-5
    range_ratio = (max_brightness - min_brightness) / denom

    l_scaled = (l_opaque - current_min) * range_ratio + min_brightness
    l_channel[opaque_mask] = np.clip(l_scaled, 0, 255)

    merged_lab = cv2.merge((l_channel, a_channel, b_channel))
    adjusted_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return adjusted_rgb


###############################################################################
#           Dynamic Determination of Boost and Threshold (Existing)           #
###############################################################################

def determine_dynamic_boost_and_threshold(color_key_array, rgb_image_uint8):
    """
    Dynamically decide 'boost' and 'threshold' for each color 
    based on the image content and how many colors are available.
    """
    global use_lab, chalks_colors

    dynamic_settings = {}
    num_colors = len(color_key_array)

    # Base defaults for threshold/boost
    if use_lab:
        base_threshold = 20
        base_boost = 1.4
    else:
        base_threshold = 28
        base_boost = 1.2

    # A simple 3-tier logic based on how many colors we have:
    if num_colors <= 7:  # Very small color set
        threshold_scale = 1.2
        boost_scale = 1.15
    elif num_colors <= 30:  # Medium palette
        threshold_scale = 1.0
        boost_scale = 1.0
    else:  # Large palette (31+)
        threshold_scale = 0.8
        boost_scale = 0.9

    if chalks_colors and num_colors > 30:
        # We might reduce threshold/boost further if truly big set
        threshold_scale *= 0.9
        boost_scale *= 0.95

    for color_key in color_key_array:
        color_number = color_key['number']
        final_threshold = max(int(base_threshold * threshold_scale), 5)
        final_boost = max(base_boost * boost_scale, 1.0)

        dynamic_settings[color_number] = {
            "boost": final_boost,
            "threshold": final_threshold
        }

    return dynamic_settings


###############################################################################
#             Selective Color Boosting (Uses Dynamic Boost/Threshold)         #
###############################################################################

def selective_color_boost_hsv(rgb_image_uint8, opaque_mask, color_key_array, dynamic_settings):
    """
    Boost saturation for pixels close to each target color in HSV space.
    If chalks_colors==False, skip boosting for certain colors (#ffe7c5, #2a3844).
    """
    global chalks_colors

    hsv_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2HSV)
    h = hsv_image[:, :, 0].astype(np.float32)
    s = hsv_image[:, :, 1].astype(np.float32)
    v = hsv_image[:, :, 2].astype(np.float32)

    for color_key in color_key_array:
        hex_code = color_key['hex'].lstrip('#').lower()
        
        # Skip boosting for #ffe7c5 or #2a3844 if chalks_colors==False
        if not chalks_colors:
            if hex_code in ('ffe7c5', '2a3844'):
                continue

        color_num = color_key['number']
        boost = dynamic_settings[color_num]['boost']
        threshold = dynamic_settings[color_num]['threshold']

        # Convert the target color to HSV
        color_rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
        color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
        target_h = color_hsv[0]

        # Hue difference
        hue_diff = np.abs(h - target_h)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)

        # Mask for pixels close to the target color
        color_mask = (hue_diff < threshold) & opaque_mask

        # Boost saturation
        s = np.where(color_mask, np.minimum(s * boost, 255), s)

    hsv_image[:, :, 1] = s
    hsv_image[:, :, 2] = v  # If you want to also adjust V, do so here

    boosted_rgb = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return boosted_rgb


import numpy as np
import cv2

def selective_color_boost_lab_fixed(
    rgb_image_uint8,
    opaque_mask,
    color_key_array,
    dynamic_settings
):
    """
    A replacement for 'selective_color_boost_lab_stub' that avoids shifting 
    dark blues/greens into red or other unintended hues in Lab space.

    - rgb_image_uint8:   (H, W, 3) uint8 image.
    - opaque_mask:       (H, W) boolean mask (True for pixels to adjust).
    - color_key_array:   (optional) data about colors; not deeply used here.
    - dynamic_settings:  (dict) might contain e.g. 'lab_boost_factor'.
    
    Returns:
        The updated rgb_image_uint8 in-place (also returned) with boosted color.
    """

    # 1) Convert from RGB to Lab (OpenCV range: L[0..255], a[0..255], b[0..255]).
    lab_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2LAB)

    # Convert to float32 for safe arithmetic.
    lab_f32 = lab_image.astype(np.float32)

    # 2) Separate the three channels
    L_channel = lab_f32[:, :, 0]
    a_channel = lab_f32[:, :, 1]
    b_channel = lab_f32[:, :, 2]

    # 3) In OpenCV's Lab:
    #    L in [0..255], a in [0..255], b in [0..255]
    #    The "real" Lab often has L in [0..100], a,b in ~[-128..+128].
    #    So let's shift a,b down by 128 so that 128->0 is neutral.
    a_channel -= 128.0
    b_channel -= 128.0

    # 4) Apply a saturation-like boost. You can tweak the factor as desired.
    #    You might store this in 'dynamic_settings.get("lab_boost_factor", 1.2)' or similar.
    boost_factor = dynamic_settings.get("lab_boost_factor", 1.25)

    # Typically, you either:
    # (A) Multiply a,b by the same factor, or
    # (B) Increase magnitude of (a,b) from their neutral center (0,0).
    # We'll do a simple uniform scale here:
    a_channel *= boost_factor
    b_channel *= boost_factor

    # 5) Clamp a,b back into the valid [-128..127] range
    a_channel = np.clip(a_channel, -128, 127)
    b_channel = np.clip(b_channel, -128, 127)

    # 6) Shift a,b back up by +128 to restore OpenCV’s range.
    a_channel += 128.0
    b_channel += 128.0

    # 7) Place the channels back, and also clamp L to [0..255].
    #    (Some code also remaps L to a narrower [0..100], but we’ll stay consistent with OpenCV.)
    lab_f32[:, :, 0] = np.clip(L_channel, 0, 255)
    lab_f32[:, :, 1] = np.clip(a_channel, 0, 255)
    lab_f32[:, :, 2] = np.clip(b_channel, 0, 255)

    # Convert back to uint8
    lab_fixed = lab_f32.astype(np.uint8)

    # 8) Convert Lab -> RGB
    boosted_rgb = cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2RGB)

    # 9) Write back only into opaque pixels, in-place.
    #    (If you prefer to modify all pixels, remove the mask indexing.)
    rgb_image_uint8[opaque_mask] = boosted_rgb[opaque_mask]

    return rgb_image_uint8



###############################################################################
#                              Main Function                                  #
###############################################################################

def preprocess_image(
    image,
    color_key_array,
    callback=None,
    gif=False
):
    """
    Preprocesses the image to enhance gradients, expand the color range,
    and improve overall image quality for better processing results.

    - image:           NumPy array (H, W, C) in RGBA or RGB format.
    - color_key_array: List of dicts with color info to boost (includes 'number', 'hex').
    - callback:        Optional callable for progress/feedback.
    - gif:             Whether the input is multi-frame (e.g. GIF).
                       If True, skip certain time-consuming or frame-incompatible steps.

    Returns:
        preprocessed_image (np.ndarray).
    """
    steps = None
    global use_lab, chalks_colors, brightness

    #--------------------------------------------------------------------------
    # 1. Separate default steps for chalks_colors = True vs. chalks_colors = False
    #--------------------------------------------------------------------------
    chalk_defaults = {
        'alpha_channel_separation': True,
        'contrast_stretch': True,
        'brightness_range_adjustment': False, 
        'gamma_correction': False,  
        'unsharp_mask': True,           
        'flat_patch_restoration': False if gif else False, 
        'clahe': False,
        'dynamic_boost_and_threshold': True,
        'color_boost': True,
        'restore_non_opaque': False,
        'recombine_alpha': True
    }

    normal_defaults = {
        'alpha_channel_separation': True,
        'contrast_stretch': True,
        'brightness_range_adjustment': True,
        'gamma_correction': True,
        'unsharp_mask': True,
        'flat_patch_restoration': False if gif else False, 
        'clahe': True,
        'dynamic_boost_and_threshold': True,
        'color_boost': True,
        'restore_non_opaque': True,
        'recombine_alpha': True
    }

    # Pick which set of defaults to use
    default_steps = chalk_defaults if chalks_colors else normal_defaults

    #--------------------------------------------------------------------------
    # 2. Merge any user-provided `steps` override (if we had a steps dict)
    #--------------------------------------------------------------------------
    if steps is not None:
        for step_name, step_value in steps.items():
            if step_name in default_steps:
                default_steps[step_name] = step_value

    #--------------------------------------------------------------------------
    # 3. Decide default processing parameters
    #--------------------------------------------------------------------------
    if chalks_colors:
        # Minimal transformations for extended palette
        params = {
            'alpha_threshold': 191,
            'clahe_clip_limit': 1.0,
            'clahe_grid_size': 4,
            'unsharp_strength': 1.2, 
            'unsharp_radius': 2.0,
            'gamma_correction': 1.0,  
            'contrast_percentiles': (0.2, 99.8), 
        }
    else:
        params = {
            'alpha_threshold': 191,
            'clahe_clip_limit': 3.5,
            'clahe_grid_size': 6,
            'unsharp_strength': 1.5,
            'unsharp_radius': 2,
            'gamma_correction': 0.9,
            'contrast_percentiles': (1, 99),
        }

    #--------------------------------------------------------------------------
    # 4. Extract color key brightness range
    #--------------------------------------------------------------------------
    if callback:
        callback("Step 1: Extracting brightness range from color keys...")
    min_brightness, max_brightness = extract_color_key_brightness_range(color_key_array)

    #--------------------------------------------------------------------------
    # 5. Separate alpha channel / create opaque mask
    #--------------------------------------------------------------------------
    if default_steps['alpha_channel_separation']:
        if callback:
            callback("Step 2: Separating alpha channel and creating opaque mask...")
        has_alpha, alpha_channel, rgb_image, opaque_mask = get_opaque_mask_and_rgb(
            image,
            params['alpha_threshold']
        )
        rgb_image_uint8 = rgb_image.astype(np.uint8).copy()
    else:
        has_alpha = False
        alpha_channel = None
        rgb_image_uint8 = image[..., :3].astype(np.uint8).copy()
        opaque_mask = np.ones(rgb_image_uint8.shape[:2], dtype=bool)

    #--------------------------------------------------------------------------
    # 6. Global Contrast Stretch
    #--------------------------------------------------------------------------
    if default_steps['contrast_stretch']:
        if callback:
            callback("Step 3: Global contrast stretching...")
        rgb_image_uint8 = global_contrast_stretch(
            rgb_image_uint8,
            opaque_mask,
            params['contrast_percentiles']
        )

    #--------------------------------------------------------------------------
    # 7. Brightness & Range Adjustment
    #--------------------------------------------------------------------------
    if default_steps['brightness_range_adjustment']:
        if callback:
            callback("Step 4: Adjusting brightness (in Lab) to respect user brightness...")

        # If chalks_colors == True and brightness is close to 0.5, we can auto
        # Else use the user brightness
        if chalks_colors and abs(brightness - 0.5) < 1e-5:
            # Automatic brightness
            if use_lab:
                rgb_image_uint8 = auto_brightness_lab(rgb_image_uint8, opaque_mask)
            else:
                rgb_image_uint8 = auto_brightness_rgb(rgb_image_uint8, opaque_mask)
        else:
            # Manual brightness shift
            if use_lab:
                rgb_image_uint8 = adjust_brightness_and_range_lab(
                    rgb_image_uint8,
                    opaque_mask,
                    brightness,
                    min_brightness,
                    max_brightness
                )
            else:
                rgb_image_uint8 = adjust_brightness_and_range_rgb(
                    rgb_image_uint8,
                    opaque_mask,
                    brightness,
                    min_brightness,
                    max_brightness
                )

    #--------------------------------------------------------------------------
    # 8. Gamma Correction
    #--------------------------------------------------------------------------
    if default_steps['gamma_correction']:
        if callback:
            callback("Step 5: Applying gamma correction...")
        rgb_image_uint8 = apply_gamma_correction(
            rgb_image_uint8,
            params['gamma_correction']
        )

    #--------------------------------------------------------------------------
    # 9. CLAHE
    #--------------------------------------------------------------------------
    if default_steps['clahe']:
        if callback:
            callback("Step 6: Applying CLAHE...")
        rgb_image_uint8 = apply_clahe(
            rgb_image_uint8,
            clahe_clip_limit=params['clahe_clip_limit'],
            clahe_grid_size=params['clahe_grid_size'],
            gamma=params['gamma_correction']
        )

    #--------------------------------------------------------------------------
    # 10. Dynamic Determination of Boost/Threshold + Selective Color Boost
    #--------------------------------------------------------------------------
    if default_steps['dynamic_boost_and_threshold'] or default_steps['color_boost']:
        if callback:
            callback("Step 7: Determining dynamic boost & threshold + selective color boost...")
        dynamic_settings = determine_dynamic_boost_and_threshold(
            color_key_array, 
            rgb_image_uint8
        )

        if default_steps['color_boost']:
            if use_lab:
                rgb_image_uint8 = selective_color_boost_lab_fixed(
                    rgb_image_uint8,
                    opaque_mask,
                    color_key_array,
                    dynamic_settings
                )
            else:
                rgb_image_uint8 = selective_color_boost_hsv(
                    rgb_image_uint8,
                    opaque_mask,
                    color_key_array,
                    dynamic_settings
                )

    #--------------------------------------------------------------------------
    # 11. Unsharp Mask
    #--------------------------------------------------------------------------
    if default_steps['unsharp_mask'] and params['unsharp_strength'] > 0:
        if callback:
            callback("Step 8: Applying unsharp masking...")
        # Convert to BGR for unsharp, then convert back
        bgr_for_unsharp = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR)
        bgr_sharpened = apply_unsharp_mask(
            bgr_for_unsharp,
            unsharp_strength=params['unsharp_strength'],
            unsharp_radius=params['unsharp_radius'],
            edge_threshold=5
        )
        rgb_image_uint8 = cv2.cvtColor(bgr_sharpened, cv2.COLOR_BGR2RGB)

    #--------------------------------------------------------------------------
    # 12. Restore Non-Opaque Pixels
    #--------------------------------------------------------------------------
    if default_steps['restore_non_opaque']:
        if callback:
            callback("Step 9: Restoring non-opaque pixels...")
        rgb_image_uint8 = restore_non_opaque_pixels(rgb_image_uint8, image[..., :3], opaque_mask)

    #--------------------------------------------------------------------------
    # 13. Recombine Alpha if needed
    #--------------------------------------------------------------------------
    if default_steps['recombine_alpha'] and has_alpha:
        if callback:
            callback("Step 10: Recombining alpha channel (if present).")
        preprocessed_image = np.dstack((rgb_image_uint8, alpha_channel))
    else:
        preprocessed_image = rgb_image_uint8

    return preprocessed_image



def crop_to_solid_area(image: Image.Image) -> Image.Image:
    """
    Crops an RGBA image to remove fully transparent (alpha=0) areas around the solid pixels.
    
    Args:
        image (Image.Image): Input image in RGBA mode.

    Returns:
        Image.Image: Cropped image containing only solid pixels.
    """
    if image.mode != "RGBA":
        raise ValueError("Image must be in RGBA mode")

    # Extract the alpha channel
    alpha = image.split()[3]  # The fourth channel is alpha in RGBA

    # Get the bounding box of the non-transparent area
    bbox = alpha.getbbox()
    if bbox:
        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)
        return cropped_image
    else:
        # If the entire image is transparent, return an empty (1x1) RGBA image
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


def resize_image(img, target_size):
    """
    Resizes the image to the target size while maintaining aspect ratio.
    Uses nearest neighbor for upscaling and LANCZOS for downscaling.
    Handles the alpha channel separately to prevent artifacting.
    """
    try:
        width, height = img.size
        scale_factor = target_size / float(max(width, height))
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        
        # Choose the resampling method based on the scaling factor.
        # Use NEAREST for upscaling (scale_factor > 1) to avoid introducing new artifacts,
        # and LANCZOS for downscaling.
        resample_method = Image.NEAREST if scale_factor > 1 else Image.LANCZOS

        # Separate the alpha channel if present.
        if img.mode == 'RGBA':
            img_no_alpha = img.convert('RGB')
            alpha = img.getchannel('A')

            # Resize RGB and alpha channels separately with the chosen resample method.
            img_no_alpha = img_no_alpha.resize((new_width, new_height), resample=resample_method)
            alpha = alpha.resize((new_width, new_height), resample=resample_method)

            # Merge the resized channels back together.
            img = Image.merge('RGBA', (*img_no_alpha.split(), alpha))
        else:
            img = img.resize((new_width, new_height), resample=resample_method)

        return img
    except Exception as e:
        raise RuntimeError(f"Failed to resize the image: {e}")



def adjust_brightness(image, brightness):
    """
    Adjusts the brightness of a PIL image in the RGB space.

    The brightness parameter should range from -0.5 to 1.5, where:
      - brightness = -0.5 yields a completely black image,
      - brightness = 0.5 yields no change, and
      - brightness = 1.5 yields a completely white image.
      
    For brightness values below 0.5, the image is darkened by multiplying
    all pixel values by a factor; for brightness values above 0.5, the image is 
    brightened by linearly blending the original image with white.

    :param image: A PIL.Image instance.
    :param brightness: A float in the range [-0.5, 1.5].
    :return: A new PIL.Image with adjusted brightness.
    """
    # Convert the PIL image to a NumPy array of type float32 for processing.
    arr = np.array(image).astype(np.float32)
    
    if brightness < 0.5:
        # For darkening: map brightness from [-0.5, 0.5] to a scale factor [0, 1].
        # At brightness = -0.5, factor = 0 (black); at brightness = 0.5, factor = 1 (no change).
        factor = (brightness + 0.5)  # This is linear: e.g., brightness=0.25 gives factor=0.75.
        new_arr = factor * arr
    else:
        # For brightening: map brightness from [0.5, 1.5] to a blend factor [0, 1].
        # At brightness = 0.5, factor = 0 (no change); at brightness = 1.5, factor = 1 (white).
        factor = (brightness - 0.5)  # For example, brightness=1.0 gives factor=0.5.
        new_arr = (1 - factor) * arr + factor * 255

    # Ensure values are within the valid range and convert back to uint8.
    new_arr = np.clip(new_arr, 0, 255).astype(np.uint8)
    
    # Convert the NumPy array back to a PIL Image and return it.
    return Image.fromarray(new_arr)



# ---------------------------------------------------------------------
# Single canonical definition of rgb_to_lab_numba (scaled L to 0..255).
# ---------------------------------------------------------------------
@njit(cache=True)
def rgb_to_lab_numba(r, g, b):
    # Convert from [0,255] to [0,1]
    R = r / 255.0
    G = g / 255.0
    B = b / 255.0

    # Gamma correction
    if R > 0.04045:
        R = ((R + 0.055) / 1.055) ** 2.4
    else:
        R = R / 12.92
    if G > 0.04045:
        G = ((G + 0.055) / 1.055) ** 2.4
    else:
        G = G / 12.92
    if B > 0.04045:
        B = ((B + 0.055) / 1.055) ** 2.4
    else:
        B = B / 12.92

    # Convert to XYZ using the sRGB matrix
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505

    # Normalize for D65 white point
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883

    # f(t) function with threshold 0.008856
    if X > 0.008856:
        fx = X ** (1.0/3.0)
    else:
        fx = 7.787 * X + 16.0/116.0

    if Y > 0.008856:
        fy = Y ** (1.0/3.0)
    else:
        fy = 7.787 * Y + 16.0/116.0

    if Z > 0.008856:
        fz = Z ** (1.0/3.0)
    else:
        fz = 7.787 * Z + 16.0/116.0

    # Compute L, a, b (scale L from [0,100] to [0,255])
    L = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    L *= (255.0 / 100.0)
    return L, a_val, b_val

# ------------------------------------------------------------------------------
# map_pixels_rgb / map_pixels_lab: single definitions for entire-image mapping
# ------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def map_pixels_rgb(pixels, palette):
    """
    For each pixel in 'pixels' (H x W x 3 float32), find the closest color in palette (N x 3 float32).
    Returns a new array of the same shape with quantized values in [0..255].
    """
    H, W, _ = pixels.shape
    out = np.empty_like(pixels)
    n_palette = palette.shape[0]

    for i in prange(H):
        for j in range(W):
            r = pixels[i, j, 0]
            g = pixels[i, j, 1]
            b = pixels[i, j, 2]
            best_index = 0
            best_dist = 1e10
            for k in range(n_palette):
                dr = r - palette[k, 0]
                dg = g - palette[k, 1]
                db = b - palette[k, 2]
                dist = dr*dr + dg*dg + db*db
                if dist < best_dist:
                    best_dist = dist
                    best_index = k
            out[i, j, 0] = palette[best_index, 0]
            out[i, j, 1] = palette[best_index, 1]
            out[i, j, 2] = palette[best_index, 2]
    return out

@njit(parallel=True, cache=True)
def map_pixels_lab(pixels, palette_rgb, palette_lab):
    """
    For each pixel in 'pixels' (H x W x 3 float32), convert to LAB, then find the closest palette color
    using LAB distance. Returns an array with quantized values in [0..255].
    """
    H, W, _ = pixels.shape
    out = np.empty_like(pixels)
    n_palette = palette_rgb.shape[0]

    for i in prange(H):
        for j in range(W):
            r = pixels[i, j, 0]
            g = pixels[i, j, 1]
            b = pixels[i, j, 2]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            best_index = 0
            best_dist = 1e10

            for k in range(n_palette):
                dL = L - palette_lab[k, 0]
                da = a_val - palette_lab[k, 1]
                db = b_val - palette_lab[k, 2]
                dist = dL*dL + da*da + db*db
                if dist < best_dist:
                    best_dist = dist
                    best_index = k

            out[i, j, 0] = palette_rgb[best_index, 0]
            out[i, j, 1] = palette_rgb[best_index, 1]
            out[i, j, 2] = palette_rgb[best_index, 2]
    return out

# ------------------------------------------------------------------------------
# Numba-compiled function for error diffusion.
# ------------------------------------------------------------------------------
@njit(cache=True)
def optimized_error_diffusion_dithering_numba(
    img_array, alpha_mask, width, height, strength,
    palette_rgb, palette_lab, diffusion_matrix, use_lab_flag
):
    """
    Loops over each pixel, finds the closest palette color,
    computes the quantization error, and distributes it.
    """
    n_palette = palette_rgb.shape[0]
    n_diff = diffusion_matrix.shape[0]

    for y in range(height):
        for x in range(width):
            if not alpha_mask[y, x]:
                continue

            old_r = img_array[y, x, 0]
            old_g = img_array[y, x, 1]
            old_b = img_array[y, x, 2]

            # Find the closest color in palette
            best_index = 0
            best_dist = 1e10

            if use_lab_flag:
                L, a_val, b_val = rgb_to_lab_numba(old_r, old_g, old_b)
                for i in range(n_palette):
                    dL = L - palette_lab[i, 0]
                    da = a_val - palette_lab[i, 1]
                    db = b_val - palette_lab[i, 2]
                    dist = dL*dL + da*da + db*db
                    if dist < best_dist:
                        best_dist = dist
                        best_index = i
            else:
                for i in range(n_palette):
                    dr = old_r - palette_rgb[i, 0]
                    dg = old_g - palette_rgb[i, 1]
                    db = old_b - palette_rgb[i, 2]
                    dist = dr*dr + dg*dg + db*db
                    if dist < best_dist:
                        best_dist = dist
                        best_index = i

            new_r = palette_rgb[best_index, 0]
            new_g = palette_rgb[best_index, 1]
            new_b = palette_rgb[best_index, 2]

            err_r = (old_r - new_r) * strength
            err_g = (old_g - new_g) * strength
            err_b = (old_b - new_b) * strength

            # Quantize current pixel
            img_array[y, x, 0] = new_r
            img_array[y, x, 1] = new_g
            img_array[y, x, 2] = new_b

            # Distribute error
            for i in range(n_diff):
                dx = int(diffusion_matrix[i, 0])
                dy = int(diffusion_matrix[i, 1])
                coeff = diffusion_matrix[i, 2]
                nx = x + dx
                ny = y + dy

                if 0 <= nx < width and 0 <= ny < height and alpha_mask[ny, nx]:
                    r_val = img_array[ny, nx, 0] + err_r * coeff
                    g_val = img_array[ny, nx, 1] + err_g * coeff
                    b_val = img_array[ny, nx, 2] + err_b * coeff

                    # Clamp
                    if r_val < 0:
                        r_val = 0.0
                    elif r_val > 255:
                        r_val = 255.0
                    if g_val < 0:
                        g_val = 0.0
                    elif g_val > 255:
                        g_val = 255.0
                    if b_val < 0:
                        b_val = 0.0
                    elif b_val > 255:
                        b_val = 255.0

                    img_array[ny, nx, 0] = r_val
                    img_array[ny, nx, 1] = g_val
                    img_array[ny, nx, 2] = b_val

# ------------------------------------------------------------------------------
# Public error diffusion function that calls the numba-compiled core.
# ------------------------------------------------------------------------------
def optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix):
    global use_lab

    img = img.copy()
    img_array = np.array(img, dtype=np.float32)
    height, width = img_array.shape[:2]

    # Alpha mask
    has_alpha = (img.mode == 'RGBA')
    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        alpha_mask = (alpha_channel > 0)
    else:
        alpha_mask = np.ones((height, width), dtype=np.bool_)

    # Build palette
    palette_rgb = np.array(list(color_key.values()), dtype=np.float32)
    if use_lab:
        n = palette_rgb.shape[0]
        palette_lab = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            r = palette_rgb[i, 0]
            g = palette_rgb[i, 1]
            b = palette_rgb[i, 2]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            palette_lab[i, 0] = L
            palette_lab[i, 1] = a_val
            palette_lab[i, 2] = b_val
    else:
        palette_lab = np.empty((0, 3), dtype=np.float32)

    diffusion_matrix = np.array(diffusion_matrix, dtype=np.float32)
    optimized_error_diffusion_dithering_numba(
        img_array, alpha_mask, width, height, strength,
        palette_rgb, palette_lab, diffusion_matrix, use_lab
    )

    # Clamp and reassemble
    rgb_result = np.clip(img_array[:, :, :3], 0, 255).astype(np.uint8)
    if has_alpha:
        alpha_result = np.array(img)[:, :, 3]
        result_array = np.dstack((rgb_result, alpha_result))
        mode = 'RGBA'
    else:
        result_array = rgb_result
        mode = 'RGB'

    return Image.fromarray(result_array, mode)

# ------------------------------------------------------------------------------
# Single-pixel palette lookup: finds the one closest color in either LAB or RGB
# ------------------------------------------------------------------------------
@njit(cache=True)
def find_closest_color_numba(pixel, palette_rgb, palette_lab, use_lab_flag):
    best_index = 0
    best_dist = 1e10
    if use_lab_flag and palette_lab.shape[0] > 0:
        L, a_val, b_val = rgb_to_lab_numba(pixel[0], pixel[1], pixel[2])
        for i in range(palette_lab.shape[0]):
            dL = L - palette_lab[i, 0]
            da = a_val - palette_lab[i, 1]
            db = b_val - palette_lab[i, 2]
            dist = dL*dL + da*da + db*db
            if dist < best_dist:
                best_dist = dist
                best_index = i
    else:
        for i in range(palette_rgb.shape[0]):
            dr = pixel[0] - palette_rgb[i, 0]
            dg = pixel[1] - palette_rgb[i, 1]
            db = pixel[2] - palette_rgb[i, 2]
            dist = dr*dr + dg*dg + db*db
            if dist < best_dist:
                best_dist = dist
                best_index = i
    return best_index

def find_closest_color(pixel, color_key):
    global use_lab
    pixel_arr = np.array(pixel, dtype=np.float32)
    palette_rgb = np.array(list(color_key.values()), dtype=np.float32)
    color_nums = list(color_key.keys())

    if use_lab:
        n = palette_rgb.shape[0]
        palette_lab = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            r = palette_rgb[i, 0]
            g = palette_rgb[i, 1]
            b = palette_rgb[i, 2]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            palette_lab[i, 0] = L
            palette_lab[i, 1] = a_val
            palette_lab[i, 2] = b_val
    else:
        palette_lab = np.empty((0, 3), dtype=np.float32)

    idx = find_closest_color_numba(pixel_arr, palette_rgb, palette_lab, use_lab)
    return color_nums[idx]

# ------------------------------------------------------------------------------
# Map an entire image’s pixels to the nearest palette color.
# ------------------------------------------------------------------------------
def find_closest_colors_image(image_array, color_key):
    global use_lab
    has_alpha = (image_array.shape[2] == 4)

    # Separate alpha if present
    if has_alpha:
        rgb_data = image_array[:, :, :3]
        alpha_channel = image_array[:, :, 3]
    else:
        rgb_data = image_array

    palette_rgb = np.array(list(color_key.values()), dtype=np.uint8)

    if use_lab:
        # Precompute LAB for palette
        n = palette_rgb.shape[0]
        palette_lab = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            r = palette_rgb[i, 0]
            g = palette_rgb[i, 1]
            b = palette_rgb[i, 2]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            palette_lab[i, 0] = L
            palette_lab[i, 1] = a_val
            palette_lab[i, 2] = b_val
        mapped_rgb = map_pixels_lab(rgb_data.astype(np.float32), palette_rgb.astype(np.float32), palette_lab)
    else:
        mapped_rgb = map_pixels_rgb(rgb_data.astype(np.float32), palette_rgb.astype(np.float32))

    mapped_rgb_uint8 = np.clip(mapped_rgb, 0, 255).astype(np.uint8)

    if has_alpha:
        mapped_data = np.dstack((mapped_rgb_uint8, alpha_channel))
    else:
        mapped_data = mapped_rgb_uint8

    return mapped_data


def rgb_to_lab_single(rgb):
    arr = np.uint8([[rgb]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0]
    return lab

def rgb_palette_to_lab(color_key):
    rgb_vals = np.array(list(color_key.values()), dtype=np.uint8).reshape(-1, 1, 3)
    lab_vals = cv2.cvtColor(rgb_vals, cv2.COLOR_RGB2LAB)
    return lab_vals.reshape(-1, 3)

def build_color_key(color_key_array):
    color_key = {}
    for item in color_key_array:
        color_num = item['number']
        hex_code = item['hex'].lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        color_key[color_num] = rgb
    return color_key

# ------------------------------------------------------------------------------
# Actual processing methods below
# ------------------------------------------------------------------------------

@register_processing_method(
    'Color Match',
    default_params={},
    description="Maps each pixel to the closest color of chalk. Basic, consistent, and reliable."
)
def color_matching(img, color_key, params):
    img_array = np.array(img)
    has_alpha = (img_array.shape[2] == 4) if img_array.ndim == 3 else False
    alpha_threshold = 191

    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        opaque_mask = (alpha_channel > alpha_threshold)
    else:
        opaque_mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=bool)

    mapped = find_closest_colors_image(img_array, color_key)

    if has_alpha:
        mapped[~opaque_mask] = img_array[~opaque_mask]

    mode = 'RGBA' if has_alpha else 'RGB'
    result_img = Image.fromarray(mapped, mode=mode)
    return result_img


@register_processing_method(
    'K-Means Mapping',
    default_params={'Clusters': 12},
    description="Simplify complex images to be less noisy! Use slider to adjust the amount of color groups. Great for limited palette."
)
def simple_k_means_palette_mapping(img, color_key, params):
    has_alpha = (img.mode == 'RGBA')
    if has_alpha:
        alpha_channel = np.array(img.getchannel('A'))
        rgb_img = img.convert('RGB')
    else:
        alpha_channel = None
        rgb_img = img

    data = np.array(rgb_img)
    data_flat = data.reshape((-1, 3))

    clusters = params['Clusters']
    if clusters == 16:
        clusters = 24  # special tweak

    with parallel_backend('threading', n_jobs=1):
        kmeans = KMeans(
            n_clusters=clusters,
            init="k-means++",
            n_init=10,
            random_state=0
        ).fit(data_flat)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Map cluster centers individually
    cluster_map = {}
    for i, center in enumerate(cluster_centers):
        center_rgb = tuple(center.astype(np.uint8))
        c_idx = find_closest_color(center_rgb, color_key)
        cluster_map[i] = color_key[c_idx]

    mapped_flat = np.array([cluster_map[label] for label in labels], dtype=np.uint8)
    mapped_data = mapped_flat.reshape(data.shape)

    if has_alpha:
        rgba_data = np.dstack((mapped_data, alpha_channel))
        result_img = Image.fromarray(rgba_data, 'RGBA')
    else:
        result_img = Image.fromarray(mapped_data, 'RGB')

    return result_img

# ---------------------------------------------------------------------------
# Numba‑jittable conversion from sRGB to CIELAB.
# (L is scaled to [0,255] so that it’s compatible with our palette.)
# ---------------------------------------------------------------------------
@njit(cache=True)
def rgb_to_lab_numba(r, g, b):
    # Convert from [0,255] to [0,1]
    R = r / 255.0
    G = g / 255.0
    B = b / 255.0
    # Gamma correction
    if R > 0.04045:
        R = ((R + 0.055) / 1.055) ** 2.4
    else:
        R = R / 12.92
    if G > 0.04045:
        G = ((G + 0.055) / 1.055) ** 2.4
    else:
        G = G / 12.92
    if B > 0.04045:
        B = ((B + 0.055) / 1.055) ** 2.4
    else:
        B = B / 12.92
    # Convert to XYZ using the sRGB matrix
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505
    # Normalize for D65 white point
    X = X / 0.95047
    Y = Y / 1.00000
    Z = Z / 1.08883
    # f(t) with threshold
    if X > 0.008856:
        fx = X ** (1/3)
    else:
        fx = 7.787 * X + 16/116.0
    if Y > 0.008856:
        fy = Y ** (1/3)
    else:
        fy = 7.787 * Y + 16/116.0
    if Z > 0.008856:
        fz = Z ** (1/3)
    else:
        fz = 7.787 * Z + 16/116.0
    # Compute L, a, b (L normally in [0,100] is scaled to [0,255])
    L = 116.0 * fy - 16.0
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    L = L * (255.0 / 100.0)
    return L, a_val, b_val

# ---------------------------------------------------------------------------
# Numba‑jitted helper: find the closest palette color for one pixel.
# Depending on use_lab_flag, the distance is computed in LAB or RGB space.
# ---------------------------------------------------------------------------
@njit(cache=True)
def find_closest_color_numba(pixel, palette_rgb, palette_lab, use_lab_flag):
    best_index = 0
    best_dist = 1e10
    if use_lab_flag:
        # Convert the pixel to LAB.
        L, a_val, b_val = rgb_to_lab_numba(pixel[0], pixel[1], pixel[2])
        for i in range(palette_lab.shape[0]):
            dL = L - palette_lab[i, 0]
            da = a_val - palette_lab[i, 1]
            db = b_val - palette_lab[i, 2]
            dist = dL * dL + da * da + db * db
            if dist < best_dist:
                best_dist = dist
                best_index = i
    else:
        for i in range(palette_rgb.shape[0]):
            dr = pixel[0] - palette_rgb[i, 0]
            dg = pixel[1] - palette_rgb[i, 1]
            db = pixel[2] - palette_rgb[i, 2]
            dist = dr * dr + dg * dg + db * db
            if dist < best_dist:
                best_dist = dist
                best_index = i
    return best_index

# ---------------------------------------------------------------------------
# Numba‑jitted function to perform hybrid error diffusion.
# Parameters:
#   img_array      : (H x W x 3) float32 array containing RGB channels.
#   saliency_array : (H x W) float32 array with values in [0,1].
#   alpha_mask     : (H x W) boolean array indicating non‑transparent pixels.
#   palette_rgb    : (N x 3) float32 palette in RGB.
#   palette_lab    : (N x 3) float32 palette in LAB (if using LAB; otherwise empty).
#   use_lab_flag   : boolean flag to select LAB vs. RGB for color matching.
#   atkinson_matrix: (M1 x 3) float32 array of (dx, dy, coeff) for Atkinson.
#   floyd_matrix   : (M2 x 3) float32 array of (dx, dy, coeff) for Floyd–Steinberg.
#   strength       : float, multiplier for quantization error.
# ---------------------------------------------------------------------------
@njit(cache=True)
def hybrid_dither_numba(img_array, saliency_array, alpha_mask, palette_rgb, palette_lab, use_lab_flag, atkinson_matrix, floyd_matrix, strength):
    height = img_array.shape[0]
    width = img_array.shape[1]
    for y in range(height):
        for x in range(width):
            if not alpha_mask[y, x]:
                continue
            # Get current pixel color (RGB)
            old_r = img_array[y, x, 0]
            old_g = img_array[y, x, 1]
            old_b = img_array[y, x, 2]
            # Build a temporary 3-element array for palette matching.
            temp = np.empty(3, dtype=np.float32)
            temp[0] = old_r
            temp[1] = old_g
            temp[2] = old_b
            # Find the index of the nearest palette color.
            idx = find_closest_color_numba(temp, palette_rgb, palette_lab, use_lab_flag)
            new_r = palette_rgb[idx, 0]
            new_g = palette_rgb[idx, 1]
            new_b = palette_rgb[idx, 2]
            # Compute quantization error.
            err_r = (old_r - new_r) * strength
            err_g = (old_g - new_g) * strength
            err_b = (old_b - new_b) * strength
            # Assign the new (quantized) color.
            img_array[y, x, 0] = new_r
            img_array[y, x, 1] = new_g
            img_array[y, x, 2] = new_b
            # Choose diffusion matrix based on the saliency at this pixel.
            if saliency_array[y, x] > 0.5:
                current_matrix = floyd_matrix
                num_neighbors = floyd_matrix.shape[0]
            else:
                current_matrix = atkinson_matrix
                num_neighbors = atkinson_matrix.shape[0]
            # Distribute the quantization error to neighboring pixels.
            for i in range(num_neighbors):
                dx = int(current_matrix[i, 0])
                dy = int(current_matrix[i, 1])
                coeff = current_matrix[i, 2]
                nx = x + dx
                ny = y + dy
                if nx >= 0 and nx < width and ny >= 0 and ny < height:
                    if alpha_mask[ny, nx]:
                        img_array[ny, nx, 0] += err_r * coeff
                        img_array[ny, nx, 1] += err_g * coeff
                        img_array[ny, nx, 2] += err_b * coeff

# ---------------------------------------------------------------------------
# The public hybrid dithering function.
#
# This routine:
#  1. Computes a saliency map from the image (using edge detection and blur).
#  2. Prepares the image (and alpha mask) as a float32 NumPy array.
#  3. Precomputes the palette (in RGB and, optionally, LAB).
#  4. Defines the Atkinson and Floyd–Steinberg diffusion matrices.
#  5. Calls the numba‑jitted hybrid_dither_numba function.
#  6. Clips and converts the result back to a PIL image.
# ---------------------------------------------------------------------------
@register_processing_method(
    'Hybrid Dither',
    default_params={'strength': 1.0},
    description="Switches between Atkinson and Floyd dithering based on texture."
)
def hybrid_dithering(img, color_key, params):
    global use_lab
    strength = params.get('strength', 0.75)
    
    # Generate a saliency map using edge detection and Gaussian blur.
    gray_img = img.convert('L')
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    saliency_map = edges.filter(ImageFilter.GaussianBlur(1.5))
    saliency_array = np.array(saliency_map, dtype=np.float32) / 255.0

    # Prepare image and alpha mask.
    has_alpha = (img.mode == 'RGBA')
    # Work in float32 for smoother error propagation.
    img_array = np.array(img, dtype=np.float32)
    height, width = img_array.shape[:2]
    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        alpha_mask = (alpha_channel > 0)
    else:
        alpha_mask = np.ones((height, width), dtype=np.bool_)

    # Precompute the palette in RGB.
    palette_rgb = np.array(list(color_key.values()), dtype=np.float32)
    
    # Precompute the LAB palette if needed.
    if use_lab:
        n = palette_rgb.shape[0]
        palette_lab = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            r = palette_rgb[i, 0]
            g = palette_rgb[i, 1]
            b = palette_rgb[i, 2]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            palette_lab[i, 0] = L
            palette_lab[i, 1] = a_val
            palette_lab[i, 2] = b_val
    else:
        palette_lab = np.empty((0, 3), dtype=np.float32)
    
    # Define diffusion matrices.
    # Atkinson (typically diffuses to 6 neighbors)
    atkinson = np.array([
        [ 1,  0, 1/8],
        [ 2,  0, 1/8],
        [-1,  1, 1/8],
        [ 0,  1, 1/8],
        [ 1,  1, 1/8],
        [ 0,  2, 1/8],
    ], dtype=np.float32)
    # Floyd–Steinberg (classic 4-neighbor pattern)
    floyd = np.array([
        [ 1,  0, 7/16],
        [-1,  1, 3/16],
        [ 0,  1, 5/16],
        [ 1,  1, 1/16],
    ], dtype=np.float32)
    
    # Call the numba‑accelerated hybrid dithering routine.
    hybrid_dither_numba(img_array, saliency_array, alpha_mask,
                        palette_rgb, palette_lab, use_lab,
                        atkinson, floyd, strength)
    
    # Clip the RGB channels to [0,255] and convert to uint8.
    img_array[:, :, :3] = np.clip(img_array[:, :, :3], 0, 255)
    img_array = img_array.astype(np.uint8)
    
    # Reattach alpha channel if needed.
    if has_alpha:
        result_img = Image.fromarray(img_array, mode='RGBA')
    else:
        result_img = Image.fromarray(img_array, mode='RGB')
    
    return result_img



@register_processing_method(
    'Pattern Dither',
    default_params={'strength': 0.33},
    description="Uses an 8x8 Bayer matrix to apply dithering in a pattern. Pretty :3"
)
def ordered_dithering(img, color_key, params):
    global use_lab
    strength = params.get('strength', 1.0)
    adjustment_factor = 0.3 * strength

    # 8x8 Bayer matrix
    bayer_8x8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ], dtype=np.float32) / 64.0

    img = img.copy()
    img_array = np.array(img, dtype=np.uint8)
    has_alpha = (img.mode == 'RGBA')
    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        alpha_mask = alpha_channel > 0
    else:
        alpha_mask = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.bool_)

    height, width = img_array.shape[:2]

    # Build palette
    palette_rgb = np.array(list(color_key.values()), dtype=np.uint8)
    if use_lab:
        n_palette = palette_rgb.shape[0]
        palette_lab = np.empty((n_palette, 3), dtype=np.float32)
        for i in range(n_palette):
            r, g, b = palette_rgb[i]
            L, a_val, b_val = rgb_to_lab_numba(r, g, b)
            palette_lab[i, 0] = L
            palette_lab[i, 1] = a_val
            palette_lab[i, 2] = b_val
    else:
        palette_lab = np.empty((0, 3), dtype=np.float32)

    # Tile the Bayer matrix
    tiled_bayer = np.tile(bayer_8x8, (height // 8 + 1, width // 8 + 1))
    tiled_bayer = tiled_bayer[:height, :width].astype(np.float32)

    # We'll separate into two specialized routines for clarity
    @njit(parallel=True, cache=True)
    def ordered_dithering_rgb(image, alpha_mask, tiled_bayer, adjustment_factor, palette_rgb):
        h, w = image.shape[:2]
        n_palette = palette_rgb.shape[0]
        for y in prange(h):
            for x in range(w):
                if not alpha_mask[y, x]:
                    continue
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                brightness = (r + g + b) / 765.0
                threshold = tiled_bayer[y, x]
                if brightness < threshold:
                    factor = 1.0 - adjustment_factor
                else:
                    factor = 1.0 + adjustment_factor

                r_adj = max(0, min(r * factor, 255))
                g_adj = max(0, min(g * factor, 255))
                b_adj = max(0, min(b * factor, 255))

                best_idx = 0
                best_dist = 1e10
                for i in range(n_palette):
                    dr = r_adj - palette_rgb[i, 0]
                    dg = g_adj - palette_rgb[i, 1]
                    db = b_adj - palette_rgb[i, 2]
                    dist = dr*dr + dg*dg + db*db
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                image[y, x, 0] = palette_rgb[best_idx, 0]
                image[y, x, 1] = palette_rgb[best_idx, 1]
                image[y, x, 2] = palette_rgb[best_idx, 2]

    @njit(parallel=True, cache=True)
    def ordered_dithering_lab(image, alpha_mask, tiled_bayer, adjustment_factor, palette_rgb, palette_lab):
        h, w = image.shape[:2]
        n_palette = palette_rgb.shape[0]
        for y in prange(h):
            for x in range(w):
                if not alpha_mask[y, x]:
                    continue
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                L, a_val, b_val = rgb_to_lab_numba(r, g, b)
                L_norm = L / 255.0
                threshold = tiled_bayer[y, x]
                if L_norm < threshold:
                    L_adj = L_norm - adjustment_factor
                    if L_adj < 0.0:
                        L_adj = 0.0
                else:
                    L_adj = L_norm + adjustment_factor
                    if L_adj > 1.0:
                        L_adj = 1.0
                L_new = L_adj * 255.0

                best_idx = 0
                best_dist = 1e10
                for i in range(n_palette):
                    dL = L_new - palette_lab[i, 0]
                    da = a_val - palette_lab[i, 1]
                    db = b_val - palette_lab[i, 2]
                    dist = dL*dL + da*da + db*db
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                image[y, x, 0] = palette_rgb[best_idx, 0]
                image[y, x, 1] = palette_rgb[best_idx, 1]
                image[y, x, 2] = palette_rgb[best_idx, 2]

    proc_img = img_array.astype(np.float32)
    if use_lab and palette_lab.shape[0] > 0:
        ordered_dithering_lab(proc_img, alpha_mask, tiled_bayer, adjustment_factor, palette_rgb.astype(np.float32), palette_lab)
    else:
        ordered_dithering_rgb(proc_img, alpha_mask, tiled_bayer, adjustment_factor, palette_rgb.astype(np.float32))

    proc_img = np.clip(proc_img, 0, 255).astype(np.uint8)
    if has_alpha:
        img_array[:, :, :3] = proc_img[:, :, :3]
        result_img = Image.fromarray(img_array, 'RGBA')
    else:
        result_img = Image.fromarray(proc_img[:, :, :3], 'RGB')

    return result_img



@register_processing_method(
    'Random Dither',
    default_params={'strength': 1.0, 'smoothing': False},
    description="Adds randomized dithering for a noisier but more natural texture."
)
def random_dithering(img, color_key, params):
    global use_lab
    strength = params.get('strength', 1.0)
    smoothing = params.get('smoothing', False)
    # The noise standard deviation (adjust as desired)
    noise_std = 32 * strength

    # Work on a copy of the image.
    img = img.copy()
    has_alpha = (img.mode == 'RGBA')
    img_array = np.array(img, dtype=np.uint8)
    
    # Separate RGB and alpha (if present)
    if has_alpha:
        rgb_data = img_array[:, :, :3]
        alpha_channel = img_array[:, :, 3]
    else:
        rgb_data = img_array

    height, width = rgb_data.shape[:2]
    
    if use_lab:
        # Convert the entire image to LAB using OpenCV (vectorized)
        lab_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2LAB)
        # Generate noise arrays for each channel.
        # For LAB, we use a lower noise std for the chromatic channels.
        l_noise = np.random.normal(0, noise_std * 0.5, size=(height, width))
        a_noise = np.random.normal(0, noise_std * 0.25, size=(height, width))
        b_noise = np.random.normal(0, noise_std * 0.25, size=(height, width))
        # Optionally smooth the noise to yield more natural transitions.
        if smoothing:
            l_noise = cv2.GaussianBlur(l_noise.astype(np.float32), (3, 3), 0)
            a_noise = cv2.GaussianBlur(a_noise.astype(np.float32), (3, 3), 0)
            b_noise = cv2.GaussianBlur(b_noise.astype(np.float32), (3, 3), 0)
        # Add the noise to the LAB image (working in float32 for precision)
        noisy_lab = lab_data.astype(np.float32)
        noisy_lab[:, :, 0] += l_noise
        noisy_lab[:, :, 1] += a_noise
        noisy_lab[:, :, 2] += b_noise
        # Clip the LAB values back into the 0-255 range and convert to uint8
        noisy_lab = np.clip(noisy_lab, 0, 255).astype(np.uint8)
        # Convert back to RGB using OpenCV (vectorized)
        noisy_rgb = cv2.cvtColor(noisy_lab, cv2.COLOR_LAB2RGB)
    else:
        # For RGB, generate noise for each channel
        noise = np.random.normal(0, noise_std, size=rgb_data.shape)
        if smoothing:
            noise = cv2.GaussianBlur(noise.astype(np.float32), (3, 3), 0)
        # Add the noise and clip to valid range
        noisy_rgb = rgb_data.astype(np.float32) + noise
        noisy_rgb = np.clip(noisy_rgb, 0, 255).astype(np.uint8)
    
    # Use the optimized, vectorized color mapping routine to snap each pixel
    # to its nearest palette color. (This function is assumed to be accelerated
    # with numba or other techniques.)
    mapped_rgb = find_closest_colors_image(noisy_rgb, color_key)
    
    # Reattach the alpha channel if needed.
    if has_alpha:
        mapped_data = np.dstack((mapped_rgb[:, :, :3], alpha_channel))
        result_mode = 'RGBA'
    else:
        mapped_data = mapped_rgb
        result_mode = 'RGB'
    
    return Image.fromarray(mapped_data, mode=result_mode)

@register_processing_method(
    'Atkinson Dither',
    default_params={'strength': 1.0},
    description="Dithering suited for smaller images! Used by the Macintosh for monochrome displays."
)
def atkinson_dithering(img, color_key, params):
    strength = params.get('strength', 1.0)
    diffusion_matrix = [
        (1, 0, 1 / 8), (2, 0, 1 / 8),
        (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ]
    return optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix)


@register_processing_method(
    'Jarvis Dither',
    default_params={'strength': 1.0},
    description="Applies diffusion over a large area. Best used for images with size over ~120."
)
def jarvis_judice_ninke_dithering(img, color_key, params):
    strength = params.get('strength', 1.0)
    diffusion_matrix = [
        (1, 0, 7 / 48), (2, 0, 5 / 48),
        (-2, 1, 3 / 48), (-1, 1, 5 / 48), (0, 1, 7 / 48), (1, 1, 5 / 48), (2, 1, 3 / 48),
        (-2, 2, 1 / 48), (-1, 2, 3 / 48), (0, 2, 5 / 48), (1, 2, 3 / 48), (2, 2, 1 / 48),
    ]
    return optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix)


@register_processing_method(
    'Stucki Dither',
    default_params={'strength': 1.0},
    description="An enhancement of Floyd-Steinberg with a wider diffusion matrix for less noisy results."
)
def stucki_dithering(img, color_key, params):
    strength = params.get('strength', 1.0)
    diffusion_matrix = [
        (1, 0, 8 / 42), (2, 0, 4 / 42),
        (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
        (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42),
    ]
    return optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix)


@register_processing_method(
    'Floyd Dither',
    default_params={'strength': 1.0},
    description="Create smooth gradients using diffusion. Best used for images with size over ~120."
)
def floyd_steinberg_dithering(img, color_key, params):
    strength = params.get('strength', 1.0)
    diffusion_matrix = [
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ]
    return optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix)


@register_processing_method(
    'Sierra Dither',
    default_params={'strength': 1.0},
    description="A Sierra variant that provides smooth gradients with less computational complexity."
)
def sierra2_dithering(img, color_key, params):
    strength = params.get('strength', 1.0)
    diffusion_matrix = [
        (1, 0, 4/16), (2, 0, 3/16),
        (-2, 1, 1/16), (-1, 1, 2/16), (0, 1, 3/16), (1, 1, 2/16), (2, 1, 1/16),
    ]
    return optimized_error_diffusion_dithering(img, color_key, strength, diffusion_matrix)




def process_image(img, color_key, process_mode, process_params):
    """
    Dispatches image processing to the appropriate method based on process_mode.
    """
    img = img.copy()
    if process_mode in processing_method_registry:
        processing_function = processing_method_registry[process_mode]
        return processing_function(img, color_key, process_params)
    else:
        # Default to color matching if unknown process_mode
        return color_matching(img, color_key, process_params)




def process_and_save_image(img, target_size, process_mode, use_lab_flag, process_params, color_key_array, remove_bg, preprocess_flag, progress_callback=None, message_callback=None, error_callback=None):
    """
    Processes and saves the image according to specified parameters.
    Ensures that only pixels with alpha > 191 are included in the output and preview.
    """
    try:
        
        if message_callback:
            message_callback("Preparing image...")

        # Prepare the image (convert to RGBA if needed)
        img = prepare_image(img)

        # Resize the image if needed
        if target_size is not None:
            img = resize_image(img, target_size)
            if message_callback:
                message_callback(f"Image resized to {img.size}")
        else:
            if message_callback:
                message_callback("Keeping original image dimensions.")


        # Preprocess the image if needed
        if preprocess_flag:
            img_np = np.array(img)
            img_np = preprocess_image(img_np, color_key_array, message_callback)
            if message_callback:
                message_callback("Image preprocessed.")
            img = Image.fromarray(img_np, 'RGBA')

        global brightness
        if brightness != 0.5:
            img = adjust_brightness(img, brightness)
            if message_callback:
                message_callback("Manual Brightness Adjusted")
        # Construct color_key from color_key_array
        color_key = build_color_key(color_key_array)
        
        img = process_image(img, color_key, process_mode, process_params)

        if message_callback:
            message_callback(f"Processing applied: {process_mode}")
            
        
        # Save a preview of the processed image in 'preview' folder
        create_and_clear_preview_folder(message_callback)

        # Apply transparency filtering for the preview
        img = img.copy()
        pixels = img.load()
        width, height = img.size
        for y in range(height):
            for x in range(width):
                pixel = pixels[x, y]
                if len(pixel) == 4 and pixel[3] <= 191:  # RGBA and alpha <= 191
                    pixels[x, y] = (0, 0, 0, 0)  # Make fully transparent

        # Save the preview image
        img = crop_to_solid_area(img)
        
        preview_path = exe_path_fs('game_data/stamp_preview/preview.png')
        save_image(img, preview_path, color_key_array)
        if message_callback:
            message_callback(f"Preview saved at: {preview_path}")

        # Save the processed image data to stamp.txt
        width, height = img.size
        scaled_width = round(width * 0.1, 1)
        scaled_height = round(height * 0.1, 1)

        current_dir = exe_path_fs('game_data/current_stamp_data')
        os.makedirs(current_dir, exist_ok=True)  # Ensure the directory exists
        output_file_path = exe_path_fs('game_data/current_stamp_data/stamp.txt')

        with open(output_file_path, 'w') as f:
            # Write the first line with scaled width, height, and 'img'
            f.write(f"{scaled_width},{scaled_height},img\n")
            if message_callback:
                message_callback(f"Scaled dimensions written: {scaled_width},{scaled_height},img")

            # Process each pixel
            pixels = img.load()
            for y in range(height - 1, -1, -1):  # Process from bottom to top
                for x in range(width):
                    try:
                        pixel = pixels[x, y]

                        # Handle both RGB and RGBA images
                        if len(pixel) == 4:  # RGBA
                            r, g, b, a = pixel
                        elif len(pixel) == 3:  # RGB
                            r, g, b = pixel
                            a = 255  # Assume fully opaque
                        else:
                            raise ValueError(f"Unexpected pixel format at ({x}, {y}): {pixel}")

                        if a <= 191:  # Skip pixels with alpha <= 191 (75% opacity)
                            continue

                        # Map the pixel to the closest color
                        closest_color_num = find_closest_color((r, g, b), color_key)

                        # Scale the coordinates
                        scaled_x = round(x * 0.1, 1)
                        scaled_y = round((height - 1 - y) * 0.1, 1)

                        # Write to the file
                        f.write(f"{scaled_x},{scaled_y},{closest_color_num}\n")

                    except Exception as e:
                        if message_callback:
                            message_callback(f"Error processing pixel at ({x}, {y}): {e}")

        if message_callback:
            message_callback(f"Processing complete! Output saved to: {output_file_path}")

    except Exception as e:
        if error_callback:
            error_callback(f"An error occurred: {e}")

def save_image(img, preview_path, color_key_array):
    """
    Processes an RGBA PIL image by mapping pixels to colors based on a color key array
    and a predefined COLOR key, then saves the result as a PNG.

    Parameters:
    - img: Input image as a PIL RGBA image.
    - preview_path: Path to save the processed image.
    - color_key_array: List of dictionaries with color information to match and transform.

    Returns:
    - None
    """
    # Define the COLOR key mapping numbers to new colors

    COLOR_KEY = {
        -1: (0, 0, 0, 0),         # FULL ALPHA (transparent)
        0: (255, 231, 197, 255),  # 'ffe7c5'
        1: (42, 56, 68, 255),     # '2a3844'
        2: (215, 11, 93, 255),    # 'd70b5d'
        3: (13, 179, 158, 255),   # '0db39e'
        4: (244, 192, 9, 255),    # 'f4c009'
        5: (255, 0, 255, 255),    # 'ff00ff'
        6: (186, 195, 87, 255),   # 'bac357'
        7: (163, 178, 210, 255),  # '#a3b2d2'
        8: (214, 206, 194, 255),  # '#d6cec2'
        9: (191, 222, 216, 255),  # '#bfded8'
        10: (169, 196, 132, 255), # '#a9c484'
        11: (93, 147, 123, 255),  # '#5d937b'
        12: (162, 166, 169, 255), # '#a2a6a9'
        13: (119, 127, 143, 255), # '#777f8f'
        14: (234, 178, 129, 255), # '#eab281'
        15: (234, 114, 134, 255), # '#ea7286'
        16: (244, 164, 191, 255), # '#f4a4bf'
        17: (160, 124, 167, 255), # '#a07ca7'
        18: (191, 121, 109, 255), # '#bf796d'
        19: (245, 209, 182, 255), # '#f5d1b6'
        20: (227, 225, 159, 255), # '#e3e19f'
        21: (255, 223, 0, 255),   # '#ffdf00'
        22: (255, 191, 0, 255),   # '#ffbf00'
        23: (196, 180, 84, 255),  # '#c4b454'
        24: (245, 222, 179, 255), # '#f5deb3'
        25: (244, 196, 48, 255),  # '#f4c430'
        26: (0, 255, 255, 255),   # '#00ffff'
        27: (137, 207, 240, 255), # '#89cff0'
        28: (77, 77, 255, 255),   # '#4d4dff'
        29: (0, 0, 139, 255),     # '#00008b'
        30: (65, 105, 225, 255),  # '#4169e1'
        31: (0, 103, 66, 255),    # '#006742'
        32: (76, 187, 23, 255),   # '#4cbb17'
        33: (46, 111, 64, 255),   # '#2e6f40'
        34: (46, 139, 87, 255),   # '#2e8b57'
        35: (192, 192, 192, 255), # '#c0c0c0'
        36: (129, 133, 137, 255), # '#818589'
        37: (137, 148, 153, 255), # '#899499'
        38: (112, 128, 144, 255), # '#708090'
        39: (255, 165, 0, 255),   # '#ffa500'
        40: (255, 140, 0, 255),   # '#ff8c00'
        41: (215, 148, 45, 255),  # '#d7942d'
        42: (255, 95, 31, 255),   # '#ff5f1f'
        43: (204, 119, 34, 255),  # '#cc7722'
        44: (255, 105, 180, 255), # '#ff69b4'
        45: (255, 16, 240, 255),  # '#ff10f0'
        46: (170, 51, 106, 255),  # '#aa336a'
        47: (244, 180, 196, 255), # '#f4b4c4'
        48: (149, 53, 83, 255),   # '#953553'
        49: (216, 191, 216, 255), # '#d8bfd8'
        50: (127, 0, 255, 255),   # '#7f00ff'
        51: (128, 0, 128, 255),   # '#800080'
        52: (255, 36, 0, 255),    # '#ff2400'
        53: (255, 68, 51, 255),   # '#ff4433'
        54: (165, 42, 42, 255),   # '#a52a2a'
        55: (145, 56, 49, 255),   # '#913831'
        56: (255, 0, 0, 255),     # '#ff0000'
        57: (59, 34, 25, 255),    # '#3b2219'
        58: (161, 110, 75, 255),  # '#a16e4b'
        59: (212, 170, 120, 255), # '#d4aa78'
        60: (230, 188, 152, 255), # '#e6bc98'
        61: (255, 231, 209, 255)  # '#ffe7d1'
    }

    # Create a mapping from hex color to number using color_key_array
    hex_to_number = {entry['hex']: entry['number'] for entry in color_key_array}

    # Convert the image to a NumPy array for processing
    img_array = np.array(img)  # Shape: (H, W, 4) for RGBA

    # Separate alpha channel for transparency handling
    rgb_array = img_array[:, :, :3]
    alpha_channel = img_array[:, :, 3]

    # Convert RGB values to hex for matching
    def rgb_to_hex(rgb):
        return '{:02x}{:02x}{:02x}'.format(*rgb)

    # Initialize output array
    output_array = np.zeros_like(img_array)

    # Process each pixel
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            pixel_rgb = tuple(rgb_array[y, x])  # Current pixel RGB
            pixel_hex = rgb_to_hex(pixel_rgb)  # Convert to hex

            if alpha_channel[y, x] == 0:  # Fully transparent
                output_array[y, x] = COLOR_KEY[-1]
            elif pixel_hex in hex_to_number:  # Match found in color_key_array
                color_number = hex_to_number[pixel_hex]
                output_array[y, x] = COLOR_KEY.get(color_number, (0, 0, 0, 255))  # Default to black if not found
            else:  # No match, leave it as black
                output_array[y, x] = (0, 0, 0, 255)

    # Convert the output array back to a PIL image
    output_img = Image.fromarray(output_array, mode='RGBA')

    # Save the processed image to the preview path
    output_img.save(preview_path, format='PNG')





# Helper function to convert hex to RGB
def hex_to_rgb(hex_str):
    """
    Converts a hexadecimal color string to an RGB tuple.

    Args:
        hex_str (str): A 6-character hexadecimal string, e.g., 'ffe7c5'.

    Returns:
        tuple: A tuple of integers representing the RGB values, e.g., (255, 231, 197).
    """
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 0, 0)  # Default to black on error

# Numba-accelerated function to find the closest color number
@numba.njit
def find_closest_color2(pixel, color_key_array, color_key_numbers):
    """
    Finds the closest color number from color_key_array for the given pixel.

    Args:
        pixel (np.ndarray): An array representing the RGB values of the pixel, e.g., [255, 231, 197].
        color_key_array (np.ndarray): A 2D array of RGB values for the color key.
        color_key_numbers (np.ndarray): A 1D array of color numbers corresponding to color_key_array.

    Returns:
        int: The color number of the closest color in color_key_array.
    """
    min_dist = 1e10
    min_index = -1
    for i in range(color_key_array.shape[0]):
        dist = 0.0
        for j in range(3):
            diff = color_key_array[i, j] - pixel[j]
            dist += diff * diff
        if dist < min_dist:
            min_dist = dist
            min_index = i
    if min_index != -1:
        return color_key_numbers[min_index]
    else:
        return -1  # Indicates no match found

def process_and_save_gif(
    image_path,
    target_size,
    process_mode,
    use_lab_flag,
    process_params,
    color_key_array,
    remove_bg,
    preprocess_flag,
    progress_callback=None,
    message_callback=None,
    error_callback=None
):
    """
    Processes and saves an animated image (GIF or WebP) according to specified parameters.
    Utilizes NumPy and Numba for optimized performance.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size for the frames (width, height).
        process_mode (str): Processing mode.
        use_lab_flag (bool): Flag to use LAB color space.
        process_params (dict): Additional processing parameters.
        color_key_array (list): List of color key dictionaries with 'number' and 'hex' keys.
        remove_bg (bool): Flag to remove background.
        preprocess_flag (bool): Flag to preprocess the image.
        progress_callback (function, optional): Callback for progress updates.
        message_callback (function, optional): Callback for messages.
        error_callback (function, optional): Callback for errors.
    """
    try:
        set_gif_ready_false()

        # Open frames.txt and clear its contents
        current_dir = exe_path_fs('game_data/current_stamp_data')
        os.makedirs(current_dir, exist_ok=True)  # Ensure the directory exists
        frames_txt_path = os.path.join(current_dir, 'frames.txt')
        with open(frames_txt_path, 'w'):
            pass  # Clears the file

        # Open the image
        img = Image.open(image_path)
        if not getattr(img, "is_animated", False):
            message = "Selected file is not an animated image with multiple frames."
            if message_callback:
                message_callback(message)
            img.close()
            return

        total_frames = img.n_frames
        if message_callback:
            message_callback(f"Total frames in image: {total_frames}")

        # Gather delays for each frame
        delays = get_frame_delays(image_path)
        # Determine delay uniformity
        uniform_delay = delays[0] if all(d == delays[0] for d in delays) else -1

        # Save frames to 'Frames' directory (and clear it first)
        save_frames(
            img,
            target_size,
            process_mode,
            use_lab_flag,
            process_params,
            remove_bg,
            preprocess_flag,
            color_key_array,
            progress_callback,
            message_callback,
            error_callback
        )

        # Create and clear 'preview' folder
        preview_folder = create_and_clear_preview_folder(message_callback)

        # Construct color_key_rgb and color_key_numbers from color_key_array
        color_key_rgb = np.array([hex_to_rgb(color['hex']) for color in color_key_array], dtype=np.float32)
        color_key_numbers = np.array([color['number'] for color in color_key_array], dtype=np.int32)

        # Load the first frame
        first_frame_path = exe_path_fs('game_data/frames/frame_1.png')

        if not os.path.exists(first_frame_path):
            error_message = f"First frame not found at {first_frame_path}"
            if error_callback:
                error_callback(error_message)
            img.close()
            return

        # Process first frame and write to stamp.txt
        with Image.open(first_frame_path) as first_frame:
            width, height = first_frame.size
            scaled_width = round(width * 0.1, 1)  # Multiply by 0.1
            scaled_height = round(height * 0.1, 1)  # Multiply by 0.1

            # Write header to stamp.txt
            stamp_txt_path = os.path.join(current_dir, 'stamp.txt')
            with open(stamp_txt_path, 'w') as stamp_file:
                stamp_file.write(f"{scaled_width},{scaled_height},gif,{total_frames},{uniform_delay}\n")
                if message_callback:
                    message_callback(f"Header written to stamp.txt: {scaled_width},{scaled_height},gif,{total_frames},{uniform_delay}")

                # Convert frame to NumPy array
                frame_array = np.array(first_frame.convert('RGBA'))
                alpha_channel = frame_array[:, :, 3]
                mask = alpha_channel > 191  # Opaque pixels
                rgb_array = frame_array[:, :, :3].astype(np.float32)

                # Initialize Frame1Array and first_frame_pixels as 2D arrays
                Frame1Array = -1 * np.ones((height, width), dtype=np.int32)
                first_frame_pixels = -1 * np.ones((height, width), dtype=np.int32)

                for y in range(height):
                    for x in range(width):
                        if mask[y, x]:
                            pixel = rgb_array[y, x]
                            color_num = find_closest_color2(pixel, color_key_rgb, color_key_numbers)
                            Frame1Array[y, x] = color_num
                            first_frame_pixels[y, x] = color_num

                            # Scale the coordinates
                            scaled_x = round(x * 0.1, 1)
                            scaled_y = round(y * 0.1, 1)

                            # Write to stamp.txt
                            stamp_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

        # Process subsequent frames
        header_frame_number = 1  # Start header numbering from 1

        for frame_number in range(2, total_frames + 1):  # Start from frame 2
            frame_path = exe_path_fs(f'game_data/frames/frame_{frame_number}.png')
            if not os.path.exists(frame_path):
                if message_callback:
                    message_callback(f"Frame {frame_number} not found at {frame_path}")
                continue

            with Image.open(frame_path) as frame:
                frame_array = np.array(frame.convert('RGBA'))
                alpha_channel = frame_array[:, :, 3]
                mask = alpha_channel > 191  # Opaque pixels
                rgb_array = frame_array[:, :, :3].astype(np.float32)

                # Initialize CurrentFrameArray
                CurrentFrameArray = -1 * np.ones((height, width), dtype=np.int32)

                for y in range(height):
                    for x in range(width):
                        if mask[y, x]:
                            pixel = rgb_array[y, x]
                            color_num = find_closest_color2(pixel, color_key_rgb, color_key_numbers)
                            CurrentFrameArray[y, x] = color_num

                # Find differences between CurrentFrameArray and Frame1Array
                diffs = np.argwhere(CurrentFrameArray != Frame1Array)

                # Write header and diffs to frames.txt
                with open(frames_txt_path, 'a') as frames_file:
                    # Include delay if variable delays
                    if uniform_delay == -1:
                        frame_delay = delays[frame_number - 1]
                        frames_file.write(f"frame,{header_frame_number},{frame_delay}\n")
                    else:
                        frames_file.write(f"frame,{header_frame_number}\n")

                    for y, x in diffs:
                        color_num = CurrentFrameArray[y, x]
                        # Scale coordinates by multiplying by 0.1
                        scaled_x = round(x * 0.1, 1)
                        scaled_y = round(y * 0.1, 1)
                        frames_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

                # Update Frame1Array
                Frame1Array = CurrentFrameArray.copy()

                # Update progress
                if progress_callback:
                    progress = (frame_number - 1) / total_frames * 100
                    progress_callback(progress)

                header_frame_number += 1  # Increment header frame number

        # After processing all frames, compare last frame to first frame to complete the loop
        diffs = np.argwhere(Frame1Array != first_frame_pixels)

        # Write header and diffs to frames.txt for the final loop
        with open(frames_txt_path, 'a') as frames_file:
            # Include delay if variable delays
            final_frame_number = header_frame_number
            if uniform_delay == -1:
                frame_delay = delays[0]  # Use the delay of the first frame
                frames_file.write(f"frame,{final_frame_number},{frame_delay}\n")
            else:
                frames_file.write(f"frame,{final_frame_number}\n")

            for y, x in diffs:
                color_num = first_frame_pixels[y, x]
                # Scale coordinates by multiplying by 0.1
                scaled_x = round(x * 0.1, 1)
                scaled_y = round(y * 0.1, 1)
                frames_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

        if message_callback:
            message_callback(f"Processing of animated image frames complete! Data saved to: {frames_txt_path}")

        # Generate preview GIF after all processing is done
        create_preview_gif(
            total_frames,
            delays,
            preview_folder,
            color_key_array,
            progress_callback,
            message_callback,
            error_callback
        )
        set_gif_ready_true()
        img.close()  # Close the image after processing
        print("GIF processing finished successfully.")

    except Exception as e:
        error_message = f"An error occurred in process_and_save_gif: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        if error_callback:
            error_callback(error_message)



def process_and_save_video(
    video_path,
    target_size,
    process_mode,
    use_lab_flag,
    process_params,
    color_key_array,
    remove_bg,
    preprocess_flag,
    progress_callback=None,
    message_callback=None,
    error_callback=None
):
    """
    Processes and saves an MP4 video into 'stamp.txt' and 'frames.txt' exactly the same way
    as GIF frames, but at a maximum of 8 FPS (discarding frames beyond 8 FPS to preserve 
    overall playback speed).

    Args:
        video_path (str): Path to the input video (mp4).
        target_size (tuple): Target size for the frames (width, height).
        process_mode (str): Processing mode.
        use_lab_flag (bool): Flag to use LAB color space.
        process_params (dict): Additional processing parameters.
        color_key_array (list): List of color key dictionaries with 'number' and 'hex' keys.
        remove_bg (bool): Flag to remove background.
        preprocess_flag (bool): Flag to preprocess the frames.
        progress_callback (function, optional): Callback for progress updates.
        message_callback (function, optional): Callback for messages.
        error_callback (function, optional): Callback for errors.
    """
    try:
        # ---------------------------------------------------------------------
        # 1) Setup: Prepare environment, files, etc.
        # ---------------------------------------------------------------------
        set_gif_ready_false()

        current_dir = exe_path_fs('game_data/current_stamp_data')
        os.makedirs(current_dir, exist_ok=True)

        frames_txt_path = os.path.join(current_dir, 'frames.txt')
        # Clear frames.txt
        with open(frames_txt_path, 'w'):
            pass

        # Clear and prepare 'Frames' directory
        frames_dir = exe_path_fs('game_data/frames')
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

        # Create and clear 'preview' folder
        preview_folder = create_and_clear_preview_folder(message_callback)

        # ---------------------------------------------------------------------
        # Build the color key from the provided array (fix for "color key is not defined")
        # ---------------------------------------------------------------------
        color_key = build_color_key(color_key_array)

        # ---------------------------------------------------------------------
        # 2) Open the video with OpenCV, determine FPS, and set up skipping
        # ---------------------------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            # Fallback in case OpenCV fails to read FPS properly
            original_fps = 30.0

        # We will capture at most 8 FPS, preserving the total duration:
        final_fps = min(original_fps, 8)
        # Skip factor (approx.) to drop frames if above 8 FPS
        skip_factor = round(original_fps / final_fps) if final_fps > 0 else 1

        # For simpler logic, we treat the final delay as uniform (in milliseconds).
        # Example: at 8 FPS, the delay is 125ms per frame; at 5 FPS, 200ms, etc.
        final_delay_ms = int(round(1000.0 / final_fps))

        if message_callback:
            message_callback(f"Original video FPS: {original_fps:.2f}")
            message_callback(f"Final FPS (capped at 8): {final_fps:.2f}")
            message_callback(f"Skip factor: {skip_factor} (frames are discarded accordingly).")

        # ---------------------------------------------------------------------
        # 3) Extract frames, resize/process, and save them as PNG
        # ---------------------------------------------------------------------
        kept_frames_paths = []
        frame_index = 0
        kept_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Decide whether we keep this frame (based on skip_factor)
            if frame_index % skip_factor == 0:
                kept_count += 1
                # Convert BGR to RGB and create a Pillow image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Make a writable copy of the image
                frame = pil_img.copy()

                # Prepare the image (convert to RGBA if needed)
                frame = prepare_image(frame)

                # Resize the image if needed
                if target_size is not None:
                    frame = resize_image(frame, target_size)

                # Preprocess the image if needed
                if preprocess_flag:
                    frame_np = np.array(frame)
                    frame_np = preprocess_image(frame_np, color_key_array, message_callback, True)
                    frame = Image.fromarray(frame_np, 'RGBA')

                global brightness
                if brightness != 0.5:
                    frame = adjust_brightness(frame, brightness)
                    if message_callback:
                        message_callback("Manual Brightness Adjusted")

                # Apply the processing method to the frame using the built color key
                frame = process_image(frame, color_key, process_mode, process_params)

                # Ensure the image is in RGBA mode and is writable
                if frame.mode != 'RGBA':
                    frame = frame.convert('RGBA')
                else:
                    frame = frame.copy()

                # Process translucent pixels: any pixel below 80% opacity is made fully transparent
                pixels = frame.load()
                width, height = frame.size
                opacity_threshold = 204  # 80% of 255
                for y in range(height):
                    for x in range(width):
                        r, g, b, a = pixels[x, y]
                        if a < opacity_threshold:
                            pixels[x, y] = (0, 0, 0, 0)

                # Save frame as PNG
                frame_filename = f"frame_{kept_count}.png"
                frame_path = os.path.join(frames_dir, frame_filename)
                frame.save(frame_path)
                kept_frames_paths.append(frame_path)

                # Progress callback
                if progress_callback:
                    progress = (frame_index / total_frames_in_video) * 100
                    progress_callback(progress)

            frame_index += 1
            if frame_index >= total_frames_in_video:
                break

        cap.release()  # Close the video file

        total_kept_frames = len(kept_frames_paths)
        if total_kept_frames == 0:
            raise ValueError("No frames were extracted. Check if the video is valid and skip factor wasn't too large.")

        if message_callback:
            message_callback(f"Total frames in video: {total_frames_in_video}")
            message_callback(f"Frames kept for final processing: {total_kept_frames}")

        # ---------------------------------------------------------------------
        # 4) Generate uniform delays array
        # ---------------------------------------------------------------------
        delays = [final_delay_ms] * total_kept_frames
        uniform_delay = delays[0]  # Uniform delay for all frames

        # ---------------------------------------------------------------------
        # 5) Prepare 'stamp.txt' by analyzing the first kept frame
        # ---------------------------------------------------------------------
        first_frame_path = kept_frames_paths[0]
        if not os.path.exists(first_frame_path):
            raise FileNotFoundError(f"First frame not found at {first_frame_path}")

        # Convert color_key_array to NumPy arrays for fast color matching
        color_key_rgb = np.array([hex_to_rgb(color['hex']) for color in color_key_array], dtype=np.float32)
        color_key_numbers = np.array([color['number'] for color in color_key_array], dtype=np.int32)

        stamp_txt_path = os.path.join(current_dir, 'stamp.txt')
        with open(stamp_txt_path, 'w') as stamp_file:
            with Image.open(first_frame_path) as first_frame:
                width, height = first_frame.size
                # Scale dimensions by 0.1
                scaled_width = round(width * 0.1, 1)
                scaled_height = round(height * 0.1, 1)

                # Write header: scaled_width, scaled_height, "gif", total_frames, uniform_delay
                stamp_file.write(f"{scaled_width},{scaled_height},gif,{total_kept_frames},{uniform_delay}\n")
                if message_callback:
                    message_callback(f"Header => {scaled_width},{scaled_height},gif,{total_kept_frames},{uniform_delay}")

                # Convert to RGBA and extract pixel data
                frame_array = np.array(first_frame.convert('RGBA'))
                alpha_channel = frame_array[:, :, 3]
                mask = alpha_channel > 191  # Opaque pixel threshold
                rgb_array = frame_array[:, :, :3].astype(np.float32)

                # Prepare arrays to track the reference and store original values
                Frame1Array = -1 * np.ones((height, width), dtype=np.int32)
                first_frame_pixels = -1 * np.ones((height, width), dtype=np.int32)

                for y in range(height):
                    for x in range(width):
                        if mask[y, x]:
                            pixel = rgb_array[y, x]
                            color_num = find_closest_color2(pixel, color_key_rgb, color_key_numbers)
                            Frame1Array[y, x] = color_num
                            first_frame_pixels[y, x] = color_num

                            # Write scaled pixel coordinates and color number
                            scaled_x = round(x * 0.1, 1)
                            scaled_y = round(y * 0.1, 1)
                            stamp_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

        # ---------------------------------------------------------------------
        # 6) Build 'frames.txt' by comparing each subsequent frame to the reference
        # ---------------------------------------------------------------------
        header_frame_number = 1  # Frame header counter

        for idx in range(1, total_kept_frames):
            current_path = kept_frames_paths[idx]
            with Image.open(current_path) as frame:
                frame_array = np.array(frame.convert('RGBA'))
                alpha_channel = frame_array[:, :, 3]
                mask = alpha_channel > 191
                rgb_array = frame_array[:, :, :3].astype(np.float32)

                CurrentFrameArray = -1 * np.ones((height, width), dtype=np.int32)
                for y in range(height):
                    for x in range(width):
                        if mask[y, x]:
                            pixel = rgb_array[y, x]
                            color_num = find_closest_color2(pixel, color_key_rgb, color_key_numbers)
                            CurrentFrameArray[y, x] = color_num

                # Identify differences between current frame and reference
                diffs = np.argwhere(CurrentFrameArray != Frame1Array)

                with open(frames_txt_path, 'a') as frames_file:
                    # Write frame header; if uniform_delay were variable, we could include it here
                    if uniform_delay == -1:
                        frame_delay = delays[idx]
                        frames_file.write(f"frame,{header_frame_number},{frame_delay}\n")
                    else:
                        frames_file.write(f"frame,{header_frame_number}\n")

                    for (dy, dx) in diffs:
                        color_num = CurrentFrameArray[dy, dx]
                        scaled_x = round(dx * 0.1, 1)
                        scaled_y = round(dy * 0.1, 1)
                        frames_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

                # Update the reference frame for the next comparison
                Frame1Array = CurrentFrameArray.copy()

                if progress_callback:
                    progress = (idx / total_kept_frames) * 100
                    progress_callback(progress)

                header_frame_number += 1

        # ---------------------------------------------------------------------
        # 7) Compare final frame to the first frame to "close the loop"
        # ---------------------------------------------------------------------
        diffs = np.argwhere(Frame1Array != first_frame_pixels)
        with open(frames_txt_path, 'a') as frames_file:
            final_frame_number = header_frame_number
            if uniform_delay == -1:
                frame_delay = delays[0]
                frames_file.write(f"frame,{final_frame_number},{frame_delay}\n")
            else:
                frames_file.write(f"frame,{final_frame_number}\n")

            for (dy, dx) in diffs:
                color_num = first_frame_pixels[dy, dx]
                scaled_x = round(dx * 0.1, 1)
                scaled_y = round(dy * 0.1, 1)
                frames_file.write(f"{scaled_x},{scaled_y},{color_num}\n")

        if message_callback:
            message_callback(f"Video frames processed! Data saved to: {frames_txt_path}")

        # ---------------------------------------------------------------------
        # 8) Create a preview GIF from the extracted frames (optional)
        # ---------------------------------------------------------------------
        create_preview_gif(
            total_kept_frames,
            delays,
            preview_folder,
            color_key_array,
            progress_callback,
            message_callback,
            error_callback
        )

        set_gif_ready_true()
        if message_callback:
            message_callback("Video processing finished successfully.")

    except Exception as e:
        error_message = f"An error occurred in process_and_save_video: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        if error_callback:
            error_callback(error_message)



def get_frame_delays(image_path):
    """
    Retrieves the delay (in milliseconds) for each frame in an animated GIF or WebP image.
    If the image is a WebP, it converts it to GIF first to extract frame delays.

    For WebP images, it keeps the valid frame durations and replaces missing or invalid
    durations with the average of the valid durations.

    Args:
        image_path (str): Path to the animated image file.

    Returns:
        list: A list of delays for each frame in milliseconds.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file format is unsupported or corrupted.
        RuntimeError: If an error occurs during image processing.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File '{image_path}' does not exist.")

    file_ext = os.path.splitext(image_path)[-1].lower()
    if file_ext not in [".gif", ".webp"]:
        raise ValueError("Unsupported file format. Only GIF and WebP are supported.")

    delays = []

    try:
        if file_ext == ".gif":
            # Handle GIFs directly
            with Image.open(image_path) as img:
                total_frames = getattr(img, "n_frames", 1)
                print(f"Total frames: {total_frames}")

                for frame_number, frame in enumerate(ImageSequence.Iterator(img)):
                    duration = frame.info.get('duration', None)

                    if duration is None:
                        duration_ms = 100  # Default to 100ms if missing
                    else:
                        duration_ms = int(duration)

                    # Validate duration
                    if duration_ms <= 0:
                        duration_ms = 100  # Default to 100ms if invalid

                    delays.append(duration_ms)

        elif file_ext == ".webp":
            # Handle WebP by extracting frames and durations
            with Image.open(image_path) as img:
                if not getattr(img, "is_animated", False):
                    raise ValueError("WebP image is not animated.")

                total_frames = getattr(img, "n_frames", 1)
                print(f"Total frames: {total_frames}")

                # Extract frames and durations
                frames = []
                durations = []
                for frame_number, frame in enumerate(ImageSequence.Iterator(img)):
                    duration = frame.info.get('duration', None)

                    if duration is None:
                        duration_ms = None  # Mark as missing
                    else:
                        duration_ms = int(duration)  # Treat duration as milliseconds

                    # Validate duration
                    if duration_ms is not None and duration_ms <= 0:
                        duration_ms = None  # Mark as invalid

                    # Append frame and duration
                    frames.append(frame.copy())
                    durations.append(duration_ms)

                # Calculate average delay from valid durations
                valid_durations = [d for d in durations if d is not None and d > 0]
                if not valid_durations:
                    average_delay = 100  # Fallback average if all durations are invalid
                    print("All frame durations are invalid or missing. Using fallback average delay of 100ms.")
                else:
                    average_delay = int(sum(valid_durations) / len(valid_durations))
                    print(f"Average delay calculated from valid durations: {average_delay}ms")

                # Assign delays: keep valid durations, replace missing/invalid with average
                delays = [
                    duration if (duration is not None and duration > 0) else average_delay
                    for duration in durations
                ]
                print(f"Assigned delays to all {len(delays)} frames.")

        return delays

    except Exception as e:
        raise RuntimeError(f"Error processing image '{image_path}': {e}")
    

def save_frames(img, target_size, process_mode, use_lab_flag, process_params, remove_bg, preprocess_flag, color_key_array,
                progress_callback, message_callback, error_callback):
    """
    Saves each frame of the animated image to the 'Frames' directory after preprocessing, resizing, and applying the selected processing method.
    """
    try:

        output_folder = exe_path_fs("game_data/frames")
        os.makedirs(output_folder, exist_ok=True)

        # Delete the contents of the 'Frames' folder before starting
        for filename in os.listdir(output_folder):
            file_path = output_folder / filename
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                if message_callback:
                    message_callback(f'Failed to delete {file_path}. Reason: {e}')

        total_frames = img.n_frames
        if message_callback:
            message_callback(f"Processing and saving {total_frames} frames...")

        # Construct color_key from color_key_array
        color_key = build_color_key(color_key_array)
     
        

        for frame_number in range(1, total_frames + 1):  # Start frame numbering from 1
            img.seek(frame_number - 1)
            frame = img.copy()  # Ensure we have a writable copy of the frame

            # Prepare the image (convert to RGBA if needed)
            frame = prepare_image(frame)


            # Resize the image if needed
            if target_size is not None:
                frame = resize_image(frame, target_size)

            # Preprocess the image if needed
            if preprocess_flag:
                frame_np = np.array(frame)
                frame_np = preprocess_image(frame_np, color_key_array, message_callback, True)
                frame = Image.fromarray(frame_np, 'RGBA')

            global brightness
            if brightness != 0.5:
                frame = adjust_brightness(frame, brightness)
                if message_callback:
                    message_callback("Manual Brightness Adjusted")
            # Apply the processing method to the frame
            frame = process_image(frame, color_key, process_mode, process_params)
            # Handle translucent pixels by creating a writable copy
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')  # Ensure image is in RGBA mode
            else:
                frame = frame.copy()  # Make a writable copy if it's already RGBA

            pixels = frame.load()
            width, height = frame.size
            opacity_threshold = 204  # 80% opacity (255 * 0.8)

            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixels[x, y]
                    if a < opacity_threshold:
                        pixels[x, y] = (0, 0, 0, 0)  # Fully transparent pixel

            # Save the frame
            frame_file = output_folder / f"frame_{frame_number}.png"
            save_image(frame, frame_file, color_key_array)

            if progress_callback:
                progress = frame_number / total_frames * 100
                progress_callback(progress)

        if message_callback:
            message_callback(f"All frames processed and saved to '{output_folder}'.")

    except Exception as e:
        if error_callback:
            error_callback(f"An error occurred while saving frames: {e}")

def create_preview_gif(total_frames, delays, preview_folder, color_key_array, progress_callback=None, message_callback=None, error_callback=None):
    """
    Creates a new GIF using the frames in the 'Frames' directory and the delay data,
    then saves it as 'preview.gif' in the 'preview' folder.
    """
    try:
        frames_folder = exe_path_fs('game_data/frames')
        output_gif_path = exe_path_fs('game_data/stamp_preview/preview.gif')
        color_key_array = 1

        frames = []
        frame_durations = []

        # Ensure we process exactly 500 frames if more are available
        frame_paths = [frames_folder / f"frame_{i}.png" for i in range(1, total_frames + 1)]
        valid_frame_paths = [path for path in frame_paths if os.path.exists(path)]
        valid_frame_paths = valid_frame_paths[:500]

        for frame_number, frame_path in enumerate(valid_frame_paths, start=1):
            frame = Image.open(frame_path).convert('RGBA')
            frames.append(frame)
            frame_durations.append(delays[frame_number - 1])  # Duration in ms

        if not frames:
            if error_callback:
                error_callback("No frames found to create preview GIF.")
            return

        # Prepare frames for GIF with transparency
        converted_frames = []
        for frame in frames:
            # Ensure the frame has an alpha channel
            frame = frame.convert('RGBA')

            # Create a transparent background
            background = Image.new('RGBA', frame.size, (0, 0, 0, 0))  # Transparent background

            # Composite the frame onto the background
            combined = Image.alpha_composite(background, frame)

            # Convert to 'P' mode (palette) with an adaptive palette
            # The 'transparency' parameter will handle the transparent color
            combined_p = combined.convert('P', palette=Image.ADAPTIVE, colors=255)

            # Find the color index that should be transparent
            # Here, we assume that the first color in the palette is transparent
            # Alternatively, you can search for a specific color
            # For robustness, let's search for the color with alpha=0
            transparent_color = None
            for idx, color in enumerate(combined_p.getpalette()[::3]):
                r = combined_p.getpalette()[idx * 3]
                g = combined_p.getpalette()[idx * 3 + 1]
                b = combined_p.getpalette()[idx * 3 + 2]
                # Check if this color is used for transparency in the original image
                # This is a simplistic check; for complex images, more logic may be needed
                if (r, g, b) == (0, 0, 0):  # Assuming black is the transparent color
                    transparent_color = idx
                    break

            if transparent_color is None:
                # If not found, append black to the palette and set it as transparent
                combined_p.putpalette(combined_p.getpalette() + [0, 0, 0])
                transparent_color = len(combined_p.getpalette()) // 3 - 1

            # Assign the transparency index
            combined_p.info['transparency'] = transparent_color

            converted_frames.append(combined_p)

        # Save the frames as a GIF with transparency
        converted_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=converted_frames[1:],
            duration=frame_durations,
            loop=0,
            transparency=converted_frames[0].info['transparency'],
            disposal=2
        )

        if message_callback:
            message_callback(f"Preview GIF saved at: {output_gif_path}")

    except Exception as e:
        if error_callback:
            error_callback(f"An error occurred in create_preview_gif: {e}")



def create_and_clear_preview_folder(message_callback=None):
    """
    Creates and clears the 'preview' folder.
    Returns the path to the 'preview' folder.
    """
    preview_folder = exe_path_fs('game_data/stamp_preview')
    os.makedirs(preview_folder, exist_ok=True)

    # Clear 'preview' folder
    for filename in os.listdir(preview_folder):
        file_path = preview_folder / filename
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            if message_callback:
                message_callback(f'Failed to delete {file_path}. Reason: {e}')
    return preview_folder



def main(image_path, remove_bg, preprocess_flag, use_lab_flag, brightness_flag, resize_dim, color_key_array, process_mode, process_params, progress_callback=None, message_callback=None, error_callback=None):
    """
    Main function to process the image, animated GIF, or video based on the provided parameters.
    """
    global use_lab
    use_lab = use_lab_flag
    global brightness
    mid_point = 0.5
    lower_bound = -0.53
    upper_bound = 1.53
    global chalks_colors
    chalks_colors = remove_bg
    # Calculate the brightness delta
    brightness = brightness_flag
    
    try:
        if message_callback:
            message_callback("Initializing...")

        # Handle 'clip' as image_path
        if image_path == 'clip':
            try:
                img = ImageGrab.grabclipboard()
                if img is None:
                    if error_callback:
                        error_callback("No image found in clipboard.")
                    return
                elif isinstance(img, list):
                    # Clipboard contains file paths
                    image_files = [f for f in img if os.path.isfile(f)]
                    if not image_files:
                        if error_callback:
                            error_callback("No image files found in clipboard.")
                        return
                    # Use the first image file
                    image_path = image_files[0]
                    img = Image.open(image_path)
                    if message_callback:
                        message_callback(f"Image loaded from clipboard file: {image_path}")
                elif isinstance(img, Image.Image):
                    if message_callback:
                        message_callback("Image grabbed from clipboard.")
                else:
                    if error_callback:
                        error_callback("Clipboard does not contain an image or image file.")
                    return
            except ImportError:
                if error_callback:
                    error_callback("PIL.ImageGrab is not available on this system.")
                return
            except Exception as e:
                if error_callback:
                    error_callback(f"Error accessing clipboard: {e}")
                return
        else:
            # Check if the file is a video (based on extension)
            if str(image_path).lower().endswith('.mp4'):
                if message_callback:
                    message_callback("Processing video...")
                process_and_save_video(image_path, resize_dim, process_mode, use_lab_flag, process_params, color_key_array, remove_bg, preprocess_flag, progress_callback, message_callback, error_callback)
                if message_callback:
                    message_callback("Processing complete!")
                return
            else:
                try:
                    img = Image.open(image_path)
                except FileNotFoundError:
                    if error_callback:
                        error_callback(f"File not found: {image_path}")
                    return
                except UnidentifiedImageError:
                    if error_callback:
                        error_callback(f"The file '{image_path}' is not a valid image.")
                    return

        # Check if image is animated (e.g., a GIF or WebP)
        is_multiframe = getattr(img, "is_animated", False)

        if is_multiframe:
            if message_callback:
                message_callback("Processing animated image...")
            # Save the image to a temporary path if it's from the clipboard
            if image_path == 'clip':
                temp_image_path = exe_path_fs('imagePawcessor/temp/clipboard_image.webp')
                img.save(temp_image_path, 'WEBP')
                image_path = temp_image_path
            process_and_save_gif(image_path, resize_dim, process_mode, use_lab_flag, process_params, color_key_array, remove_bg, preprocess_flag, progress_callback, message_callback, error_callback)
        else:
            if message_callback:
                message_callback("Processing image...")
            process_and_save_image(img, resize_dim, process_mode, use_lab_flag, process_params, color_key_array, remove_bg, preprocess_flag, progress_callback, message_callback, error_callback)

        if message_callback:
            message_callback("Processing complete!")

    except Exception as e:
        if error_callback:
            error_callback(str(e))



class WorkerSignals(QObject):
    progress = Signal(float)  # For progress percentage
    message = Signal(str)     # For status messages
    error = Signal(str)

class ClickableLabel(QLabel):
    """
    A QLabel that emits a signal when clicked and allows toggling clickability.
    Only emits the signal on mouse release if the label was not dragged.
    """
    clicked = Signal()

    def __init__(self, parent=None, is_clickable=True):
        super().__init__(parent)
        self.is_clickable = is_clickable  # Initialize with default clickability
        self._drag_threshold = 3  # Reduced pixel threshold for detecting drag
        self._mouse_start_pos = None  # To track the starting position of the mouse press
        self._dragged = False  # Track if the label has been dragged

    def mousePressEvent(self, event):
        if self.is_clickable and event.button() == Qt.LeftButton:
            self._mouse_start_pos = event.position().toPoint()  # Use position() and convert to QPoint
            self._dragged = False  # Reset dragged state on mouse press
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_clickable and self._mouse_start_pos is not None:
            # Calculate the distance moved since the mouse press
            distance = (event.position().toPoint() - self._mouse_start_pos).manhattanLength()
            if distance > self._drag_threshold:
                self._dragged = True  # Mark as dragged if distance exceeds threshold
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_clickable and event.button() == Qt.LeftButton:
            # Only emit clicked if not dragged and threshold not exceeded
            if not self._dragged:
                self.clicked.emit()
        super().mouseReleaseEvent(event)




class CanvasWorker(QObject):
    """
    Worker class to handle JSON updates, monitoring, and image generation in a separate thread.
    """
    show_message = Signal(str, bool)  # (message, is_error)
    images_finished = Signal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.COLOR_MAP = {
            0: 'ffe7c5',
            1: '2a3844',
            2: 'd70b5d',
            3: '0db39e',
            4: 'f4c009',
            5: 'ff00ff',
            6: 'bac357',
            7: 'a3b2d2',
            8: 'd6cec2',
            9: 'bfded8',
            10: 'a9c484',
            11: '5d937b',
            12: 'a2a6a9',
            13: '777f8f',
            14: 'eab281',
            15: 'ea7286',
            16: 'f4a4bf',
            17: 'a07ca7',
            18: 'bf796d',
            19: 'f5d1b6',
            20: 'e3e19f',
            21: 'ffdf00',
            22: 'ffbf00',
            23: 'c4b454',
            24: 'f5deb3',
            25: 'f4c430',
            26: '00ffff',
            27: '89cff0',
            28: '4d4dff',
            29: '00008b',
            30: '4169e1',
            31: '006742',
            32: '4cbb17',
            33: '2e6f40',
            34: '2e8b57',
            35: 'c0c0c0',
            36: '818589',
            37: '899499',
            38: '708090',
            39: 'ffa500',
            40: 'ff8c00',
            41: 'd7942d',
            42: 'ff5f1f',
            43: 'cc7722',
            44: 'ff69b4',
            45: 'ff10f0',
            46: 'aa336a',
            47: 'f4b4c4',
            48: '953553',
            49: 'd8bfd8',
            50: '7f00ff',
            51: '800080',
            52: 'ff2400',
            53: 'ff4433',
            54: 'a52a2a',
            55: '913831',
            56: 'ff0000',
            57: '3b2219',
            58: 'a16e4b',
            59: 'd4aa78',
            60: 'e6bc98',
            61: 'ffe7d1'
        }


    @Slot(str, str)  # Receives config_path and json_path as strings
    def process_canvas(self, config_path, json_path):
        print("process_canvas called")

        try:
            # Step 1: Update JSON request
            try:
                with open(config_path, "r+") as file:
                    data = json.load(file)
                    data["walky_talky_webfish"] = "get the canvas data bozo"
                    data["walky_talky_menu"] = "nothing new!"
                    file.seek(0)
                    json.dump(data, file, indent=4)
                    file.truncate()
                print("JSON updated successfully.")
                self.show_message.emit("Canvas request sent!", False)
            except Exception as e:
                print(f"Failed to update JSON: {e}")
                print("Failed to update JSON")
                create_default_config()
                return
            # Step 2: Monitor JSON status
            timeout = time.time() + 7  # 7-second timeout
            success = False
            previous_menu_value = "nothing new!"

            while time.time() < timeout:
                time.sleep(0.5)  # Poll every 0.5 seconds
                try:
                    with open(config_path, "r") as file:
                        data = json.load(file)

                        # Track changes in "walky_talky_menu"
                        current_menu_value = data.get("walky_talky_menu", "nothing new!")
                        if current_menu_value != previous_menu_value:
                            previous_menu_value = current_menu_value

                        # Check if "walky_talky_webfish" is reset
                        if data.get("walky_talky_webfish") == "nothing new!":
                            menu_value = current_menu_value
                            if menu_value != "nothing new!":
                                if menu_value == "Canvas data exported!":
                                    success = True
                                    break
                                else:
                                    self.show_message.emit(menu_value, True)
                                    self.images_finished.emit(False)
                                    return

                except json.JSONDecodeError as json_error:
                    print(f"JSON parsing error: {json_error}")
                    create_default_config()
                except FileNotFoundError as file_error:
                    print(f"File not found: {file_error}")
                    create_default_config()
                except Exception as e:
                    print(f"Unexpected error reading JSON: {e}")
                    create_default_config()

            # Handle success or failure after the loop
            if not success:
                self.show_message.emit("Game is probably not open.", True)
                self.images_finished.emit(False)
                return

            # Step 3: Generate Images
            print("Starting image generation...")
            self.generate_images_from_json(Path(json_path))

            # Notify the UI that image generation is complete
            self.images_finished.emit(True)
            print("Image generation complete.")

        except Exception as e:
            print(f"Exception in process_canvas: {e}")

    def generate_images_from_json(self, json_path: Path):
        """
        Processes exported canvas data JSON and generates PNG images.
        """
        output_directory = exe_path_fs("game_data/game_canvises")
        output_directory.mkdir(parents=True, exist_ok=True)

        try:
            with open(json_path, "r") as file:
                canvas_data = json.load(file)

            def process_canvas(canvas_name: str, points: list):
                img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
                pixels = img.load()
                for i in range(0, len(points), 3):
                    try:
                        x, y, color_idx = points[i:i + 3]
                        if not (0 <= x < 200 and 0 <= y < 200):
                            continue
                        hex_color = self.COLOR_MAP.get(color_idx, "000000")
                        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                        pixels[x, y] = (r, g, b, 255)
                    except Exception as e:
                        print(f"Error processing canvas '{canvas_name}': {e}")

                rotated_img = img.transpose(Image.ROTATE_270)

                output_path = output_directory / f"{canvas_name.replace(' ', '_').lower()}.png"
                rotated_img.save(output_path)
                print(f"Saved image: {output_path}")


            print("Processing canvas data...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                for canvas_name, points in canvas_data.items():
                    executor.submit(process_canvas, canvas_name, points)

            print("Image generation complete.")

        except Exception as e:
            print(f"Error generating images: {e}")


class HoverButton(QPushButton):
    def __init__(self, default_icon_path, hover_icon_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_icon = QIcon(default_icon_path)
        self.hover_icon = QIcon(hover_icon_path)
        self.setIcon(self.default_icon)
        self.setStyleSheet("border: none; background: transparent;")
        self.setFixedSize(72, 72)

    def enterEvent(self, event):
        self.setIcon(self.hover_icon)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(self.default_icon)
        super().leaveEvent(event)

class CropLabel(QLabel):
    crop_started = Signal(QPoint)
    crop_updated = Signal(QPoint)
    crop_finished = Signal(QPoint)
    erase_pixel = Signal(QPoint)  # New signal for erasing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMouseTracking(True)
        self.crop_rect = QRect()
        self.is_drawing = False

    def mousePressEvent(self, event):
        if self.main_window.crop_mode and event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.crop_rect.setTopLeft(event.position().toPoint())
            self.crop_rect.setBottomRight(event.position().toPoint())
            self.main_window.push_undo_stack()
            self.crop_started.emit(event.position().toPoint())
            self.update()
#        elif self.main_window.erase_mode and event.button() == Qt.LeftButton:
#            self.erase_pixel.emit(event.position().toPoint())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.main_window.crop_mode and self.is_drawing:
            self.crop_rect.setBottomRight(event.position().toPoint())
            self.crop_updated.emit(event.position().toPoint())
            self.update()
        """
        elif self.main_window.erase_mode and event.buttons() & Qt.LeftButton:
            self.erase_pixel.emit(event.position().toPoint())
            self.update()  # Trigger repaint for eraser outline
        else:
            super().mouseMoveEvent(event)
        """
    def mouseReleaseEvent(self, event):
        if self.main_window.crop_mode and event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.crop_rect.setBottomRight(event.position().toPoint())
            self.crop_finished.emit(event.position().toPoint())
            self.update()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.main_window.crop_mode and not self.crop_rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.crop_rect.normalized())
        """
        elif self.main_window.erase_mode:
            # Draw the eraser outline
            cursor_pos = QCursor.pos()
            widget_pos = self.mapFromGlobal(cursor_pos)
            # Ensure the cursor is within the label
            if self.rect().contains(widget_pos):
                painter = QPainter(self)
                pen = QPen(Qt.green, 1, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                
                # Calculate the display eraser size
                # Average the scaling factors for uniform scaling
                avg_scale = (self.main_window.scale_x + self.main_window.scale_y) / 2
                eraser_size_display = self.main_window.eraser_size / avg_scale
                
                painter.drawEllipse(widget_pos, eraser_size_display, eraser_size_display)
        """

class ImageProcessingThread(threading.Thread):
    def __init__(self, params, signals):
        super().__init__()
        self.params = params
        self.signals = signals
        self.executor = ThreadPoolExecutor(max_workers=6)  # Adjust based on your system's capability

    def run(self):
        try:
            # Submit the main function to the thread pool
            future = self.executor.submit(
                main,
                image_path=self.params['image_path'],
                remove_bg=self.params['remove_bg'],
                preprocess_flag=self.params['preprocess_flag'],
                use_lab_flag=self.params['use_lab'],
                brightness_flag=self.params['brightness'],
                resize_dim=self.params['resize_dim'],
                color_key_array=self.params['color_key_array'],
                process_mode=self.params['process_mode'],
                process_params=self.params['process_params'],
                progress_callback=self.signals.progress.emit,
                message_callback=self.signals.message.emit,
                error_callback=self.signals.error.emit
            )

            # Wait for the task to complete and capture exceptions if any
            result = future.result()
            self.signals.message.emit("Processing finished")
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.executor.shutdown()


class MainWindow(QMainWindow):
    start_process_canvas = Signal(str, str)
    bringfront = Signal()
    def __init__(self):
        super().__init__()

        self.bringfront.connect(self.bring_to_front)

        self.back_button = None
        self._is_dragging = False
        self._drag_position = QPoint()
        self.delete_mode = False
        self.last_message_displayed = None
        self.connected = False
        self.window_titles = [
            "are you kidding me?",
            "purple chalk???",
            "video game lover",
            "Color?? i hardly know 'er",
            "Pupple Puppyy wuz here",
            "u,mm, Haiiii X3",
            "animal",
            "aaaaand its all over my screen",
            "if ive gone missin ive gon fishn!",
            "the world is SPINNING, SPINNING!",
            "Now with ai!",
            "Actually i removed the ai now with no ai",
            "Full of spagetti",
            "i ated purple chalk!",
            "made by ChatGBT in just 8 minutes",
            "Fuck my chungus life",
            "Whaaatt? you dont have qhd???",
            "hello everybody my name is welcome",
            "Hi im a computer beep boop",
            "whats a python???",
            "purplepuppy more like uhh stupidpuppy gotem",
            "Waka waka waka",
            "This is a bucket",
            "bork meooow",
            "\"shrivel\" -grandma ",
            "Gnarp gnap",
            "all roads lead deeper into the woods",
            "numba based optimizations by baltdev",
            "Would you like to sign my petition?",
            "Guns don't kill. I do"
        ]
        self.setWindowTitle(random.choice(self.window_titles))
        self.setFixedSize(700, 768)
        self.move_to_center()
        self.padding_x = 0
        self.padding_y = 0
        self._is_dragging = False
        self.crop_mode = False
        self.drag_enabled = True
        self.temp_canvas_path = ""
        self._drag_position = QPoint()
        self.original_image_path = ""
        self.undo_stack = []
        # Initialize variables
        self.processing = False
        self.image_path = None
        self.image = None
        self.current_temp_file = None
        self.is_gif = False
        self.manual_change = False
        self.canpaste = True
        self.current_image_pixmap = None
        self.parameter_widgets = {}
        self.new_color = None  # Single color 5
        self.autocolor = True
        self.default_color_key_array = [
            {'number': 0, 'hex': 'ffe7c5', 'boost': 1.2, 'threshold': 20},
            {'number': 1, 'hex': '2a3844', 'boost': 1.2, 'threshold': 20},
            {'number': 2, 'hex': 'd70b5d', 'boost': 1.2, 'threshold': 20},
            {'number': 3, 'hex': '0db39e', 'boost': 1.2, 'threshold': 20},
            {'number': 4, 'hex': 'f4c009', 'boost': 1.2, 'threshold': 20},
            {'number': 6, 'hex': 'bac357', 'boost': 1.2, 'threshold': 20},
        ]
        self.button_stylesheet = """
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7b1fa2, stop:1 #9c27b0);
                color: white;
                border-radius: 15px;  /* Rounded corners */
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #9c27b0, stop:1 #d81b60);
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #6a0080, stop:1 #880e4f);
            }
        """
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_action)
        # Setup UI
        self.setup_ui()
        self.init_worker()
        self.bring_to_front()


    def mousePressEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and not self.crop_mode
#            and not self.erase_mode
            # Optionally, restrict dragging to certain areas
        ):
            self._is_dragging = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            new_pos = event.globalPosition().toPoint() - self._drag_position
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def init_worker(self):
        # Initialize worker and thread
        self.worker_thread = QThread()
        self.worker = CanvasWorker()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals and slots
        self.start_process_canvas.connect(self.worker.process_canvas)
        self.worker.show_message.connect(self.show_floating_message)
        self.worker.images_finished.connect(self.on_images_finished)
        
        # Shortcut for Undo (Ctrl+Z)
        
    def move_to_center(self):
        """
        Moves the window to the center of the appropriate screen.
        """
        # Ensure the window is shown to have valid size
        self.show()

        # Determine the target screen
        current_screen = self.get_current_screen()

        if not current_screen:
            current_screen = QApplication.primaryScreen()

        # Get the available geometry of the target screen
        screen_geometry = current_screen.availableGeometry()

        # Get the size of the window
        window_geometry = self.frameGeometry()

        # Calculate the center point
        center_point = screen_geometry.center()

        # Move the window's center to the screen's center
        window_geometry.moveCenter(center_point)

        # Apply the new top-left position to the window
        self.move(window_geometry.topLeft())

    def get_current_screen(self):
        """
        Determines which screen the window is currently on.
        Returns the QScreen object or None if not found.
        """
        # Get the current center position of the window
        window_pos = self.frameGeometry().center()

        # Find the screen at the window's center position
        return QApplication.screenAt(window_pos)

    @Slot()
    def bring_to_front(self):
        """Brings the window to the front without disabling the close button."""
        # Ensure the window is visible
        self.show()
        self.activateWindow()
        self.raise_()

        # Get the current window flags
        current_flags = self.windowFlags()
        
        # Temporarily add WindowStaysOnTopHint without calling show()
        self.setWindowFlags(current_flags | Qt.WindowStaysOnTopHint)
        self.show()  # This enforces the "always on top" effect
        
        # Use QTimer to restore the original flags after a short delay
        QTimer.singleShot(100, lambda: self.setWindowFlags(current_flags) or self.show())


    def setup_ui(self):
        """
        Sets up the main UI with the stacked widget, menus, and persistent signature.
        """
        # Apply dark-themed stylesheet with purple accents
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1a33; /* Very dark purple */
                color: #ffffff;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
                font-size: 16px; /* Slightly larger */
            }
            QPushButton {
                background-color: #7b1fa2;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px; /* Slightly larger */
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
            }
            QPushButton:hover {
                background-color: #9c27b0;
            }
            QPushButton:disabled {
                background-color: #4a148c;
            }
            QLabel {
                font-size: 16px; /* Slightly larger */
                color: #ffffff;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
            }
            QCheckBox {
                font-size: 16px; /* Slightly larger */
                color: #ffffff;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #7b1fa2;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ba68c8;
                border: 1px solid #ffffff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox, QSpinBox, QLineEdit {
                font-size: 16px; /* Slightly larger */
                padding: 5px;
                border: 1px solid #7b1fa2;
                border-radius: 5px;
                background-color: #424242;
                color: #ffffff;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
            }
            QProgressBar {
                height: 15px;
                border: 1px solid #7b1fa2;
                border-radius: 7px;
                text-align: center;
                background-color: #424242;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
                font-size: 16px; /* Slightly larger */
            }
            QProgressBar::chunk {
                background-color: #ba68c8;
                width: 1px;
            }
            QGroupBox {
                border: 1px solid #7b1fa2;
                border-radius: 5px;
                margin-top: 10px;
                color: #ffffff;
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-weight: bold; /* Bold text */
                font-size: 16px; /* Slightly larger */
            }
        """)


        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout for the entire application
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Stacked widget for switching between menus
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Initial menu
        self.setup_initial_menu()



    def keyPressEvent(self, event):
        """
        Override keyPressEvent to detect Ctrl+V (Paste).
        """
        if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier) and self.canpaste:
            self.open_image_from_clipboard(True)
        else:
            super().keyPressEvent(event)

    def setup_initial_menu(self):
        """
        Sets up the initial menu with enhanced buttons and a random image 
        from the 'menu_pics' directory as a background image.
        Adds Save Current and Load buttons below the primary ones.
        """


        # Initialize the main widget
        initial_widget = QWidget()
        initial_layout = QVBoxLayout()
        initial_layout.setContentsMargins(0, 0, 0, 0)  # Remove all margins
        initial_layout.setSpacing(10)  # Minimal spacing between elements
        initial_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        initial_widget.setLayout(initial_layout)

        # -------------------------
        # Top Layout: Always on Top Checkbox
        # -------------------------
        top_layout = QHBoxLayout()
        top_layout.setSpacing(0)  # No spacing needed


        # Pin button container
        pin_container = QWidget()
        pin_layout = QHBoxLayout()
        pin_layout.setContentsMargins(0, 0, 0, 0)  # No margins for precise placement
        pin_layout.setAlignment(Qt.AlignRight | Qt.AlignTop)
        pin_container.setLayout(pin_layout)

        # Add the pin button
        self.always_on_top_checkbox = QCheckBox()
        self.always_on_top_checkbox.setFixedSize(80, 80) 
        self.always_on_top_checkbox.setStyleSheet(f"""
            QCheckBox {{
                background: transparent;
                border: none;
            }}
            QCheckBox::indicator {{
                width: 80px;
                height: 80px;
                image: url({exe_path_str("imagePawcessor/font_stuff/tack.svg")});
            }}
            QCheckBox::indicator:checked {{
                image: url({exe_path_str("imagePawcessor/font_stuff/tack_down.svg")});
            }}
            QCheckBox::indicator:hover {{
                image: url({exe_path_str("imagePawcessor/font_stuff/tack_hover.svg")});
            }}
        """)
        self.always_on_top_checkbox.setChecked(False)
        
        self.always_on_top_checkbox.toggled.connect(self.toggle_always_on_top)
        pin_layout.addWidget(self.always_on_top_checkbox)

        # Add the pin_container to the top_layout
        top_layout.addWidget(pin_container)

        # Add the top_layout to the initial_layout
        initial_layout.addLayout(top_layout)

        # -------------------------
        # Background Container: Image and Buttons
        # -------------------------
        background_container = QWidget()
        background_layout = QVBoxLayout()
        background_layout.setContentsMargins(0, 0, 0, 0)  # Minimal top margin
        background_layout.setSpacing(0)
        background_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)  # Center contents
        background_container.setLayout(background_layout)
        self.my_spacer = QWidget()
        # Create a ClickableLabel to hold the background image
        self.background_label = ClickableLabel()
        self.background_label.setFixedSize(680, 460)
        self.background_label.setPixmap(self.load_and_display_random_image())
        self.background_label.setAlignment(Qt.AlignCenter)

        self.background_label.setScaledContents(False)  # Prevent automatic scaling
        background_layout.addWidget(self.background_label, alignment=Qt.AlignCenter)
        spacer = QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        background_layout.addItem(spacer)

        # Optionally, connect the clicked signal
        # Example: self.background_label.clicked.connect(self.handle_background_click)

        # -------------------------
        # Button Container: Stamp Buttons and Control Buttons
        # -------------------------
        button_container = QWidget()
        button_layout = QVBoxLayout()
        button_layout.setSpacing(20)  # Space between button rows
        button_layout.setAlignment(Qt.AlignTop)  # Align buttons to the top
        button_container.setLayout(button_layout)

        # Button Stylesheet
        button_stylesheet = """
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7b1fa2, stop:1 #9c27b0);
                color: white;
                border-radius: 15px;  /* Rounded corners */
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #9c27b0, stop:1 #d81b60);
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #6a0080, stop:1 #880e4f);
            }
        """

        # First row of buttons: "Stamp from Files" and "Stamp from Clipboard"
        top_button_layout = QHBoxLayout()
        top_button_layout.setSpacing(20)
        top_button_layout.setAlignment(Qt.AlignCenter)

        self.new_image_files_button = QPushButton("Stamp from Files")
        self.new_image_files_button.setStyleSheet(button_stylesheet)
        self.new_image_files_button.setMinimumSize(200, 60)
        self.new_image_files_button.clicked.connect(self.open_image_from_files)
        top_button_layout.addWidget(self.new_image_files_button)

        self.new_image_clipboard_button = QPushButton("Save In-Game Art")
        self.new_image_clipboard_button.setStyleSheet(button_stylesheet)
        self.new_image_clipboard_button.setMinimumSize(200, 60)
        self.new_image_clipboard_button.clicked.connect(self.request_and_monitor_canvas)
        top_button_layout.addWidget(self.new_image_clipboard_button)

        button_layout.addLayout(top_button_layout)

        # Second row of buttons: "Save Menu" and "Exit"
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.setSpacing(20)
        bottom_button_layout.setAlignment(Qt.AlignCenter)

        self.save_button = QPushButton("Save Menu")
        self.save_button.setStyleSheet(button_stylesheet)
        self.save_button.setMinimumSize(160, 60)
        self.save_button.clicked.connect(self.show_save_menu)
        bottom_button_layout.addWidget(self.save_button)


        self.clip_button = QPushButton("Clipboard")
        self.clip_button.setStyleSheet(button_stylesheet)
        self.clip_button.setMinimumSize(160, 60)
        #self.clip_button.clicked.connect(self.request_and_monitor_canvas)
        bottom_button_layout.addWidget(self.clip_button)
        self.clip_button.clicked.connect(lambda: self.open_image_from_clipboard())


        self.exit_button = QPushButton("Keybinds / Info")
        self.exit_button.setStyleSheet(button_stylesheet)
        self.exit_button.setMinimumSize(200, 60)
        #self.exit_button.clicked.connect(self.request_and_monitor_canvas)
        bottom_button_layout.addWidget(self.exit_button)
        self.exit_button.clicked.connect(self.open_website)
        button_layout.addLayout(bottom_button_layout)
        

        # Add the button_container to the background_layout
        background_layout.addWidget(button_container, alignment=Qt.AlignCenter)


        # -------------------------
        # Add the background_container to the initial_layout
        # -------------------------
        initial_layout.addWidget(background_container, alignment=Qt.AlignCenter)

        self.signature_label = QLabel("By PurplePuppy & baltdev")

        self.signature_label.setStyleSheet("""
            color: #A45EE5;
            font-size: 16px;
            font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
            font-weight: bold;
        """)
        self.signature_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.signature_label.setFixedSize(200, 20)  # Adjust width and height as needed
        self.signature_label.setParent(self)

# Set initial position for the signature label
        padding_bottom = 20  # Adjust padding to avoid clipping
        oscillation_amount = 5  # Subtle bounce (reduce from 10 to 5)

        self.signature_label.move(10, self.height() - padding_bottom - self.signature_label.height())

        self.signature_label.move(10, self.height() - padding_bottom - self.signature_label.height())

        # Create a timer to toggle the label's position
        self.toggle_timer = QTimer(self)
        self.toggle_timer.timeout.connect(self.toggle_position)
        self.toggle_timer.start(542)  # Switch every 500ms
        
        initial_layout.addWidget(self.signature_label)

                # -------------------------
        # Add the initial widget to the stacked widget
        # -------------------------
        self.stacked_widget.addWidget(initial_widget)



    def toggle_position(self):
        """Toggle the position of the signature label."""
        current_pos = self.signature_label.pos()
        new_pos = QPoint(
            current_pos.x(),
            self.height() - 20 - self.signature_label.height() - (2 if current_pos.y() == self.height() - 20 - self.signature_label.height() else 0)
        )
        self.signature_label.move(new_pos)

    def open_website(self):
        webbrowser.open("https://github.com/unpaid-intern/StampMod/?tab=readme-ov-file#keybinds")


    def setup_save_menu1(self):
        """
        Sets up Save Menu 1 with a 2x2 grid of images, each with a title above.
        Includes a larger Home button with a hover SVG and no background.
        """
        # Create widget and layout for Save Menu 1
        self.save_menu1_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Home button
        home_button = HoverButton(
            exe_path_str("imagePawcessor/font_stuff/home.svg"),
            exe_path_str("imagePawcessor/font_stuff/home_hover.svg")
        )
        home_button.setFixedSize(72, 72)
        home_button.setIconSize(QSize(72, 72))
        home_button.clicked.connect(self.go_to_initial_menu)
        layout.addWidget(home_button, alignment=Qt.AlignLeft)
        # Grid layout for images
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Placeholder for images and their paths
        self.image_labels = []
        self.image_paths = []  # To be populated dynamically in update_save_menu1

        canvas_names = ["Canvas 1", "Canvas 2", "Canvas 3", "Canvas 4"]
        for idx, name in enumerate(canvas_names):
            # Create a vertical container for each image and its title
            container = QVBoxLayout()
            container.setSpacing(5)
            container.setAlignment(Qt.AlignCenter)

            # Name label (title above the image)
            name_label = QLabel(name)
            name_label.setStyleSheet("""
                font-size: 32px;
                color: white;
                text-align: center;
                font-weight: bold;
            """)
            name_label.setAlignment(Qt.AlignCenter)
            container.addWidget(name_label)

            # Outer frame surrounding the image
            frame = QFrame()
            frame.setFixedSize(262, 262)  # Slightly larger than the image to accommodate the border
            frame.setStyleSheet("""
                QFrame {
                    border: 3px solid #FFFFFF; /* Thicker border */
                    background: transparent;
                }
            """)

            # Image button inside the frame
            image_label = QPushButton(frame)
            image_label.setFixedSize(256, 256)  # Exact size of the image
            image_label.setStyleSheet("""
                QPushButton {
                    border: none; /* No border for the image itself */
                    background: transparent;
                }
            """)
            frame_layout = QVBoxLayout(frame)  # Add layout to center the image in the frame
            frame_layout.setContentsMargins(0, 0, 0, 0)  # No padding in the frame
            frame_layout.addWidget(image_label, alignment=Qt.AlignCenter)
            
            container.addWidget(frame, alignment=Qt.AlignCenter)  # Add the frame to the container

            self.image_labels.append(image_label)  # Keep references for updates

            # Add the container to the grid layout
            grid_layout.addLayout(container, idx // 2, idx % 2)  # Row and column positioning

        layout.addLayout(grid_layout)

        # Set layout and add to stacked widget
        self.save_menu1_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.save_menu1_widget)



    def update_save_menu1(self, directory_path):
        """
        Updates the Save Menu 1 with images from the specified directory.
        Initializes the menu if it doesn't already exist and switches to it.
        """
        # Initialize menu if it doesn't exist
        if not hasattr(self, 'save_menu1_widget'):
            self.setup_save_menu1()
        self.undo_stack.clear()
        # Get PNG images from the directory
        image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
        image_files = sorted(image_files)[:4]  # Limit to the first 4 images

        if len(image_files) < 4:
            self.show_floating_message("Not enough images in the directory", True)
            return

        # Update images and assign click functionality
        for idx, (image_label, image_file) in enumerate(zip(self.image_labels, image_files)):
            image_path = os.path.join(directory_path, image_file)
            self.image_paths.append(image_path)

            # Load and scale image to 256x256 using NEAREST neighbor
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.FastTransformation)
            image_label.setIcon(QIcon(scaled_pixmap))
            image_label.setIconSize(QSize(256, 256))

            try:
                image_label.clicked.disconnect()  # Disconnect previous connections, if any
            except:
                pass
            image_label.clicked.connect(lambda _, path=image_path: self.go_to_second_menu(path))

        # Switch to Save Menu 1
        self.stacked_widget.setCurrentWidget(self.save_menu1_widget)



    def go_to_second_menu(self, path):
        """
        Navigates to Menu 2, displaying the image from the provided path.
        """
        # Check if the file exists
        if not os.path.exists(path):
            QMessageBox.critical(self, "Error", f"The file at {path} does not exist.")
            return

        # Load the image in Menu 2
        if not hasattr(self, 'setup_menu2'):
            QMessageBox.critical(self, "Error", "Menu 2 is not yet implemented.")
            return

        # Set up Menu 2 with the image path
        self.setup_menu2(path)

        # Switch to Menu 2
        self.stacked_widget.setCurrentWidget(self.menu2_widget)

    def setup_menu2(self, image_path):
        """
        Sets up Menu 2 for displaying and editing the selected image.
        Includes rotate, erase, crop functionality, and save/reset buttons.
        """
        # Create widget and layout for Menu 2
        self.temp_canvas_path = exe_path_fs("imagePawcessor/temp/temp_canvas.png")
        self.original_image_path = image_path
        shutil.copy(image_path, self.temp_canvas_path)

        self.menu2_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)

        # Home button using HoverButton
        home_button = HoverButton(
            exe_path_str("imagePawcessor/font_stuff/home.svg"),
            exe_path_str("imagePawcessor/font_stuff/home_hover.svg")
        )
        home_button.setFixedSize(80, 80)
        home_button.setIconSize(QSize(80, 80))
        home_button.clicked.connect(self.go_to_initial_menu)
        layout.addWidget(home_button, alignment=Qt.AlignLeft)

        # Horizontal layout for the image and tools
        image_tools_layout = QHBoxLayout()

        # Display the selected image
        self.image_label_canvas = CropLabel(self)
        self.image_label_canvas.setFixedSize(512, 512)
        self.image_label_canvas.setStyleSheet("border: 1px solid #7b1fa2; background: #000000;")
        with Image.open(self.temp_canvas_path) as img:
            original_size = img.size
        self.scale_x = original_size[0] / self.image_label_canvas.width()
        self.scale_y = original_size[1] / self.image_label_canvas.height()

        pixmap = QPixmap(self.temp_canvas_path).scaled(
            self.calculate_display_size(self.temp_canvas_path),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.image_label_canvas.setPixmap(pixmap)
        self.image_label_canvas.setAlignment(Qt.AlignCenter)  # Center alignment

        # Store scaling factors in the CropLabel
        self.image_label_canvas.scale_x = self.scale_x
        self.image_label_canvas.scale_y = self.scale_y
        image_tools_layout.addWidget(self.image_label_canvas, alignment=Qt.AlignLeft)

        # Connect CropLabel signals
        self.image_label_canvas.crop_started.connect(self.on_crop_started)
        self.image_label_canvas.crop_updated.connect(self.on_crop_updated)
        self.image_label_canvas.crop_finished.connect(self.on_crop_finished)
        self.image_label_canvas.erase_pixel.connect(self.handle_erase)

        # Column of tool buttons aligned to the right of the image
        tools_layout = QVBoxLayout()
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(10)  # Increased spacing for better UI

        # Add a spacer at the top
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Undo Button
        undo_button = HoverButton(
            exe_path_str("imagePawcessor/font_stuff/undo.svg"),
            exe_path_str("imagePawcessor/font_stuff/undo_hover.svg")
        )
        undo_button.setFixedSize(80,80)
        undo_button.setIconSize(QSize(80,80))
        undo_button.clicked.connect(self.undo_action)
        tools_layout.addWidget(undo_button, alignment=Qt.AlignCenter)

        # Add a spacer at the bottom
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Rotate Left Button
        rotate_left_button = HoverButton(
            exe_path_str("imagePawcessor/font_stuff/rotate_left.svg"),
            exe_path_str("imagePawcessor/font_stuff/rotate_left_hover.svg")
        )
        rotate_left_button.setFixedSize(80,80)
        rotate_left_button.setIconSize(QSize(80,80))
        rotate_left_button.clicked.connect(lambda: self.rotate_image(90))
        tools_layout.addWidget(rotate_left_button, alignment=Qt.AlignCenter)

        # Spacer between Rotate Left and Rotate Right
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Rotate Right Button
        rotate_right_button = HoverButton(
            exe_path_str("imagePawcessor/font_stuff/rotate_right.svg"),
            exe_path_str("imagePawcessor/font_stuff/rotate_right_hover.svg")
        )
        rotate_right_button.setFixedSize(80,80)
        rotate_right_button.setIconSize(QSize(80,80))
        rotate_right_button.clicked.connect(lambda: self.rotate_image(-90))
        tools_layout.addWidget(rotate_right_button, alignment=Qt.AlignCenter)

        # Spacer between Rotate Right and Crop Checkbox
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Crop Mode Checkbox with SVGs
        self.crop_checkbox = QCheckBox()
        self.crop_checkbox.setFixedSize(80,80)
        self.crop_checkbox.setStyleSheet(f"""
            QCheckBox {{
                background: transparent;
                border: none;
            }}
            QCheckBox::indicator {{
                width: 80px;
                height: 80px;
                image: url({exe_path_str("imagePawcessor/font_stuff/crop.svg")});
            }}
            QCheckBox::indicator:checked {{
                image: url({exe_path_str("imagePawcessor/font_stuff/crop_on.svg")});
            }}
            QCheckBox::indicator:hover {{
                image: url({exe_path_str("imagePawcessor/font_stuff/crop_hover.svg")});
            }}
        """)
        self.crop_checkbox.stateChanged.connect(self.toggle_crop_mode)
        tools_layout.addWidget(self.crop_checkbox, alignment=Qt.AlignCenter)

        # Spacer between Crop Checkbox and Erase Button
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        """
        self.erase_checkbox = QCheckBox()
        self.erase_checkbox.setFixedSize(60, 60)
        self.erase_checkbox.setStyleSheet(f"""
#            QCheckBox {{
#               background: transparent;
#               border: none;
#            }}
#            QCheckBox::indicator {{
#                width: 60px;
#                height: 60px;
#                image: url({exe_path_str("imagePawcessor/font_stuff/erase.svg")});
#            }}
#            QCheckBox::indicator:checked {{
#                image: url({exe_path_str("imagePawcessor/font_stuff/erase_on.svg")});
#            }}
#            QCheckBox::indicator:hover {{
#                image: url({exe_path_str("imagePawcessor/font_stuff/erase_hover.svg")});
#            }}
        """)
        self.erase_checkbox.stateChanged.connect(self.toggle_erase_mode)
        tools_layout.addWidget(self.erase_checkbox, alignment=Qt.AlignCenter)
        # Spacer between Erase Button and Undo Button
        tools_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        """
        # Add the tools layout to the image-tools layout
        image_tools_layout.addLayout(tools_layout)

        # Add the image-tools layout to the main layout
        layout.addLayout(image_tools_layout)

        """
        eraser_controls_layout = QHBoxLayout()
        # Eraser Size Label
        eraser_size_label = QLabel("Eraser size in pixels:")
        eraser_size_label.setStyleSheet("color: white; font-size: 14px;")
        eraser_controls_layout.addWidget(eraser_size_label, alignment=Qt.AlignVCenter)

        # Eraser Size Slider
        self.eraser_size_slider = QSlider(Qt.Horizontal)
        self.eraser_size_slider.setMinimum(2)
        self.eraser_size_slider.setMaximum(30)
        self.eraser_size_slider.setValue(6)
        self.eraser_size_slider.setTickPosition(QSlider.TicksBelow)
        self.eraser_size_slider.setTickInterval(2)
        self.eraser_size_slider.valueChanged.connect(self.update_eraser_size)
        self.eraser_size_slider.setFixedWidth(150)  # Adjust width as needed
        eraser_controls_layout.addWidget(self.eraser_size_slider, alignment=Qt.AlignVCenter)

        # Add the eraser controls layout to the main layout (beneath image and tools)
        layout.addLayout(eraser_controls_layout)
        """

        # Save and Reset Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(20)

        save_button = QPushButton("Save!")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #7b1fa2;
                color: white;
                font-size: 50px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #9c27b0;
            }
        """)
        save_button.clicked.connect(lambda: self.process_png_to_stamp(self.temp_canvas_path))
        buttons_layout.addWidget(save_button)

        reset_button = QPushButton("Reset")
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ba000d;
                color: white;
                font-size: 50px;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        reset_button.clicked.connect(lambda: self.reset_image())
        buttons_layout.addWidget(reset_button)
        # Add buttons layout to main layout
        layout.addLayout(buttons_layout)

        # Set layout and add to stacked widget
        self.menu2_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.menu2_widget)
        # After connecting crop signals
        
        self.image_label_canvas.erase_pixel.connect(self.handle_erase)

    def update_eraser_size(self, value):
        """
        Updates the eraser size based on the slider value.
        """
        self.eraser_size = value
        print(f"Eraser size set to: {self.eraser_size}")  # Debug statement
        self.image_label_canvas.update()  # Trigger repaint to update the preview


    def toggle_erase_mode(self, state):
        """
        Toggle erase mode on or off based on the checkbox state.
        """
        self.update_scaling_factors()
        if state:
            self.erase_mode = True
            # Deactivate Crop Mode if active
            if self.crop_checkbox.isChecked():
                self.crop_checkbox.setChecked(False)
            self.drag_enabled = False  # Disable window dragging during erase
            self.image_label_canvas.setCursor(Qt.PointingHandCursor)
            self.push_undo_stack()  # Save state for undo (only once)
            print("Erase mode enabled")  # Debug statement
        else:
            self.erase_mode = False
            self.drag_enabled = True  # Re-enable window dragging
            self.image_label_canvas.setCursor(Qt.ArrowCursor)
            self.image_label_canvas.current_mouse_pos = QPoint()  # Reset mouse position
            self.image_label_canvas.update()  # Remove eraser outline
            print("Erase mode disabled")  # Debug statement



    def calculate_display_size(self, image_path):
        """
        Calculate the display size to scale the image so that the largest dimension is 512 pixels.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > height:
                    return QSize(512, int((height / width) * 512))
                else:
                    return QSize(int((width / height) * 512), 512)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {e}")
            return QSize(512, 512)

    def toggle_crop_mode(self, state):
        """
        Toggle crop mode on or off based on the checkbox state.
        """
        if state:
            self.crop_mode = True
            # Deactivate Erase Mode if active
#            if self.erase_checkbox.isChecked():
#                self.erase_checkbox.setChecked(False)
            self.drag_enabled = False  # Disable window dragging during crop
            self.image_label_canvas.setCursor(Qt.CrossCursor)
            self.push_undo_stack()  # Save state for undo
            print("Crop mode enabled")  # Debug statement
        else:
            self.crop_mode = False
            self.drag_enabled = True  # Re-enable window dragging
            self.image_label_canvas.setCursor(Qt.ArrowCursor)
            print("Crop mode disabled")  # Debug statement


    def on_crop_started(self, point):
        """
        Handle the start of cropping.
        """
        self.crop_start = point

    def on_crop_updated(self, point):
        """
        Handle the update of cropping.
        """
        self.crop_end = point
        self.image_label_canvas.crop_rect = QRect(self.crop_start, self.crop_end)
        self.image_label_canvas.update()

    def on_crop_finished(self, point):
        """
        Handle the completion of cropping.
        """
        self.crop_end = point
        self.image_label_canvas.crop_rect = QRect(self.crop_start, self.crop_end)
        self.image_label_canvas.update()
        self.perform_crop()
        self.crop_mode = False
        self.crop_checkbox.setChecked(False)
        self.drag_enabled = True
        self.image_label_canvas.setCursor(Qt.ArrowCursor)
        
        # Clear the crop rectangle to remove artifacts
        self.image_label_canvas.crop_rect = QRect()
        self.image_label_canvas.update()
     
    def perform_crop(self):
        """
        Crops the image based on the drawn rectangle and updates the display.
        """
        if not self.crop_start or not self.crop_end:
            QMessageBox.warning(self, "Crop Error", "Invalid crop area.")
            return

        # Calculate the crop rectangle in the QLabel's coordinate system
        x1, y1 = self.crop_start.x(), self.crop_start.y()
        x2, y2 = self.crop_end.x(), self.crop_end.y()
        crop_rect = QRect(QPoint(min(x1, x2), min(y1, y2)), QPoint(max(x1, x2), max(y1, y2)))

        # Map the crop rectangle to the original image's coordinate system
        label_size = self.image_label_canvas.size()
        try:
            with Image.open(self.temp_canvas_path) as img:
                original_size = img.size  # Original image size (e.g., 200x200)
                scale_x = original_size[0] / label_size.width()
                scale_y = original_size[1] / label_size.height()

                # Calculate the corresponding crop rectangle on the original image
                original_crop_rect = QRect(
                    int(crop_rect.x() * scale_x),
                    int(crop_rect.y() * scale_y),
                    int(crop_rect.width() * scale_x),
                    int(crop_rect.height() * scale_y)
                )

                # Ensure the crop rectangle is within the image bounds
                original_crop_rect = original_crop_rect.intersected(QRect(0, 0, original_size[0], original_size[1]))

                if original_crop_rect.width() == 0 or original_crop_rect.height() == 0:
                    QMessageBox.warning(self, "Crop Error", "Crop area is too small.")
                    return

                # Crop the image
                cropped_img = img.crop((
                    original_crop_rect.x(),
                    original_crop_rect.y(),
                    original_crop_rect.x() + original_crop_rect.width(),
                    original_crop_rect.y() + original_crop_rect.height()
                ))

                # Remove fully transparent pixels
                bbox = cropped_img.getbbox()
                if bbox:
                    cropped_img = cropped_img.crop(bbox)

                # Save the cropped image back to temp_canvas_path without resizing
                cropped_img.save(self.temp_canvas_path)

                # Update the display
                self.update_display(self.temp_canvas_path)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"The file at {self.temp_canvas_path} does not exist.")
        except UnidentifiedImageError:
            QMessageBox.critical(self, "Error", "Cannot identify the image file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during cropping: {e}")


    def update_display(self, image_path):
        """
        Updates the QLabel to display the image at image_path, scaling the largest dimension to 512px.
        """
        pixmap = QPixmap(image_path)
        display_size = self.calculate_display_size(image_path)
        scaled_pixmap = pixmap.scaled(
            display_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.image_label_canvas.setPixmap(scaled_pixmap)
        self.image_label_canvas.setAlignment(Qt.AlignCenter)  # Center the pixmap
        self.image_label_canvas.update()

    def update_scaling_factors(self):
        """
        Recalculates and updates the scaling factors and padding offsets based on the current image size.
        """
        label_size = self.image_label_canvas.size()
        try:
            with Image.open(self.temp_canvas_path) as img:
                original_size = img.size
            self.scale_x = original_size[0] / float(label_size.width())
            self.scale_y = original_size[1] / float(label_size.height())

            # Calculate scaling factors in CropLabel
            self.image_label_canvas.scale_x = self.scale_x
            self.image_label_canvas.scale_y = self.scale_y

            # Calculate padding offsets based on aspect ratio
            pixmap = QPixmap(self.temp_canvas_path).scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.image_label_canvas.setPixmap(pixmap)
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()

            padding_x = (label_size.width() - pixmap_width) / 2.0
            padding_y = (label_size.height() - pixmap_height) / 2.0

            self.image_label_canvas.padding_x = padding_x
            self.image_label_canvas.padding_y = padding_y

            self.image_label_canvas.update()
            print(f"Scaling factors updated: scale_x={self.scale_x}, scale_y={self.scale_y}")
            print(f"Padding updated: padding_x={self.padding_x}, padding_y={self.padding_y}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update scaling factors: {e}")


    def rotate_image(self, angle):
        """
        Rotates the image by the specified angle and updates the display while preserving aspect ratio.
        """
        try:
            self.push_undo_stack()  # Save state before rotating
            with Image.open(self.temp_canvas_path) as img:
                rotated_img = img.rotate(angle, expand=True)
                rotated_img.save(self.temp_canvas_path)
            self.update_display(self.temp_canvas_path)
            self.update_scaling_factors()  # Update scaling after rotation
            print(f"Image rotated by {angle} degrees.")  # Debug statement
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rotate image: {e}")

    def handle_erase(self, point):
        """
        Erases pixels in a circular area around the given point.
        """
        try:
            with Image.open(self.temp_canvas_path) as img:
                img = img.convert("RGBA")
                original_size = img.size

                # Retrieve scaling factors and padding from image_label_canvas
                scale_x = self.image_label_canvas.scale_x
                scale_y = self.image_label_canvas.scale_y
                padding_x = self.image_label_canvas.padding_x
                padding_y = self.image_label_canvas.padding_y

                print(f"handle_erase called with point: {point}")
                print(f"scale_x: {scale_x}, scale_y: {scale_y}")
                print(f"padding_x: {padding_x}, padding_y: {padding_y}")

                # Adjust the cursor position by removing padding
                adjusted_x = point.x() - padding_x
                adjusted_y = point.y() - padding_y

                print(f"Adjusted coordinates: ({adjusted_x}, {adjusted_y})")

                # Ensure the adjusted coordinates are within the image_label_canvas area
                if (adjusted_x < 0 or adjusted_y < 0 or
                    adjusted_x > self.image_label_canvas.pixmap().width() or
                    adjusted_y > self.image_label_canvas.pixmap().height()):
                    # Cursor is outside the image area; do not erase
                    print("Cursor is outside the image area; skipping erase.")
                    return

                # Map the point to the original image coordinates with precise scaling
                original_x = int(round(adjusted_x * scale_x))
                original_y = int(round(adjusted_y * scale_y))

                print(f"Original image coordinates: ({original_x}, {original_y})")

                # Define the eraser radius (scaled appropriately)
                avg_scale = (scale_x + scale_y) / 2
                radius = int(round(self.eraser_size * avg_scale))

                print(f"Eraser radius: {radius}")

                # Use ImageDraw for efficient erasing
                draw = ImageDraw.Draw(img)
                # Draw a fully transparent circle
                draw.ellipse(
                    (original_x - radius, original_y - radius,
                    original_x + radius, original_y + radius),
                    fill=(0, 0, 0, 0)  # Fully transparent
                )

                img.save(self.temp_canvas_path)

        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"The file at {self.temp_canvas_path} does not exist.")
        except UnidentifiedImageError:
            QMessageBox.critical(self, "Error", "Cannot identify the image file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during erasing: {e}")

        # Update the display
        self.update_display(self.temp_canvas_path)



    def save_image(self):
        """
        Saves the modified image to a user-specified location.
        """
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if save_path:
            try:
                shutil.copy(self.temp_canvas_path, save_path)
                QMessageBox.information(self, "Saved", f"Image saved to {save_path}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def reset_image(self):
        """
        Resets temp_canvas.png to the original image and updates the display.
        """
        try:
            shutil.copy(self.original_image_path, self.temp_canvas_path)
            self.update_display(self.temp_canvas_path)
            self.show_floating_message("Reset!")
            self.undo_stack.clear()  # Clear undo history on reset
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset image: {e}")

    def push_undo_stack(self):
        """
        Pushes the current state of the image onto the undo stack.
        """
        try:
            with Image.open(self.temp_canvas_path) as img:
                self.undo_stack.append(img.copy())
                # Limit the undo stack size if necessary
                if len(self.undo_stack) > 20:
                    self.undo_stack.pop(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to push to undo stack: {e}")

    def undo_action(self):
        """
        Undoes the last action.
        """
        if not self.undo_stack:
            self.show_floating_message("Nothing to undo.")
            return
        try:
            last_img = self.undo_stack.pop()
            last_img.save(self.temp_canvas_path)
            self.update_display(self.temp_canvas_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to undo action: {e}")




    def request_and_monitor_canvas(self):
        """
        Initiates the canvas request and monitoring process.
        """
        if self.processing:
            self.show_floating_message("Request already sent")
            return
        
        self.worker_thread.start()
        config_path = get_config_path()  # Define this function appropriately
        json_path = exe_path_fs("game_data/game_canvises/game_canvises.json")  # Define this function

        if not os.path.exists(config_path):
            self.show_floating_message("Config path does not exist.", True)
            return

        if not os.path.exists(json_path):
            self.show_floating_message("JSON path does not exist.", True)
            return

        # Set the processing flag
        self.processing = True

        # Emit signal to start processing in the worker thread
        self.start_process_canvas.emit(str(config_path), str(json_path))
        print("Emitted start_process_canvas signal.")



    @Slot(bool)
    def on_images_finished(self, success):
        """
        Callback when image generation is complete.
        """
        self.worker_thread.quit() 
        self.worker_thread.wait() 
        self.processing = False
        if success:
            print("Image generation completed successfully!")
            self.update_save_menu1(exe_path_fs("game_data/game_canvises"))


    def process_png_to_stamp(self, input_png_path):
        """
        Processes a PNG image and saves its data in a text file with color mappings.
        Also manages the preview directory by clearing it and copying the input PNG.

        Parameters:
            input_png_path (str): The file path to the input PNG image.
        """
        # Helper function to convert hex color to RGB tuple
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) != 6:
                raise ValueError(f"Invalid hex color: {hex_color}")
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Helper function to find the closest color number from the color key
        def find_closest_color(target_rgb, color_key_rgb):
            min_distance = float('inf')
            closest_color_num = None
            for idx, color in enumerate(color_key_rgb):
                distance = math.sqrt(
                    (target_rgb[0] - color[0]) ** 2 +
                    (target_rgb[1] - color[1]) ** 2 +
                    (target_rgb[2] - color[2]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_color_num = idx
            return closest_color_num

        try:
            # Define color key as per the provided mapping
            color_key = {
                0: 'ffe7c5',
                1: '2a3844',
                2: 'd70b5d',
                3: '0db39e',
                4: 'f4c009',
                5: 'ff00ff',
                6: 'bac357',
                7: 'a3b2d2',
                8: 'd6cec2',
                9: 'bfded8',
                10: 'a9c484',
                11: '5d937b',
                12: 'a2a6a9',
                13: '777f8f',
                14: 'eab281',
                15: 'ea7286',
                16: 'f4a4bf',
                17: 'a07ca7',
                18: 'bf796d',
                19: 'f5d1b6',
                20: 'e3e19f',
                21: 'ffdf00',
                22: 'ffbf00',
                23: 'c4b454',
                24: 'f5deb3',
                25: 'f4c430',
                26: '00ffff',
                27: '89cff0',
                28: '4d4dff',
                29: '00008b',
                30: '4169e1',
                31: '006742',
                32: '4cbb17',
                33: '2e6f40',
                34: '2e8b57',
                35: 'c0c0c0',
                36: '818589',
                37: '899499',
                38: '708090',
                39: 'ffa500',
                40: 'ff8c00',
                41: 'd7942d',
                42: 'ff5f1f',
                43: 'cc7722',
                44: 'ff69b4',
                45: 'ff10f0',
                46: 'aa336a',
                47: 'f4b4c4',
                48: '953553',
                49: 'd8bfd8',
                50: '7f00ff',
                51: '800080',
                52: 'ff2400',
                53: 'ff4433',
                54: 'a52a2a',
                55: '913831',
                56: 'ff0000',
                57: '3b2219',
                58: 'a16e4b',
                59: 'd4aa78',
                60: 'e6bc98',
                61: 'ffe7d1'
            }

            # Convert hex colors to RGB tuples
            color_key_rgb = [hex_to_rgb(color_key[i]) for i in range(len(color_key))]

            # Paths
            preview_dir = exe_path_fs('game_data/stamp_preview/')
            preview_image_path = os.path.join(preview_dir, 'preview.png')
            current_stamp_dir = exe_path_fs('game_data/current_stamp_data/')
            stamp_txt_path = os.path.join(current_stamp_dir, 'stamp.txt')

            # Step 1: Manage Preview Directory
            if os.path.exists(preview_dir):
                # Clear the directory
                shutil.rmtree(preview_dir)
            # Recreate the directory
            os.makedirs(preview_dir, exist_ok=True)
            # Copy the input PNG to the preview directory as 'preview.png'
            shutil.copyfile(input_png_path, preview_image_path)
            print(f"Preview directory cleared and '{input_png_path}' copied to '{preview_image_path}'.")

            # Step 2: Process the Image
            with Image.open(input_png_path) as img:
                img = img.convert('RGBA')  # Ensure image has an alpha channel
                width, height = img.size
                pixels = img.load()

            # Calculate scaled dimensions
            scaled_width = round(width * 0.1, 1)
            scaled_height = round(height * 0.1, 1)

            # Ensure the output directory exists
            os.makedirs(current_stamp_dir, exist_ok=True)

            with open(stamp_txt_path, 'w') as f:
                # Write the first line with scaled dimensions
                f.write(f"{scaled_width},{scaled_height},img\n")
                print(f"Scaled dimensions written: {scaled_width},{scaled_height},img")

                # Iterate through pixels from bottom to top, left to right
                for y in range(height - 1, -1, -1):
                    for x in range(width):
                        pixel = pixels[x, y]

                        # Extract RGBA values
                        if len(pixel) == 4:
                            r, g, b, a = pixel
                        elif len(pixel) == 3:
                            r, g, b = pixel
                            a = 255
                        else:
                            print(f"Unexpected pixel format at ({x}, {y}): {pixel}")
                            continue  # Skip unexpected formats

                        # Skip pixels with alpha <= 191
                        if a <= 191:
                            continue

                        # Find the closest color number from the color key
                        closest_color_num = find_closest_color((r, g, b), color_key_rgb)

                        if closest_color_num is None:
                            print(f"No matching color found for pixel at ({x}, {y}): ({r}, {g}, {b})")
                            continue  # Skip if no matching color is found

                        # Scale the coordinates
                        scaled_x = round(x * 0.1, 1)
                        scaled_y = round((height - 1 - y) * 0.1, 1)

                        # Write to the file
                        f.write(f"{scaled_x},{scaled_y},{closest_color_num}\n")

            print(f"Processing complete! Output saved to: {stamp_txt_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

        self.save_current()
        self.go_to_initial_menu(True)

    def toggle_always_on_top(self, checked):
        """
        Toggles the window's 'Always on Top' property.
        """
        if checked:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            self.show_floating_message("always on top: ON", True)
        else:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            self.show_floating_message("always on top: OFF", True)
        self.show()

    def validate_image(self, entry):
        """
        Validates that the image exists and can be loaded by QPixmap.

        Args:
            entry (dict): Dictionary containing image information.

        Returns:
            bool: True if valid, False otherwise.
        """
        file_path = entry['path']
        if os.path.exists(file_path):
            try:
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    return True
            except Exception as e:
                print(f"Error loading image with QPixmap: {e}")
        return False

    def get_valid_image(self, combined_list, max_attempts=10):
        """
        Selects a random valid image from the combined list by trying up to max_attempts times.

        Args:
            combined_list (list): List of image entries.
            max_attempts (int): Maximum number of attempts to find a valid image.

        Returns:
            dict or None: Selected image entry or None if no valid images are found.
        """
        attempts = 0
        temp_list = combined_list.copy()
        while attempts < max_attempts and temp_list:
            selected = random.choice(temp_list)
            if self.validate_image(selected):
                return selected
            else:
                temp_list.remove(selected)
                attempts += 1
        return None
        
    def load_and_display_random_image(self):
        """
        Loads a random, non-animated image from either the menu_pics directory or the saved_stamps.json.
        Sets up click handlers based on the image source.
        Displays the image within the provided layout.
        """
        self.reset_movie()
        # 1. Gather images from menu_pics_dir
        menu_pics_dir = exe_path_str("imagePawcessor/menu_pics")
        if not os.path.exists(menu_pics_dir):
            QMessageBox.warning(self, "Error", f"Menu pictures directory not found: {menu_pics_dir}")
            return

        # List all image files in the directory with valid extensions, excluding .gif
        image_files = [
            f for f in os.listdir(menu_pics_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        # Normalize paths to use forward slashes
        menu_pics = [{'type': 'menu_pic', 'path': os.path.join(menu_pics_dir, f).replace("\\", "/")} for f in image_files]

        # 2. Gather images from saved_stamps.json
        appdata_dir = get_appdata_dir()
        saved_stamps_json_path = appdata_dir / "saved_stamps.json"
        saved_stamps_dir = appdata_dir / "saved_stamps"
        saved_stamp_entries = []

        if saved_stamps_json_path.exists():
            try:
                with open(saved_stamps_json_path, 'r') as f:
                    saved_stamps = json.load(f)

                for hash_key, value in saved_stamps.items():
                    if not value.get("is_gif", False):  # Skip animated GIFs
                        preview_full_path = (saved_stamps_dir / hash_key / "preview.webp").as_posix()
                        saved_stamp_entries.append({
                            'type': 'saved_stamp',
                            'hash': hash_key,
                            'path': preview_full_path
                        })
            except Exception as e:
                print(f"Failed to load saved stamps: {e}")
                # Continue with empty saved_stamp_entries

        # 3. Combine all images
        combined_images = menu_pics + saved_stamp_entries

        if not combined_images:
            QMessageBox.warning(self, "Error", "No images available to select.")
            return

        # 4. Select a valid, non-animated image with limited attempts
        selected_image_entry = self.get_valid_image(combined_images)

        if not selected_image_entry:
            QMessageBox.warning(self, "Error", "No valid, non-animated images could be loaded from the sources.")
            return

        # 5. Load and display the selected image
        pixmap = QPixmap(selected_image_entry['path'])
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load image: {selected_image_entry['path']}")
            return

        # Resize the image while maintaining aspect ratio
        if pixmap.width() > 680 or pixmap.height() > 460:
            transformation_mode = Qt.FastTransformation  # Downscaling
        else:
            transformation_mode = Qt.FastTransformation  # Upscaling

        scaled_pixmap = pixmap.scaled(
            680, 460,  # Max dimensions
            Qt.KeepAspectRatio,
            transformation_mode
        )


        # Apply transparency
        transparent_pixmap = QPixmap(scaled_pixmap.size())
        transparent_pixmap.fill(Qt.transparent)

        painter = QPainter(transparent_pixmap)
        painter.setOpacity(0.9)
        painter.drawPixmap(0, 0, scaled_pixmap)
        painter.end()

        # Disconnect any previously connected signals
        if self.connected:
            self.background_label.clicked.disconnect()

        # Connect the click event based on the image source
        if selected_image_entry['type'] == 'menu_pic':
            self.background_label.clicked.connect(lambda: self.open_image_from_menu(selected_image_entry['path']))
        elif selected_image_entry['type'] == 'saved_stamp':
            self.background_label.clicked.connect(lambda: self.load_thumbnail(selected_image_entry['hash']))

        self.connected = True
        return transparent_pixmap
        
    def display_new_stamp(self):
        self.reset_movie()
        # Check and load the appropriate file
        preview_png_path = exe_path_fs('game_data/stamp_preview/preview.png')
        preview_gif_path = exe_path_fs('game_data/stamp_preview/preview.gif')

        if self.connected:
            self.background_label.clicked.disconnect()
            self.connected = False

        if Path(preview_png_path).exists():
            # Load the PNG
            pixmap = QPixmap(str(preview_png_path))  # Convert Path to string

            # Resize the pixmap while maintaining the aspect ratio
            transformation_mode = Qt.FastTransformation  # Use hard edges
            scaled_pixmap = pixmap.scaled(
                680, 460,  # Max dimensions
                Qt.KeepAspectRatio,
                transformation_mode
            )

            # Update the label with the resized image
            self.background_label.clear()  # Clear any existing content
            self.background_label.setPixmap(scaled_pixmap)

        elif Path(preview_gif_path).exists():
            try:
                with Image.open(preview_gif_path) as gif:
                    # Extract all frames from the GIF
                    frames = []
                    durations = []
                    for frame in ImageSequence.Iterator(gif):
                        # Resize each frame with NEAREST interpolation
                        frame = frame.convert("RGBA")
                        scale_factor = min(680 / frame.width, 460 / frame.height)
                        new_size = (int(frame.width * scale_factor), int(frame.height * scale_factor))
                        resized_frame = frame.resize(new_size, Image.NEAREST)

                        # Convert to QImage for QPixmap
                        data = resized_frame.tobytes("raw", "RGBA")
                        qimage = QImage(data, resized_frame.width, resized_frame.height, QImage.Format_RGBA8888)
                        pixmap = QPixmap.fromImage(qimage)

                        # Store frame and duration
                        frames.append(pixmap)
                        durations.append(frame.info.get("duration", 100))  # Default to 100ms if no duration

                    if frames:
                        # Animate the frames using a QTimer
                        self.current_frame = 0
                        self.timer = QTimer(self)
                        self.timer.timeout.connect(lambda: self.update_gif_frame(frames))
                        self.timer.start(durations[0])  # Start with the first frame's duration
                        self.gif_frames = frames
                        self.gif_durations = durations
            except Exception as e:
                pass

        else:
            # Handle the case where neither file exists
            pass

    def update_gif_frame(self, frames):
        # Update QLabel with the current frame
        self.background_label.setPixmap(self.gif_frames[self.current_frame])

        # Increment the frame index
        self.current_frame = (self.current_frame + 1) % len(self.gif_frames)

        # Update timer interval for the next frame
        next_duration = self.gif_durations[self.current_frame]
        self.timer.start(next_duration)


    def reset_movie(self):
        # Stop the movie if it is playing
        if hasattr(self, 'movie') and self.movie is not None:
            self.movie.stop()

        # Stop the timer if using manual animation
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()
            self.timer = None

        # Clear the QLabel
        self.background_label.clear()

        # Reset related attributes
        self.movie = None
        self.gif_frames = None
        self.gif_durations = None
        self.current_frame = None
    
    def setup_secondary_menu(self):
            """
            Sets up the secondary menu with the corrected layout:
            - Image container on the left without any black borders around the image.
            - Checkboxes and sliders to the right of the image box in a vertical column.
            - Process button and accompanying elements fixed at the bottom.
            """

            # Secondary widget
            self.secondary_widget = QWidget()
            secondary_layout = QVBoxLayout()  # Main vertical layout
            secondary_layout.setContentsMargins(10, 10, 10, 10)
            self.secondary_widget.setLayout(secondary_layout)

            # Top horizontal layout for image and checkboxes
            top_layout = QHBoxLayout()

            # Image container without background
            image_container = QFrame()
            image_container.setStyleSheet("background-color: transparent;")  # Remove background
            image_container.setFixedSize(420, 300)  # Frame is the same size as the image

            # Stack layout for image and back button
            image_layout = QStackedLayout()
            image_container.setLayout(image_layout)

            # Image label
            self.image_label = QLabel("Whoops Sorry haha")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("background-color: transparent; border: none;")
            self.image_label.setFixedSize(420, 300)
            image_layout.addWidget(self.image_label)

            # Back button
            self.back_button = QPushButton(self)
            self.back_button.setStyleSheet(f"""
                QPushButton {{
                    border: none; 
                    background-color: transparent;
                    image: url({exe_path_str('imagePawcessor/font_stuff/home.svg')});
                }}
                QPushButton:hover {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/home_hover.svg')});
                }}    
                QPushButton:pressed {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/home_hover.svg')});
                }}
            """)
            self.back_button.setFixedSize(60, 60)
            self.back_button.setCursor(Qt.PointingHandCursor)
            self.back_button.clicked.connect(self.go_to_initial_menu)
            self.back_button.move(-5, -9)
            self.back_button.show()
            self.back_button.raise_()
            image_layout.addWidget(self.back_button)

            # Refresh button 60px to the right of the back button
            self.refresh_button = QPushButton(self)
            self.refresh_button.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    background-color: transparent; 
                    image: url({exe_path_str('imagePawcessor/font_stuff/refresh.svg')});
                }}
                QPushButton:hover {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/refresh_hover.svg')});
                }}    
                QPushButton:pressed {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/refresh_hover.svg')});
                }}       
            """)
            self.refresh_button.setFixedSize(60, 60)
            self.refresh_button.setCursor(Qt.PointingHandCursor)
            self.refresh_button.clicked.connect(self.reset_color_options)
            self.refresh_button.move(self.back_button.x() + 50, self.back_button.y())
            self.refresh_button.show()
            self.refresh_button.raise_()
            image_layout.addWidget(self.refresh_button)

            top_layout.addWidget(image_container)

            # Ring-style frame to wrap all options
            ring_frame = QFrame()
            ring_frame.setStyleSheet("""
                QFrame {
                    border: 3px solid #7b1fa2; /* Purple border */
                    border-radius: 15px;
                    padding: 5px;
                    margin: 0px;
                    background-color: transparent;
                }
            """)
            ring_layout = QVBoxLayout()
            ring_layout.setContentsMargins(5, 5, 5, 5)
            ring_layout.setSpacing(5)
            ring_frame.setLayout(ring_layout)

            # Title
            title_label = QLabel("Processing Options")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    margin: 0px;
                    border: none;
                    background: none;
                }
            """)
            ring_layout.addWidget(title_label)

            # Preprocess checkbox
            self.preprocess_checkbox = QCheckBox("Preprocess Image")
            self.preprocess_checkbox.setChecked(True)
            self.preprocess_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 16px;
                    color: white;
                    border: none;
                    background: none;
                    margin: 0px;
                }}
                QCheckBox::indicator {{
                    width: 24px;
                    height: 24px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/uncheck.svg')});
                }}
                QCheckBox::indicator:checked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/check.svg')});
                }}
            """)
            ring_layout.addWidget(self.preprocess_checkbox)

            # LAB Colors checkbox
            self.lab_color_checkbox = QCheckBox("Use LAB Colors")
            self.lab_color_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 16px;
                    color: white;
                    border: none;
                    background: none;
                    margin: 0px;
                }}
                QCheckBox::indicator {{
                    width: 24px;
                    height: 24px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/uncheck.svg')});
                }}
                QCheckBox::indicator:checked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/check.svg')});
                }}
            """)
            ring_layout.addWidget(self.lab_color_checkbox)

            # Placing on Canvas
            self.oncanvascheckbox = QCheckBox("Placing on Canvas")
            self.oncanvascheckbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 16px;
                    color: white;
                    border: none;
                    background: none;
                    margin: 0px;
                }}
                QCheckBox::indicator {{
                    width: 24px;
                    height: 24px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/uncheck.svg')});
                }}
                QCheckBox::indicator:checked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/check.svg')});
                }}
            """)
            ring_layout.addWidget(self.oncanvascheckbox)

            # Placing on Grass
            self.ongrasscheckbox = QCheckBox("Placing on Grass")
            self.ongrasscheckbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 16px;
                    color: white;
                    border: none;
                    background: none;
                    margin: 0px;
                }}
                QCheckBox::indicator {{
                    width: 24px;
                    height: 24px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/uncheck.svg')});
                }}
                QCheckBox::indicator:checked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/check.svg')});
                }}
            """)
            ring_layout.addWidget(self.ongrasscheckbox)

            # Chalks (client side)
            self.bg_removal_checkbox = QCheckBox("Use Chalks (client side)")
            self.bg_removal_checkbox.setStyleSheet(f"""
                QCheckBox {{
                    font-size: 16px;
                    color: white;
                    border: none;
                    background: none;
                    margin: 0px;
                }}
                QCheckBox::indicator {{
                    width: 24px;
                    height: 24px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/uncheck.svg')});
                }}
                QCheckBox::indicator:checked {{
                    image: url({exe_path_str('imagePawcessor/font_stuff/check.svg')});
                }}
            """)
            ring_layout.addWidget(self.bg_removal_checkbox)

            global has_chalks
            if not has_chalks:
                self.bg_removal_checkbox.setVisible(False)

            # Processing Method label + combobox
            processing_label = QLabel("Processing Method:")
            processing_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
            processing_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 17px;
                    font-weight: bold;
                    border: none;
                    margin-bottom: 2px;
                    padding-bottom: 0px;
                }
            """)
            ring_layout.addWidget(processing_label)

            self.processing_methods = [
                {"name": name, "description": getattr(func, "description", "")}
                for name, func in processing_method_registry.items()
            ]
            self.processing_combobox = QComboBox()
            self.processing_combobox.addItems([method["name"] for method in self.processing_methods])
            self.processing_combobox.setStyleSheet("""
                QComboBox {
                    background-color: #7b1fa2;
                    color: white;
                    border-radius: 5px;
                    font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 5px;
                    margin: 0px;
                }
                QComboBox:hover {
                    background-color: #9c27b0;
                }
                QComboBox::drop-down {
                    border-radius: 0px;
                }
                QComboBox QAbstractItemView {
                    background-color: #7b1fa2;
                    color: white;
                    selection-background-color: #9c27b0;
                }
            """)
            self.processing_combobox.currentTextChanged.connect(self.processing_method_changed)
            ring_layout.addWidget(self.processing_combobox)

            ring_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            ring_frame.setMaximumSize(256, 1000)

            # Add the ring frame to the top layout
            top_layout.addWidget(ring_frame, alignment=Qt.AlignBottom)

            # Add the top layout (image + checkboxes) to the main layout
            secondary_layout.addLayout(top_layout)

            # Group/box to hold description box + brightness & resize controls
            from PySide6.QtWidgets import QGroupBox, QTextEdit
            sliders_group_box = QGroupBox()
            sliders_group_layout = QVBoxLayout()
            sliders_group_box.setLayout(sliders_group_layout)
            sliders_group_box.setStyleSheet("QGroupBox { border: none; }")

            # The description box (read-only, bigger bold Comic Sans, arrow cursor)
            self.blank = QTextEdit()
            self.blank.setStyleSheet("""
                QTextEdit {
                    background-color: #4A148C;
                    color: white;
                    font-size: 16px; 
                    font-weight: bold;
                    font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                    border-radius: 8px;
                    padding: 8px;
                }
            """)
            self.blank.setReadOnly(True)
            self.blank.setCursor(Qt.ArrowCursor)
            self.blank.setMinimumHeight(40)
            self.blank.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.blank.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.blank.setPlainText("Sample description text here...")

            sliders_group_layout.addWidget(self.blank)

            # Small spacer
            sliders_group_layout.addSpacing(8)

            # Resize (max dim)
            resize_layout = QHBoxLayout()
            resize_layout.setSpacing(10)
            resize_label = QLabel("Resize (max dim):")
            resize_label.setAlignment(Qt.AlignTop)
            resize_layout.addWidget(resize_label, alignment=Qt.AlignTop)

            self.resize_slider = QSlider(Qt.Horizontal)
            self.resize_slider.setRange(6, 400)
            self.resize_slider.setValue(128)
            self.resize_slider.setTickInterval(10)
            self.resize_slider.setTickPosition(QSlider.TicksBelow)
            resize_layout.addWidget(self.resize_slider, alignment=Qt.AlignTop)

            self.resize_value_label = QLabel("128")
            self.resize_value_label.setAlignment(Qt.AlignTop)
            resize_layout.addWidget(self.resize_value_label, alignment=Qt.AlignTop)

            self.resize_slider.valueChanged.connect(self.resize_slider_changed)
            sliders_group_layout.addLayout(resize_layout)

            # Brightness
            self.method_options_layout = QFormLayout()
            method_options_widget = QWidget()
            method_options_widget_layout = QVBoxLayout()
            method_options_widget_layout.setAlignment(Qt.AlignTop)
            method_options_widget_layout.setContentsMargins(0, 0, 0, 0)
            method_options_widget.setLayout(method_options_widget_layout)

            brightness_layout = QHBoxLayout()
            brightness_layout.setSpacing(10)

            self.brightness_label = QLabel("Brightness:")
            self.brightness_label.setAlignment(Qt.AlignTop)
            brightness_layout.addWidget(self.brightness_label, alignment=Qt.AlignTop)

            self.brightness_slider = QSlider(Qt.Horizontal)
            self.brightness_slider.setRange(30, 80)
            self.brightness_slider.setValue(55)
            self.brightness_slider.setTickInterval(1)
            brightness_layout.addWidget(self.brightness_slider, alignment=Qt.AlignTop)



            method_options_widget_layout.addLayout(brightness_layout)
            method_options_widget_layout.addLayout(self.method_options_layout)

            sliders_group_layout.addWidget(method_options_widget)

            # Add this sliders group box to the main layout
            secondary_layout.addWidget(sliders_group_box)

            # Initialize parameter widgets
            self.parameter_widgets = {}

            # Color options
            self.setup_color_options_ui(secondary_layout)

            # Initially populate method options
            self.processing_method_changed('self.processing_combobox.currentText()')

            # Action layout for process button, status label, and progress bar
            self.action_layout = QStackedWidget()

            # Process button
            self.process_button = QPushButton("Yeaaah Process it!")
            self.process_button.setStyleSheet("""
                QPushButton {
                    background-color: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #7b1fa2, stop:1 #9c27b0);
                    color: white;
                    border-radius: 15px;
                    font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                    font-size: 24px;
                    font-weight: bold;
                    padding: 15px 30px;
                }
                QPushButton:hover {
                    background-color: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #9c27b0, stop:1 #d81b60);
                }
                QPushButton:pressed {
                    background-color: qlineargradient(
                        spread:pad, x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #6a0080, stop:1 #880e4f);
                }
            """)
            self.process_button.setMinimumHeight(60)
            self.process_button.setCursor(Qt.PointingHandCursor)
            self.process_button.clicked.connect(self.process_image)
            self.action_layout.addWidget(self.process_button)

            # Status layout with status label and progress bar
            status_widget = QWidget()
            status_layout = QVBoxLayout()
            status_widget.setLayout(status_layout)

            self.status_label = QLabel("Status: Ready")
            self.status_label.setAlignment(Qt.AlignCenter)
            self.status_label.setVisible(False)
            status_layout.addWidget(self.status_label)

            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumHeight(20)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
            status_layout.addWidget(self.progress_bar)

            self.action_layout.addWidget(status_widget)

            # Add the action layout at the bottom
            process_widget = QWidget()
            process_layout = QVBoxLayout()
            process_layout.addWidget(self.action_layout)
            process_widget.setLayout(process_layout)
            process_widget.setFixedHeight(80)
            secondary_layout.addWidget(process_widget, alignment=Qt.AlignBottom)

            # Add the secondary widget to the stacked widget
            self.stacked_widget.addWidget(self.secondary_widget)

            # Finally, set up the exclusivity for Grass/Canvas
            self.make_placement_exclusive()


    def make_placement_exclusive(self):
        """
        Ensures that only one of 'Placing on Grass' or 'Placing on Canvas' can be checked.
        Checking one will uncheck the other if it's checked.
        """
        self.oncanvascheckbox.stateChanged.connect(self._grass_canvas_exclusive)
        self.ongrasscheckbox.stateChanged.connect(self._grass_canvas_exclusive)

    def _grass_canvas_exclusive(self):
        """ Helper slot to enforce exclusivity between Grass & Canvas checkboxes. """
        # If 'Placing on Canvas' was just checked, uncheck 'Placing on Grass'
        if self.sender() == self.oncanvascheckbox and self.oncanvascheckbox.isChecked():
            self.ongrasscheckbox.setChecked(False)

        # If 'Placing on Grass' was just checked, uncheck 'Placing on Canvas'
        if self.sender() == self.ongrasscheckbox and self.ongrasscheckbox.isChecked():
            self.oncanvascheckbox.setChecked(False)


    def setup_result_menu(self):
        """
        Sets up the result menu with the processed image display and styled buttons.
        Ensures the displayed GIF or image maintains its aspect ratio and is not stretched or glued to edges.
        """
        # Result widget
        self.result_widget = QWidget()
        result_layout = QVBoxLayout()
        result_layout.setAlignment(Qt.AlignCenter)
        self.result_widget.setLayout(result_layout)
        self.current_label = QLabel("Current Stamp:")
        self.current_label.setAlignment(Qt.AlignCenter)
        self.current_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 64px; /* Larger text */
                font-weight: bold;
                border: none; /* No ring */
                margin-bottom: 2px; /* Reduced margin below the label */
            }
        """)
        result_layout.addWidget(self.current_label)


        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout.addWidget(self.result_image_label)




        # Buttons container
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.setAlignment(Qt.AlignCenter)
        button_container.setLayout(button_layout)

        # Styling for buttons (reused from the initial menu)
        button_stylesheet = """
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7b1fa2, stop:1 #9c27b0);
                color: white;
                border-radius: 15px;  /* Rounded corners */
                font-family: 'Comic Sans MS', 'Comic Neue', 'DejaVu Sans', 'FreeSans', sans-serif;
                font-size: 30px;  /* Corrected font size syntax */
                font-weight: bold;
                padding: 15px 30px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #9c27b0, stop:1 #d81b60);
            }
            QPushButton:pressed {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #6a0080, stop:1 #880e4f);
            }
        """

        # "Maybe not..." button
        self.maybe_not_button = QPushButton("Back to Options")
        self.maybe_not_button.setStyleSheet(button_stylesheet)
        self.maybe_not_button.setMinimumSize(240, 60)
        self.maybe_not_button.clicked.connect(self.retry_processing)
        button_layout.addWidget(self.maybe_not_button)

        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet(button_stylesheet)
        self.save_button.setMinimumSize(100, 60)
        # Placeholder action for Save button
        self.save_button.clicked.connect(self.save_current)
        button_layout.addWidget(self.save_button)

        # "Awrooo!" button
        self.awrooo_button = QPushButton("Home")
        self.awrooo_button.setStyleSheet(button_stylesheet)
        self.awrooo_button.setMinimumSize(120, 60)
        self.awrooo_button.clicked.connect(lambda: self.go_to_initial_menu(True))
        button_layout.addWidget(self.awrooo_button)


        # Add the button container to the result layout
        result_layout.addWidget(button_container)
        # Add result widget to stacked widget
        self.stacked_widget.addWidget(self.result_widget)


    def setup_save_menu(self):
        """
        Sets up the Save Menu with a top button layout and a dynamic, scrollable thumbnail grid.
        """
        self.save_menu_widget = QWidget()
        self.save_menu_layout = QVBoxLayout()
        self.save_menu_layout.setContentsMargins(0, 0, 0, 0)
        self.save_menu_layout.setSpacing(0)
        self.save_menu_widget.setLayout(self.save_menu_layout)

        # Set background color for the entire menu
        self.save_menu_widget.setStyleSheet("background-color: #1E1A33;")

        # Button Container
        button_container = QWidget()
        button_layout = QHBoxLayout()  # Use horizontal layout for a single row
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(16)  # Add spacing between buttons
        button_container.setLayout(button_layout)
        self.save_menu_layout.addWidget(button_container, alignment=Qt.AlignTop)

        # Add Buttons
        buttons = [
            {
                "normal": exe_path_str('imagePawcessor/font_stuff/home.svg'),
                "hover": exe_path_str("imagePawcessor/font_stuff/home_hover.svg"),
                "action": self.go_to_initial_menu,
            },
            {
                "normal": exe_path_str("imagePawcessor/font_stuff/save.svg"),
                "hover": exe_path_str("imagePawcessor/font_stuff/save_hover.svg"),
                "action": lambda: self.save_current(True),
            },
            {
                "normal": exe_path_str("imagePawcessor/font_stuff/rand.svg"),
                "hover": exe_path_str("imagePawcessor/font_stuff/rand_hover.svg"),
                "action": self.randomize_saved_stamps,
            },
            {
                "normal": exe_path_str("imagePawcessor/font_stuff/delete.svg"),
                "hover": exe_path_str("imagePawcessor/font_stuff/delete_hover.svg"),
                "action": lambda: self.toggle_delete_mode(True),
            },
        ]

        self.buttons = []  # Store button references for toggle_delete_mode
        for button_info in buttons:
            button = QPushButton()
            button.setIcon(QIcon(button_info["normal"]))
            button.setIconSize(QSize(72, 72))  # Increased icon size
            button.setFixedSize(96, 96)  # Increased button size
            button.setFlat(True)

            # Add hover effects
            normal_icon = button_info["normal"]
            hover_icon = button_info["hover"]

            button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                }
                QPushButton:hover {
                    background-color: transparent;
                }
            """)
            button.enterEvent = lambda event, b=button, h=hover_icon: b.setIcon(QIcon(h))
            button.leaveEvent = lambda event, b=button, n=normal_icon: b.setIcon(QIcon(n))

            button.clicked.connect(button_info["action"])
            button_layout.addWidget(button)

            # Save reference for toggle_delete_mode
            button_info["button"] = button
            self.buttons.append(button_info)

        # Spacer under buttons
        spacer = QSpacerItem(20, 16, QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.save_menu_layout.addSpacerItem(spacer)

        # Scrollable Grid Layout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide vertical scroll bar
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        self.save_menu_layout.addWidget(self.scroll_area)

        # Create a container widget for the grid
        grid_container = QWidget()
        self.grid_layout = QGridLayout()
        grid_container.setLayout(self.grid_layout)
        grid_container.setStyleSheet("background: transparent;")
        self.scroll_area.setWidget(grid_container)

        # Save references
        self.grid_container = grid_container

        self.populate_grid(self.grid_layout)
        self.stacked_widget.addWidget(self.save_menu_widget)

    def populate_grid(self, grid_layout):
        """
        Populates the grid with placeholders and aligns them top-left with 1-pixel borders.
        """
        self.thumbnails = []
        self.loaded_thumbnails = 0
        self.total_thumbnails = len(self.thumbnail_data)

        # Set spacing and margins
        grid_layout.setSpacing(8)
        grid_layout.setContentsMargins(6, 0, 0, 0)  # Margins around the grid

        # Clear any layout alignment constraints to ensure top-left alignment
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        for i, thumbnail_data in enumerate(self.thumbnail_data):
            thumbnail_widget = QWidget()
            thumbnail_widget.setFixedSize(128, 128)  # Icon dimensions
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            thumbnail_widget.setLayout(layout)

            # Use the processed WebP preview instead of referencing original files
            preview_path = Path(thumbnail_data["path"])
            if thumbnail_data["is_gif"]:
                # Load GIF as an animated WebP
                gif_label = QLabel()
                gif_movie = QMovie(str(preview_path))  # Use processed WebP
                gif_label.setMovie(gif_movie)
                gif_movie.start()
                layout.addWidget(gif_label)
            else:
                # Load static WebP image
                pixmap = QPixmap(str(preview_path))
                if not pixmap.isNull():
                    image_label = QLabel()
                    image_label.setPixmap(pixmap)  # Set QPixmap object
                    layout.addWidget(image_label)

            thumbnail_widget.setProperty("hash", thumbnail_data["key"])
            thumbnail_widget.mousePressEvent = lambda event, key=thumbnail_data["key"]: self.handle_thumbnail_click(event, key)

            # Ensure widgets align top-left by positioning them explicitly
            row, col = divmod(i, 5)  # 5 columns per row
            grid_layout.addWidget(thumbnail_widget, row, col, alignment=Qt.AlignTop | Qt.AlignLeft)
            self.thumbnails.append(thumbnail_widget)


    def load_visible_thumbnails(self):
        """
        Loads visible thumbnails as the user scrolls or when triggered programmatically.
        """

        if not hasattr(self, 'thumbnails') or not hasattr(self, 'thumbnail_data'):
            print("Thumbnails or thumbnail data not initialized.")
            return

        scroll_area = self.scroll_area
        visible_area = scroll_area.viewport().rect()
        viewport_top = scroll_area.verticalScrollBar().value()
        viewport_bottom = viewport_top + visible_area.height()

        for i, thumbnail_widget in enumerate(self.thumbnails):
            widget_top = thumbnail_widget.y()
            widget_bottom = widget_top + thumbnail_widget.height()

            # Check if the thumbnail is in the visible range
            if widget_bottom >= viewport_top and widget_top <= viewport_bottom:
                if not thumbnail_widget.property("loaded"):
                    thumbnail_data = self.thumbnail_data[i]
                    layout = thumbnail_widget.layout()

                    # Clear placeholder
                    for j in reversed(range(layout.count())):
                        layout.itemAt(j).widget().deleteLater()

                    if thumbnail_data["is_gif"]:
                        # Create a temporary copy of preview.webp
                        original_path = Path(thumbnail_data["path"])
                        temp_path = original_path.parent / f"temp_{original_path.name}"
                        shutil.copy(str(original_path), str(temp_path))

                        gif_label = QLabel()
                        gif_movie = QMovie(str(temp_path))
                        gif_label.setMovie(gif_movie)
                        gif_movie.start()
                        layout.addWidget(gif_label)

                        # Store reference to QMovie and temp file for later cleanup
                        thumbnail_widget.gif_movie = gif_movie
                        thumbnail_widget.temp_path = temp_path
                    else:
                        # Load static WebP image
                        pixmap = QPixmap(str(thumbnail_data["path"]))
                        if not pixmap.isNull():
                            image_label = QLabel()
                            image_label.setPixmap(pixmap)
                            layout.addWidget(image_label)

                    thumbnail_widget.setProperty("loaded", True)  # Mark as loaded



    def lazy_load_thumbnails(self):
        """
        Sets up lazy loading of thumbnails based on scroll position.
        """
        if not self.scroll_area.verticalScrollBar():
            print("Scroll area does not have a vertical scrollbar!")
            return

        self.scroll_area.verticalScrollBar().valueChanged.connect(self.load_visible_thumbnails)
        print("Lazy loading connected to scroll.")
        
    def load_thumbnail_data(self):
        """
        Load thumbnail data from saved_stamps.json and cache as QPixmap objects.
        """
        # Use the new AppData directory
        appdata_dir = get_appdata_dir()
        saved_stamps_json = appdata_dir / "saved_stamps.json"
        saved_stamps_dir = appdata_dir / "saved_stamps/"

        self.thumbnail_data = []
        self.thumbnail_cache = {}

        if not saved_stamps_json.exists():
            print("No saved_stamps.json file found.")
            return

        try:
            with open(saved_stamps_json, 'r') as f:
                saved_stamps = json.load(f)
        except Exception as e:
            print(f"Error reading saved_stamps.json: {e}")
            return

        # Define the filenames we consider valid previews
        valid_preview_files = ["preview.webp", "preview.png", "preview.gif"]

        for key, value in saved_stamps.items():
            folder_path = saved_stamps_dir / key

            # Look for any valid preview file in the folder
            found_preview_path = None
            for filename in valid_preview_files:
                candidate_path = folder_path / filename
                if candidate_path.exists():
                    found_preview_path = candidate_path
                    break

            # If no preview file is found, print a message and move on
            if not found_preview_path:
                print(f"Missing any valid preview file for key: {key}")
                continue

            # Try loading the found preview file into a QPixmap
            try:
                pixmap = QPixmap(str(found_preview_path))
                if not pixmap.isNull():
                    self.thumbnail_cache[key] = pixmap
                    self.thumbnail_data.append({
                        "path": str(found_preview_path),
                        "is_gif": value.get("is_gif", False),
                        "key": key
                    })
                else:
                    print(f"Failed to load pixmap for {found_preview_path}")
            except Exception as e:
                print(f"Error loading pixmap for {found_preview_path}: {e}")

        print(f"Loaded {len(self.thumbnail_data)} thumbnails.")


    def handle_thumbnail_click(self, event, thumbnail_hash):
        """
        Handles a click on a thumbnail, performing an action based on delete mode.
        """
        if getattr(self, 'delete_mode', False):
            # Call the delete function if delete mode is enabled
            self.delete_thumbnail(thumbnail_hash)
        else:
            # Call the load function if delete mode is disabled
            self.load_thumbnail(thumbnail_hash)


    def delete_thumbnail(self, thumbnail_hash):
        """
        Removes the entry for the specified hash from saved_stamps.json.
        """
        # Use the new AppData directory
        appdata_dir = get_appdata_dir()
        saved_stamps_json = appdata_dir / "saved_stamps.json"

        try:
            # Load and update saved_stamps.json
            if saved_stamps_json.exists():
                with open(saved_stamps_json, "r") as json_file:
                    saved_stamps = json.load(json_file)

                if thumbnail_hash in saved_stamps:
                    print(f"Removing entry for hash {thumbnail_hash} from JSON.")
                    del saved_stamps[thumbnail_hash]

                    # Write updated JSON back to file
                    with open(saved_stamps_json, "w") as json_file:
                        json.dump(saved_stamps, json_file, indent=4)
                else:
                    print(f"Hash {thumbnail_hash} not found in JSON.")

                self.show_floating_message("Entry Deleted")
            else:
                print("No saved_stamps.json file found.")
                self.show_floating_message("Error", True)

        except Exception as e:
            print(f"Error while deleting JSON entry: {e}")
            self.show_floating_message("Error", True)

        self.repopulate_grid()

    def load_thumbnail(self, thumbnail_hash):
        """
        Loads a thumbnail by replacing files in the current_stamp_data directory
        with files from the corresponding hash directory.
        """
        # Use the new AppData directory for saved stamps
        appdata_dir = get_appdata_dir()
        saved_stamp_dir = appdata_dir / "saved_stamps" / thumbnail_hash
        current_stamp_dir = exe_path_fs("game_data/current_stamp_data/")

        if not saved_stamp_dir.exists():
            self.show_floating_message("Directory Not Found", True)
            return

        try:
            # Ensure the current_stamp_data directory exists
            current_stamp_dir.mkdir(parents=True, exist_ok=True)

            # Replace stamps.txt if it exists in the hash directory
            stamps_file = saved_stamp_dir / "stamp.txt"
            if stamps_file.exists():
                target_stamps_file = current_stamp_dir / "stamp.txt"
                target_stamps_file.write_text(stamps_file.read_text())

            # Replace frames.txt if it exists in the hash directory
            frames_file = saved_stamp_dir / "frames.txt"
            if frames_file.exists():
                target_frames_file = current_stamp_dir / "frames.txt"
                target_frames_file.write_text(frames_file.read_text())

            self.show_floating_message("Loaded!")

        except Exception as e:
            print(f"Error while loading thumbnail: {e}")
            self.show_floating_message("Error", True)


    def repopulate_grid(self):
        """
        Clears and repopulates the grid layout with updated thumbnail data in order.
        """
        if not hasattr(self, 'grid_layout') or not hasattr(self, 'thumbnail_data'):
            print("Grid layout or thumbnail data not initialized.")
            return

        # Clear existing widgets in the grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        for thumbnail_widget in self.thumbnails:
            if hasattr(thumbnail_widget, 'gif_movie'):
                thumbnail_widget.gif_movie.stop()
                thumbnail_widget.gif_movie.deleteLater()
        # Reset thumbnails list and reload data
        self.thumbnails = []
        self.loaded_thumbnails = 0

        self.load_thumbnail_data()  # Reload data from JSON
        self.populate_grid(self.grid_layout)  # Repopulate the grid

        # Trigger immediate visible thumbnail loading
        self.load_visible_thumbnails()


    def toggle_delete_mode(self, callback = True):
        """
        Toggles delete mode and makes the 'delete' button in the top menu shake
        when delete mode is enabled.
        """
        # Toggle delete mode
        self.delete_mode = not self.delete_mode

        # Find the delete button in the menu
        delete_button = None
        for button_info in self.buttons:
            if button_info["normal"].endswith("delete.svg"):
                delete_button = button_info["button"]
                break

        if not delete_button:
            print("Delete button not found!")
            return

        if self.delete_mode:
            # Create animation for shaking effect
            animation = QPropertyAnimation(delete_button, b"pos")
            animation.setDuration(100)
            animation.setLoopCount(-1)  # Loop indefinitely
            current_pos = delete_button.pos()

            # Define random shaking movement
            animation.setKeyValueAt(0, current_pos)
            animation.setKeyValueAt(0.25, current_pos + QPoint(random.randint(-8, 8), random.randint(-8, 8)))
            animation.setKeyValueAt(0.5, current_pos + QPoint(random.randint(-8, 8), random.randint(-8, 8)))
            animation.setKeyValueAt(0.75, current_pos + QPoint(random.randint(-8, 8), random.randint(-8, 8)))
            animation.setKeyValueAt(1, current_pos)  # Back to center

            animation.start()
            delete_button.animation = animation  # Store reference to prevent garbage collection
        else:
            # Stop shaking
            if hasattr(delete_button, "animation"):
                delete_button.animation.stop()
                del delete_button.animation
                
        if callback:
            if self.delete_mode:
                self.show_floating_message("Click to DELETE", True)
            else:
                self.show_floating_message("Delete Off", True)
            

    def show_save_menu(self):
        """
        Switch to the save menu screen.
        """


        global first 
        if not first:
            cleanup_saved_stamps()
            first = True

        if self.processing:
            return
        
        self.repopulate_grid()
        if not hasattr(self, 'thumbnail_data'):
            self.load_thumbnail_data()

        if not hasattr(self, 'save_menu_widget'):
            self.setup_save_menu()
            
        self.stacked_widget.setCurrentWidget(self.save_menu_widget)

        self.delete_mode = True
        self.toggle_delete_mode(False)

    def close_application(self):
        for timer in self.color_timers.values():
            timer.stop()
        self.close()
        

    def update_cluster_label(self):
        """
        Updates the cluster count label dynamically as the slider changes.
        """
        self.cluster_label_value.setText(str(self.cluster_slider.value()))


    def retry_processing(self):
        
        if not hasattr(self, 'secondary_widget'):
            self.setup_secondary_menu()
            
        self.stacked_widget.setCurrentWidget(self.secondary_widget)
        # Ensure the back button is visible and brought to the front
        if self.back_button:
            self.back_button.show()
            self.back_button.raise_()
        if self.refresh_button:
            self.refresh_button.show()
            self.refresh_button.raise_()
                
    def resize_slider_changed(self, value):
        self.resize_value_label.setText(str(value))
        if not self.manual_change:
            if not self.is_gif:
                if value > 200:
                    if "Hybrid Dither" in [method["name"] for method in self.processing_methods]:
                        self.processing_combobox.blockSignals(True)  # Block signals
                        self.processing_combobox.setCurrentText("Hybrid Dither")
                        self.processing_combobox.blockSignals(False)  # Unblock signals
                        self.processing_method_changed("Hybrid Dither", strength=False, manual=False)
                elif value > 64:
                    if "Pattern Dither" in [method["name"] for method in self.processing_methods]:
                        self.processing_combobox.blockSignals(True)
                        self.processing_combobox.setCurrentText("Pattern Dither")
                        self.processing_combobox.blockSignals(False)
                        self.processing_method_changed("Pattern Dither", strength=False, manual=False)
                else:
                    if "Color Match" in [method["name"] for method in self.processing_methods]:
                        self.processing_combobox.blockSignals(True)
                        self.processing_combobox.setCurrentText("Color Match")
                        self.processing_combobox.blockSignals(False)
                        self.processing_method_changed("Color Match", strength=False, manual=False)
            else:
                if value > 80:
                    if "Pattern Dither" in [method["name"] for method in self.processing_methods]:
                        self.processing_combobox.blockSignals(True)
                        self.processing_combobox.setCurrentText("Pattern Dither")
                        self.processing_combobox.blockSignals(False)
                        self.processing_method_changed("Pattern Dither", strength=False, manual=False)
                else:
                    if "Color Match" in [method["name"] for method in self.processing_methods]:
                        self.processing_combobox.blockSignals(True)
                        self.processing_combobox.setCurrentText("Color Match")
                        self.processing_combobox.blockSignals(False)
                        self.processing_method_changed("Color Match", strength=False, manual=False)



    def setup_color_options_ui(self, layout):
        """
        Creates a 1x6 grid layout of color options with:
        - Color squares (100x100) in a 1x6 grid.
        - Options (Enable, RGB, Blank checkboxes) inside each color square.
        - Boost and Threshold labels and sliders horizontally aligned beneath each color square.
        - Boost text and slider appear only when Preprocess Image is checked.
        """
        
        # Helper function to determine text color based on background color
        def get_contrast_color(hex_color):
            """
            Returns 'white' or 'black' based on the luminance of the provided hex color.
            """
            # Convert hex to RGB
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif len(hex_color) == 3:
                r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
            else:
                # Default to white if format is unexpected
                return 'white'
            
            # Calculate luminance using the formula
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            return 'white' if luminance < 100 else 'black'

        # Decorative ring-style border for the entire section
        ring_frame = QFrame()
        ring_frame.setStyleSheet("""
            QFrame {
                border: 4px solid #7b1fa2; /* Increased Purple border */
                border-radius: 15px;
                padding: 4px; /* Inner padding */
                margin: 0px;  /* Outer margin */
            }
        """)
        ring_layout = QVBoxLayout()
        ring_layout.setSpacing(1)  # Increased spacing for better layout
        ring_layout.setContentsMargins(5, 5, 5, 5)  # Increased inner padding
        ring_frame.setLayout(ring_layout)

        # Grid layout for color squares
        color_layout = QGridLayout()
        color_layout.setHorizontalSpacing(1)  # Adjusted horizontal spacing
        color_layout.setVerticalSpacing(25)    # Adjusted vertical spacing for better alignment
        ring_layout.addLayout(color_layout)

        # Initialize dictionaries to track widgets and border colors
        self.color_checkboxes = {}
        self.rgb_checkboxes = {}
        self.blank_checkboxes = {}
        self.boost_sliders = {}
        self.boost_labels = {}
        self.threshold_sliders = {}
        self.threshold_labels = {}
        self.color_labels = {}
        self.color_timers = {}  # Initialize timers for RGB animation
        self.border_colors = {}  # Store original border colors

        # Track the currently selected Boost and Blank
        self.current_boost_color = None
        self.current_blank_color = None

        # Create the 1x6 grid of color options
        for i, color in enumerate(self.default_color_key_array):
            color_number = color['number']  # Use the actual color number from the array
            color_hex = color['hex']

            text_color = get_contrast_color(color_hex)

            # Dynamically set icons based on text color
            if text_color == "white":
                unchecked_icon = exe_path_str('imagePawcessor/font_stuff/uncheck_white.svg')
                checked_icon = exe_path_str('imagePawcessor/font_stuff/check_white.svg')
                border_color = "#ffffff"  # White border for light text
            else:
                unchecked_icon = exe_path_str('imagePawcessor/font_stuff/uncheck.svg')
                checked_icon = exe_path_str('imagePawcessor/font_stuff/check.svg')
                border_color = "#e3a8e6"

            # Store the border color
            self.border_colors[color_number] = border_color

            # Create a container widget for the color box and its options
            color_container = QWidget()
            color_container_layout = QVBoxLayout()
            color_container_layout.setAlignment(Qt.AlignTop)
            color_container_layout.setSpacing(10)
            color_container_layout.setContentsMargins(0, 0, 0, 0)
            color_container.setLayout(color_container_layout)

            # Color box (replacing QLabel with QWidget)
            color_box = QWidget()
            color_box.setFixedSize(100, 100)
            color_box.setStyleSheet(f"""
                QWidget {{
                    background-color: #{color_hex};
                    border: 4px solid {border_color}; /* Dynamic border color */
                    border-radius: 10px;
                }}
            """)
            color_container_layout.addWidget(color_box, alignment=Qt.AlignCenter)

            # Layout for checkboxes inside the color box
            checkbox_layout = QVBoxLayout()
            checkbox_layout.setSpacing(8)
            checkbox_layout.setContentsMargins(8, 10, 5, 5)  # Offset checkboxes by 5 pixels right and down
            color_box.setLayout(checkbox_layout)

            # Define a common stylesheet template for the checkboxes
            checkbox_stylesheet = f"""
                QCheckBox {{
                    color: {text_color}; /* Dynamic text color based on background */
                    font-size: 15px; /* Adjusted font size */
                    font-weight: bold;
                    background: transparent; /* Ensure no background */
                    border: none; /* Remove any border/frame */
                }}
                QCheckBox::indicator {{
                    width: 20px;
                    height: 20px;
                }}
                QCheckBox::indicator:unchecked {{
                    image: url({unchecked_icon});
                }}
                QCheckBox::indicator:checked {{
                    image: url({checked_icon});
                }}
            """

            # Enable checkbox
            enable_checkbox = QCheckBox("Enable")
            enable_checkbox.setChecked(True)
            enable_checkbox.setStyleSheet(checkbox_stylesheet)
            enable_checkbox.toggled.connect(
                lambda checked, num=color_number: self.toggle_enable_options(num, checked)
            )
            self.color_checkboxes[color_number] = enable_checkbox
            checkbox_layout.addWidget(enable_checkbox)

            # RGB checkbox
            rgb_checkbox = QCheckBox("RGB")
            rgb_checkbox.setStyleSheet(checkbox_stylesheet)
            rgb_checkbox.toggled.connect(
                lambda checked, num=color_number: self.toggle_rgb(num, checked)
            )
            self.rgb_checkboxes[color_number] = rgb_checkbox
            checkbox_layout.addWidget(rgb_checkbox)

            # Blank checkbox
            blank_checkbox = QCheckBox("Blank")
            blank_checkbox.setStyleSheet(checkbox_stylesheet)
            blank_checkbox.toggled.connect(
                lambda checked, num=color_number: self.toggle_blank(num, checked)
            )
            self.blank_checkboxes[color_number] = blank_checkbox
            checkbox_layout.addWidget(blank_checkbox)


            # Spacer to push checkboxes to the top
            checkbox_layout.addStretch()

            # Boost label
            boost_label = QLabel("Boost")
            boost_label.setAlignment(Qt.AlignCenter)
            boost_label.setStyleSheet("""
                QLabel {
                    color: white; /* Always white */
                    font-size: 17px; /* Adjusted font size */
                    font-weight: bold;
                    border: none; /* No ring */
                    margin-bottom: 0px; /* Reduce bottom margin */
                    padding-bottom: 0px; /* Reduce bottom padding */
                    background: transparent; /* Ensure no background */
                }
            """)
            boost_label.setVisible(False)
            self.boost_labels[color_number] = boost_label
            color_container_layout.addWidget(boost_label)

            # Boost slider
            boost_slider = QSlider(Qt.Horizontal)
            boost_slider.setRange(0, 27)
            boost_slider.setValue(14)
            boost_slider.setTickInterval(1)
            boost_slider.setTickPosition(QSlider.TicksBelow)
            boost_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #7b1fa2;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #ffffff;
                    border: 1px solid #7b1fa2;
                    width: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
            """)
            boost_slider.setVisible(False)
            self.boost_sliders[color_number] = boost_slider
            boost_slider.setFixedWidth(100)  # Set the width to match the color box
            color_container_layout.addWidget(boost_slider, alignment=Qt.AlignCenter)

            # Threshold label
            threshold_label = QLabel("Threshold")
            threshold_label.setAlignment(Qt.AlignCenter)
            threshold_label.setStyleSheet("""
                QLabel {
                    color: white; /* Always white */
                    font-size: 14px; /* Adjusted font size */
                    font-weight: bold;
                    border: none; /* No ring */
                    margin-bottom: 0px; /* Reduce bottom margin */
                    padding-bottom: 0px; /* Reduce bottom padding */
                    background: transparent; /* Ensure no background */
                }
            """)
            threshold_label.setVisible(False)
            self.threshold_labels[color_number] = threshold_label
            color_container_layout.addWidget(threshold_label)

            # Threshold slider
            threshold_slider = QSlider(Qt.Horizontal)
            threshold_slider.setRange(0, 100)
            threshold_slider.setValue(20)
            threshold_slider.setTickInterval(1)
            threshold_slider.setTickPosition(QSlider.TicksBelow)
            threshold_slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 6px;
                    background: #7b1fa2;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #ffffff;
                    border: 1px solid #7b1fa2;
                    width: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }
            """)
            threshold_slider.setVisible(False)
            self.threshold_sliders[color_number] = threshold_slider
            threshold_slider.setFixedWidth(100)  # Set the width to match the color box
            color_container_layout.addWidget(threshold_slider, alignment=Qt.AlignCenter)

            # Add a spacer to ensure Boost and Threshold space remains consistent
            #color_container_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

            # Assign the color_box to self.color_labels for reference
            self.color_labels[color_number] = color_box

            # Add the color container to the grid layout
            row = 0
            col = i
            color_layout.addWidget(color_container, row, col)

        # Add the ring frame to the parent layout
        layout.addWidget(ring_frame, alignment=Qt.AlignBottom)

        # Connect toggle functions
        self.preprocess_checkbox.toggled.connect(self.toggle_boost_elements)
        self.bg_removal_checkbox.toggled.connect(self.brightness_toggle)
        self.lab_color_checkbox.toggled.connect(self.lab_value_toggle)


    def toggle_enable_options(self, color_number, enabled):
        """
        Enables/disables RGB and Blank checkboxes based on the state of the Enable checkbox.
        Prevents disabling the last enabled color.
        """
        # Ensure at least one color remains enabled
        if not enabled and all(not cb.isChecked() for cb in self.color_checkboxes.values()):
            self.color_checkboxes[color_number].setChecked(True)
            QMessageBox.warning(self, "Warning", "At least one color must remain enabled.")
            return

        # Show or hide the RGB and Blank checkboxes based on Enable state
        self.rgb_checkboxes[color_number].setVisible(enabled)
        self.blank_checkboxes[color_number].setVisible(enabled)

        if enabled and self.preprocess_checkbox.isChecked():
            self.boost_labels[color_number].setVisible(False)
            self.boost_sliders[color_number].setVisible(False)
            self.threshold_labels[color_number].setVisible(False)
            self.threshold_sliders[color_number].setVisible(False)
        else:
            self.boost_labels[color_number].setVisible(False)
            self.boost_sliders[color_number].setVisible(False)
            self.threshold_labels[color_number].setVisible(False)
            self.threshold_sliders[color_number].setVisible(False)


    def brightness_toggle(self, checked):
        """
        Toggles the visibility of Boost labels and sliders based on the state of the Preprocess Image checkbox.
        They only reappear if their corresponding color box is enabled.
        """
        if not checked and self.preprocess_checkbox.isChecked():
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)
        else:
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)

    def toggle_boost_elements(self, checked):
        """
        Toggles the visibility of Boost labels and sliders based on the state of the Preprocess Image checkbox.
        They only reappear if their corresponding color box is enabled.
        """
        if checked and not self.bg_removal_checkbox.isChecked():
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)
        else:
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)

        for color_number in self.boost_labels:
            if checked and self.color_checkboxes[color_number].isChecked():  
                self.boost_labels[color_number].setVisible(False)
                self.boost_sliders[color_number].setVisible(False)
                self.threshold_labels[color_number].setVisible(False)
                self.threshold_sliders[color_number].setVisible(False)
            else:  
                self.boost_labels[color_number].setVisible(False)
                self.boost_sliders[color_number].setVisible(False)
                self.threshold_labels[color_number].setVisible(False)
                self.threshold_sliders[color_number].setVisible(False)


    def toggle_blank(self, color_number, checked):
        """
        Toggles the border visibility of the color square based on the Blank checkbox state.
        Ensures mutual exclusivity with the RGB checkbox.
        """
        # Find the corresponding color data
        color_data = next((color for color in self.default_color_key_array if color['number'] == color_number), None)
        if not color_data:
            print(f"Error: No color data found for color_number {color_number}")
            return

        if checked:
            # Uncheck the currently active blank if there is one
            if self.current_blank_color is not None and self.current_blank_color != color_number:
                self.blank_checkboxes[self.current_blank_color].setChecked(False)
            self.current_blank_color = color_number
            self.rgb_checkboxes[color_number].setChecked(False)
            self.color_labels[color_number].setStyleSheet(f"""
                QWidget {{
                    background-color: #{color_data['hex']};
                    border: none;
                    border-radius: 10px;
                }}
            """)
        else:
            self.current_blank_color = None
            # Use the stored border color instead of hardcoding
            border_color = self.border_colors.get(color_number, "#ffffff")
            self.color_labels[color_number].setStyleSheet(f"""
                QWidget {{
                    background-color: #{color_data['hex']};
                    border: 4px solid {border_color};
                    border-radius: 10px;
                }}
            """)


    def toggle_rgb(self, color_number, checked):
        """
        Toggles RGB animation for the color square and ensures mutual exclusivity with the Blank checkbox.
        """
        # Find the corresponding color data
        color_data = next((color for color in self.default_color_key_array if color['number'] == color_number), None)
        if not color_data:
            print(f"Error: No color data found for color_number {color_number}")
            return

        if checked:
            # Uncheck the currently active RGB if there is one
            if self.current_boost_color is not None and self.current_boost_color != color_number:
                self.rgb_checkboxes[self.current_boost_color].setChecked(False)
            self.current_boost_color = color_number
            self.blank_checkboxes[color_number].setChecked(False)

            # Stop existing timers for the color to avoid duplicates
            if color_number in self.color_timers:
                self.color_timers[color_number].stop()
                del self.color_timers[color_number]

            # Start the RGB animation
            timer = QTimer(self)
            timer.setInterval(100)  
            timer.timeout.connect(lambda: self.update_rgb_border(color_number))
            self.color_timers[color_number] = timer
            timer.start()
        else:
            # Stop the RGB animation and reset the border
            if color_number in self.color_timers:
                self.color_timers[color_number].stop()
                del self.color_timers[color_number]

            self.current_boost_color = None
            # Use the stored border color instead of hardcoding
            border_color = self.border_colors.get(color_number, "#ffffff")
            self.color_labels[color_number].setStyleSheet(f"""
                QWidget {{
                    background-color: #{color_data['hex']};
                    border: 4px solid {border_color};
                    border-radius: 10px;
                }}
            """)


    def update_rgb_border(self, color_number):
        """
        Updates the border color of the specified color square to cycle through RGB.
        """
        color_label = self.color_labels[color_number]
        rgb_cycle = ["red", "green", "blue"]

        # Extract current border color from the stylesheet
        current_style = color_label.styleSheet()
        current_color = next((color for color in rgb_cycle if f"border: 4px solid {color}" in current_style), None)
        if current_color is None:
            next_color = "red"
        else:
            next_color = rgb_cycle[(rgb_cycle.index(current_color) + 1) % len(rgb_cycle)]

        # Get the background color
        color_data = next(
            (color for color in self.default_color_key_array if color['number'] == color_number),
            None
        )
        if not color_data:
            hex_color = "ffffff"
        else:
            hex_color = color_data['hex']

        # Update the widget's stylesheet with the next RGB color
        color_label.setStyleSheet(
            f"background-color: #{hex_color}; border: 4px solid {next_color}; border-radius: 10px;"
        )


    def open_image_from_files(self):
        # Clear any previous states
        if not hasattr(self, 'secondary_widget'):
            self.setup_secondary_menu()
        if self.processing:
            return
        self.reset_to_initial_state()     
        file_dialog = QFileDialog(self)
        # Allow both images and MP4 videos to be picked.
        file_dialog.setNameFilters(["Images and Videos (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.mp4)"])
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.image_path = file_path
            self.load_image(file_path)

            if not hasattr(self, 'secondary_widget'):
                self.setup_secondary_menu()
                    
            self.stacked_widget.setCurrentWidget(self.secondary_widget)

    def open_image_from_menu(self, path):
        if not hasattr(self, 'secondary_widget'):
            self.setup_secondary_menu()
        self.reset_to_initial_state()   
        self.image_path = path
        self.load_image(path)
        self.stacked_widget.setCurrentWidget(self.secondary_widget)


    def open_image_from_clipboard(self, center = False):
        """
        Handles retrieving image content from the clipboard, ensuring proper handling of
        both static and animated images. Saves the clipboard image as a WebP file for further processing.
        """


        if not hasattr(self, 'secondary_widget'):
            self.setup_secondary_menu()

        if self.processing:
            return

        self.canpaste = False
        try:
            # Step 1: Retrieve clipboard content
            clipboard_content = get_clipboard_image()
            if clipboard_content is None:
                self.show_floating_message("No image found", center)  # Floating message instead of error popup
                self.canpaste = True
                return

            # Step 2: Handle clipboard containing file paths
            if isinstance(clipboard_content, list):
                # Filter for valid image files
                image_files = [f for f in clipboard_content if os.path.isfile(f)]
                if not image_files:
                    self.show_floating_message("No image found", center)  # Floating message
                    self.canpaste = True
                    return

                # Open the first image file in the list
                img = Image.open(image_files[0])

            # Step 3: Handle clipboard containing a direct image
            elif isinstance(clipboard_content, Image.Image):
                img = clipboard_content
            else:
                self.show_floating_message("Clipboard does not contain an image or image file") 
                self.canpaste = True
                return
            
            self.reset_to_initial_state()   
            self.canpaste = False
            # Step 4: Detect if the image is animated
            is_multiframe = getattr(img, "is_animated", False)
            # Step 5: Save the image to a temporary WebP file in directory
            temp_image_path = exe_path_fs('imagePawcessor/temp/clipboard_image.webp')
            if is_multiframe:
                # Save as an animated WebP
                img.save(temp_image_path, format="WEBP", save_all=True, duration=img.info.get("duration", 100), loop=img.info.get("loop", 0))
            else:
                # Save as a static WebP
                img.save(temp_image_path, format="WEBP")
            print(temp_image_path)
            # Step 6: Pass the temporary file path to load_image
            self.image_path = temp_image_path  # Indicate clipboard source
            self.load_image(temp_image_path)  # Treat like a regular image or GIF
            self.stacked_widget.setCurrentWidget(self.secondary_widget)
            self.canpaste = True

        except Exception as e:
            self.show_floating_message(f"Failed to process clipboard: {str(e)}") 
            self.canpaste = True

    def reset_color_options(self):

        for color_number in self.color_checkboxes.keys():
            # Enable all colors
            self.color_checkboxes[color_number].setChecked(True)
            # Disable RGB and Blank checkboxes
            self.rgb_checkboxes[color_number].setChecked(False)
            self.rgb_checkboxes[color_number].setVisible(True)  # Ensure visibility
            self.blank_checkboxes[color_number].setChecked(False)
            self.blank_checkboxes[color_number].setVisible(True)  # Ensure visibility

            # Reset Boost sliders to 1.2 (value 12)
            if color_number in self.boost_sliders:
                self.boost_sliders[color_number].setValue(14)
                self.boost_sliders[color_number].setVisible(False)  # Reset visibility

            # Reset Threshold sliders to 20
            if color_number in self.threshold_sliders:
                self.threshold_sliders[color_number].setValue(20)
                self.threshold_sliders[color_number].setVisible(False)  # Reset visibility

            # Hide Boost and Threshold labels
            if color_number in self.boost_labels:
                self.boost_labels[color_number].setVisible(False)
            if color_number in self.threshold_labels:
                self.threshold_labels[color_number].setVisible(False)

        if self.is_gif:
            if "Color Match" in [method["name"] for method in self.processing_methods]:
                self.processing_combobox.setCurrentText("Color Match")
                self.processing_method_changed("Color Match")

        else:
            if "Pattern Dither" in [method["name"] for method in self.processing_methods]:
                self.processing_combobox.setCurrentText("Pattern Dither")
                self.processing_method_changed("Pattern Dither")

        self.preprocess_checkbox.setChecked(True)
        self.lab_color_checkbox.setChecked(True)
        self.oncanvascheckbox.setChecked(False)
        self.ongrasscheckbox.setChecked(False)
        if not self.bg_removal_checkbox.isChecked():
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)
        else:
            self.brightness_label.setVisible(True)
            self.brightness_slider.setVisible(True)

        for color_number in self.boost_labels:
                self.boost_labels[color_number].setVisible(False)
                self.boost_sliders[color_number].setVisible(False)
                self.threshold_labels[color_number].setVisible(False)
                self.threshold_sliders[color_number].setVisible(False)

        self.brightness_slider.setValue(55)
        self.manual_change = False
        self.resize_slider_changed(self.resize_slider.value())

    def lab_value_toggle(self, checked):
        for color_number in self.boost_labels:
            if checked:
                if color_number in self.boost_sliders:
                    self.boost_sliders[color_number].setValue(14)

                # Reset Threshold sliders to 20
                if color_number in self.threshold_sliders:
                    self.threshold_sliders[color_number].setValue(20)

                
                self.brightness_slider.setValue(55)


            else:

                if color_number in self.boost_sliders:
                    self.boost_sliders[color_number].setValue(12)


                # Reset Threshold sliders to 20
                if color_number in self.threshold_sliders:
                    self.threshold_sliders[color_number].setValue(28)
                
                self.brightness_slider.setValue(55)





    @Slot(str, bool)
    def show_floating_message(self, message, centered=False):
        """
        Creates a floating particle effect for the given message.
        The message drifts upward with exaggerated Y-axis movement, random wandering on the X-axis,
        sporadic chaotic movement, and an additional chance for extreme rapid shaking.

        Parameters:
            message (str): The text to display.
            centered (bool): If True, the message originates from the center of the GUI near the bottom.
        """
        # Doge meme-inspired colors
        doge_colors = ['#FFDD00', '#FF4500', '#1E90FF', '#32CD32', '#FF69B4', '#9400D3']

        # Create the floating label
        label = QLabel(message, self)
        random_color = random.choice(doge_colors)
        label.setStyleSheet(f"""
            QLabel {{
                color: {random_color};  /* Doge color */
                font-size: 36px;  /* Larger text */
                font-weight: 900; /* Extra bold */
                background-color: transparent;
            }}
        """)
        label.setAttribute(Qt.WA_TransparentForMouseEvents)  # Ignore mouse events
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)  # Enable text wrapping

        # Adjust size to fit the GUI width
        max_width = self.width() - 40  # Allow padding from the edges
        label.setFixedWidth(max_width)
        label.adjustSize()

        # Calculate start position
        if centered:
            label_width = label.width()
            label_height = label.height()
            start_x = (self.width() - label_width) // 2
            start_y = self.height() - 50 - label_height
            start_pos = QPoint(start_x, start_y)
        else:
            # Start position: center the label on the cursor
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            label_width = label.width()
            label_height = label.height()
            start_pos = QPoint(cursor_pos.x() - label_width // 2, cursor_pos.y() - label_height // 2)

        label.move(start_pos)
        label.show()

        # Random tilt (±5 degrees)
        rotation_angle = random.uniform(-5, 5)
        label.setStyleSheet(label.styleSheet() + f"""
            transform: rotate({rotation_angle}deg);
        """)

        # Animation: Exaggerated upward movement with sporadic chaos
        move_animation = QPropertyAnimation(label, b"pos", self)
        move_animation.setDuration(5000)  # 5 seconds
        move_animation.setStartValue(start_pos)

        # Randomized end position with large vertical drift and sporadic horizontal wandering
        end_x = start_pos.x() + random.randint(-100, 100)  # Wider horizontal range
        end_y = start_pos.y() - random.randint(600, 1000)  # Extreme upward drift

        # Add a chance for chaotic movement
        if random.random() < 0.3:  # 30% chance for sporadic movement
            mid_x = start_pos.x() + random.randint(-200, 200)
            mid_y = start_pos.y() - random.randint(200, 400)
            move_animation.setKeyValueAt(0.5, QPoint(mid_x, mid_y))  # Insert chaos mid-way
            

        # Add a chance for continuous rapid shaking
        if random.random() < 0.05:
            for i in range(70):
                shake_x = start_pos.x() + random.randint(-100, 100)
                shake_y = start_pos.y() - (i * (start_pos.y() - end_y) // 40) + random.randint(-30, 30)
                move_animation.setKeyValueAt(i / 70, QPoint(shake_x, shake_y))

        move_animation.setEndValue(QPoint(end_x, end_y))
        move_animation.setEasingCurve(QEasingCurve.OutQuad)

        # Animation: Fade out the label
        fade_animation = QPropertyAnimation(label, b"windowOpacity", self)
        fade_animation.setDuration(5000)  # 5 seconds
        fade_animation.setStartValue(1)  # Fully opaque
        fade_animation.setEndValue(0)  # Fully transparent

        # Start both animations
        move_animation.start()
        fade_animation.start()

        # Ensure the label is deleted after the animations are done
        fade_animation.finished.connect(label.deleteLater)

    def load_image(self, file_path):
        """
        Loads an image file or video file. For static images and animated images (GIF, WebP),
        the existing logic is used. If the file is an MP4 video, the new display_video function
        is called to create and show a preview GIF.
        """     
        try:
            # If the file is a video (e.g. MP4), process it with display_video.
            isvideo = False
            # Convert file_path to string for safe lower() comparison.
            if str(file_path).lower().endswith('.mp4'):
                self.is_gif = True
                isvideo = True
                self.image_path = file_path
                # Optionally, adjust any UI elements or sliders for video here.
                self.display_video(file_path)
                self.resize_slider.setMaximum(160)
                self.resize_slider.setValue(96)
            else:
                # Otherwise, open with PIL.
                img = Image.open(file_path)
                is_animated = getattr(img, "is_animated", False)

                if is_animated:
                    # Handle animated content (GIF or WebP)
                    self.is_gif = True
                    self.image_path = file_path

                    image_width, image_height = img.size
                    max_dimension = max(image_width, image_height)  # Use the larger dimension

                    if max_dimension > 160:
                        self.resize_slider.setMaximum(160)
                        self.resize_slider.setValue(96)
                    else:
                        self.resize_slider.setMaximum(160)
                        self.resize_slider.setValue(max_dimension)
                        self.resize_slider_changed(max_dimension)

                    self.display_gif(file_path)

                else:
                    # For static images
                    img = img.convert("RGBA")  # Ensure RGBA, if needed
                    width, height = img.size
                    largest_dim = max(width, height)
                    smallest_dim = min(width, height)
                        
                    # -- 1) Check special cases ------------------------------------------
                    if width == height:
                        max_dim = 200
                    elif (width == 2 * height) or (height == 2 * width):
                        max_dim = 400
                    else:
                        scale_factor_a = min(400 / width, 200 / height)
                        scale_factor_b = min(200 / width, 400 / height)
                        best_scale_factor = max(scale_factor_a, scale_factor_b)
                        scaled_max_dimension = best_scale_factor * largest_dim
                        max_dim = int(round(scaled_max_dimension))
                        if max_dim > 400:
                            max_dim = 400
                        
                    self.resize_slider.setMaximum(max_dim)  # max_dim is already ≤ 400
                        
                    if largest_dim > max_dim:
                        self.resize_slider.setValue(128)
                    else:
                        self.resize_slider.setValue(largest_dim)
                        
                    self.resize_slider_changed(self.resize_slider.value())
                    self.image = ImageQt.ImageQt(img)
                    self.display_image()
                
            # (The rest of your UI settings for checkboxes, sliders, etc. remain unchanged.)
            for color_number in self.color_checkboxes.keys():
                self.color_checkboxes[color_number].setChecked(True)
                self.rgb_checkboxes[color_number].setChecked(False)
                self.rgb_checkboxes[color_number].setVisible(True)
                self.blank_checkboxes[color_number].setChecked(False)
                self.blank_checkboxes[color_number].setVisible(True)
                if color_number in self.boost_sliders:
                    self.boost_sliders[color_number].setValue(14)
                    self.boost_sliders[color_number].setVisible(True)
                if color_number in self.threshold_sliders:
                    self.threshold_sliders[color_number].setValue(20)
                    self.threshold_sliders[color_number].setVisible(True)
                if color_number in self.boost_labels:
                    self.boost_labels[color_number].setVisible(True)
                if color_number in self.threshold_labels:
                    self.threshold_labels[color_number].setVisible(True)

            self.preprocess_checkbox.setChecked(True)
            self.oncanvascheckbox.setChecked(False)
            self.ongrasscheckbox.setChecked(False)
            self.lab_color_checkbox.setChecked(True)

            if self.back_button:
                self.back_button.show()
                self.back_button.raise_()
            if self.refresh_button:
                self.refresh_button.show()
                self.refresh_button.raise_()
            if self.is_gif:
                if "Color Match" in [method["name"] for method in self.processing_methods]:
                    self.processing_combobox.setCurrentText("Color Match")
                    self.processing_method_changed("Color Match")
            else:
                if "Pattern Dither" in [method["name"] for method in self.processing_methods]:
                    self.processing_combobox.setCurrentText("Pattern Dither")
                    self.processing_method_changed("Pattern Dither")

            for color_number in self.boost_labels:
                self.boost_labels[color_number].setVisible(False)
                self.boost_sliders[color_number].setVisible(False)
                self.threshold_labels[color_number].setVisible(False)
                self.threshold_sliders[color_number].setVisible(False)

            self.brightness_slider.setValue(55)
            self.manual_change = False
            self.resize_slider_changed(self.resize_slider.value())

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")


    def display_gif(self, file_path, progress_callback=None, message_callback=None, error_callback=None):
        """
        Processes and displays an animated GIF or WebP with fixed dimensions (416x256).
        The GIF retains its aspect ratio, with one dimension reaching 416 or 256, and is aligned
        bottom-center in the larger frame. Downscaling uses bicubic; upscaling uses nearest neighbor.
        Frame delays are preserved to maintain the original animation speed.
        Only the first 500 frames are processed and displayed.
        
        Parameters:
            file_path (str or Path): Path to the input GIF or WebP file.
            progress_callback (callable, optional): Function to report progress, accepts (current, total).
            message_callback (callable, optional): Function to display messages, accepts (message).
            error_callback (callable, optional): Function to handle errors, accepts (error_message).
        """
        MAX_FRAMES = 500  # Maximum number of frames to process

        try:
            # Ensure the passed widget is a QLabel
            if not isinstance(self.image_label, QLabel):
                raise ValueError("The 'image_label' attribute must be an instance of QLabel.")

            # Define fixed dimensions
            frame_width, frame_height = 416, 256

            # Temporary file for the resized GIF/WebP
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webp")
            temp_path = temp_file.name
            temp_file.close()  # Close the file so PIL can write to it

            with Image.open(file_path) as img:
                # Determine aspect ratio and new dimensions
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height

                if aspect_ratio > 1:  # Wider than tall
                    new_width = frame_width
                    new_height = int(frame_width / aspect_ratio)
                    if new_height > frame_height:
                        new_height = frame_height
                        new_width = int(new_height * aspect_ratio)
                else:  # Taller than wide
                    new_height = frame_height
                    new_width = int(frame_height * aspect_ratio)
                    if new_width > frame_width:
                        new_width = frame_width
                        new_height = int(new_width / aspect_ratio)

                # Calculate offsets for bottom-center alignment
                x_offset = (frame_width - new_width) // 2
                y_offset = frame_height - new_height

                # Create a blank RGBA canvas for the fixed frame dimensions
                blank_frame = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))

                # Process frames with a limit of MAX_FRAMES
                frames = []
                delays = []
                frame_count = 0
                for frame in ImageSequence.Iterator(img):
                    if frame_count >= MAX_FRAMES:
                        if message_callback:
                            message_callback(f"Reached the maximum of {MAX_FRAMES} frames. Additional frames are ignored.")
                        break

                    frame = frame.convert("RGBA")  # Ensure consistent format

                    # Determine resampling method based on scaling direction
                    resample_method = (
                        Image.Resampling.BICUBIC if img_width > new_width or img_height > new_height else Image.Resampling.NEAREST
                    )

                    resized_frame = frame.resize((new_width, new_height), resample=resample_method)

                    # Paste resized frame onto the blank canvas
                    positioned_frame = blank_frame.copy()
                    positioned_frame.paste(resized_frame, (x_offset, y_offset), resized_frame)
                    frames.append(positioned_frame)

                    # Preserve frame delay (default to 100ms if not provided)
                    delay = frame.info.get("duration", 100)
                    delays.append(delay)

                    frame_count += 1

                    # Report progress if callback is provided
                    if progress_callback:
                        progress_callback(frame_count, MAX_FRAMES)

                if not frames:
                    if error_callback:
                        error_callback("No frames were processed to create the animation.")
                    return

                # Save the frames as a new WebP animation with preserved delays
                frames[0].save(
                    temp_path,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    loop=0,
                    duration=delays,
                    disposal=2  # Clear previous frames
                )

            # Load the resized GIF/WebP into QMovie
            movie = QMovie(temp_path)

            # Configure QLabel appearance and alignment
            self.image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            self.image_label.setStyleSheet("background-color: transparent; border: none;")

            # Set the QMovie to the QLabel and start the animation
            self.image_label.setMovie(movie)
            movie.start()

        except Exception as e:
            if error_callback:
                error_callback(f"An error occurred in display_gif: {e}")
            else:
                # If no error_callback is provided, you might want to log the error or handle it differently
                print(f"An error occurred in display_gif: {e}")



    def display_video(self, file_path, progress_callback=None, message_callback=None, error_callback=None):
        """
        Processes and displays a video file (MP4) by creating a preview animated GIF.
        The preview GIF is generated by sampling frames from the video, resizing and aligning
        them on a fixed canvas (416x256, bottom-center aligned), and then loading the GIF
        via QMovie for display.
        """
        try:
            # Fixed canvas dimensions (matching display_gif)
            frame_width, frame_height = 416, 256
            MAX_FRAMES = 100  # Maximum number of frames to include in the preview

            # Create a temporary file to save the preview GIF.
            import tempfile  # Ensure tempfile is imported
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            temp_path = temp_file.name
            temp_file.close()

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Calculate a skip factor so that we sample at most MAX_FRAMES frames.
            skip_factor = max(1, total_frames // MAX_FRAMES)

            frames = []
            delays = []
            frame_index = 0
            sampled = 0

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if frame_index % skip_factor == 0:
                    # Convert BGR to RGB and create a PIL image.
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb).convert("RGBA")

                    # Determine new dimensions and bottom-center alignment (stealing from display_gif)
                    img_width, img_height = pil_frame.size
                    aspect_ratio = img_width / img_height

                    if aspect_ratio > 1:  # Wider than tall
                        new_width = frame_width
                        new_height = int(frame_width / aspect_ratio)
                        if new_height > frame_height:
                            new_height = frame_height
                            new_width = int(frame_height * aspect_ratio)
                    else:  # Taller than wide
                        new_height = frame_height
                        new_width = int(frame_height * aspect_ratio)
                        if new_width > frame_width:
                            new_width = frame_width
                            new_height = int(frame_width / aspect_ratio)

                    # Calculate offsets for bottom-center alignment.
                    x_offset = (frame_width - new_width) // 2
                    y_offset = frame_height - new_height

                    # Create a blank RGBA canvas.
                    blank_frame = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))

                    # Resize the frame (using bicubic for downscaling).
                    resized_frame = pil_frame.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

                    # Paste the resized frame onto the blank canvas.
                    positioned_frame = blank_frame.copy()
                    positioned_frame.paste(resized_frame, (x_offset, y_offset), resized_frame)

                    frames.append(positioned_frame)
                    delays.append(100)  # Fixed delay (in ms) per frame

                    sampled += 1
                    if progress_callback:
                        progress_callback(sampled, MAX_FRAMES)
                    if sampled >= MAX_FRAMES:
                        break

                frame_index += 1

            cap.release()

            if not frames:
                raise ValueError("No frames were extracted from the video.")

            # Save the collected frames as an animated GIF.
            frames[0].save(
                temp_path,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=delays,
                disposal=2
            )

            # Load the preview GIF into a QMovie and display it.
            movie = QMovie(temp_path)
            self.image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
            self.image_label.setStyleSheet("background-color: transparent; border: none;")
            self.image_label.setMovie(movie)
            movie.start()

        except Exception as e:
            if error_callback:
                error_callback(f"An error occurred in display_video: {e}")
            else:
                print(f"An error occurred in display_video: {e}")


    def display_image(self):
        """
        Displays a static image resized to 420x420.
        Nearest-neighbor scaling is applied for upscaling, and bicubic scaling is used for downscaling.
        """
        if not self.image:
            QMessageBox.warning(self, "Error", "No image available to display.")
            return

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(self.image)

        # Determine scaling method
        if pixmap.width() > 416 or pixmap.height() > 256:
            # Downscaling: Use smooth transformation (bicubic)
            transformation_mode = Qt.SmoothTransformation
        else:
            # Upscaling: Use fast transformation (nearest-neighbor)
            transformation_mode = Qt.FastTransformation

        # Resize the image to 420x380
        resized_pixmap = pixmap.scaled(
            416, 256, Qt.KeepAspectRatio, transformation_mode
        )

        # Display the image in the QLabel
        self.image_label.setPixmap(resized_pixmap)
        self.image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        self.image_label.setStyleSheet("background-color: transparent; border: none;")  # Ensure no black border

            
    def processing_method_changed(self, method_name, strength=False, manual=True):
        """
        Updates the parameter input UI dynamically when the processing method is changed.
        Handles descriptions and ensures compatibility with the new decorator structure.
        """
        # Clear existing parameter widgets
        while self.method_options_layout.rowCount() > 0:
            self.method_options_layout.removeRow(0)
        self.parameter_widgets.clear()

        # Retrieve the processing function
        processing_function = processing_method_registry.get(method_name)
        if not processing_function:
            return

        # Retrieve and display the description, if any, in self.blank
        method_description = getattr(processing_function, "description", "")
        if method_description:
            self.blank.setPlainText(method_description)

        # Retrieve default parameters, etc. if needed
        default_params = getattr(processing_function, "default_params", {})

        if manual:
            self.manual_change = True

        # Dynamically create input widgets for parameters
        for param_name, default_value in default_params.items():
            label = QLabel(f"{method_name} {param_name.capitalize()}:")
            
            # Special handling for 'Clusters'
            if param_name == 'Clusters':
                # Create slider
                slider = QSlider(Qt.Horizontal)
                slider.setRange(2, 16)  # Range for clusters
                slider.setValue(9) 
                slider.setTickPosition(QSlider.TicksBelow)
                slider.setTickInterval(1)  # Step size for clusters
                slider.valueChanged.connect(self.parameter_value_changed)

                # Display value dynamically
                value_label = QLabel(str(slider.value()))
                value_label.setAlignment(Qt.AlignCenter)
                value_label.setFixedWidth(60)

                # Combine slider and value label into a horizontal layout
                slider_layout = QHBoxLayout()
                slider_layout.addWidget(slider)
                slider_layout.addWidget(value_label)

                # Update value label when the slider changes
                def update_value_label(value):
                    if value < 16:
                        value_label.setText(str(value))
                    else:
                        value_label.setText("Lots")
                slider.valueChanged.connect(update_value_label)
                update_value_label(9)
                # Add slider layout to the form
                self.method_options_layout.addRow(label, slider_layout)

                # Save the slider to parameter widgets
                self.parameter_widgets[param_name] = slider


            elif isinstance(default_value, (float, int)):
                slider = QSlider(Qt.Horizontal)
                if isinstance(default_value, float):
                    slider.setRange(0, 100)
                    slider.setValue(int(default_value * 100))
                    slider.setTickInterval(10)

                else:
                    slider.setRange(1, 100)
                    slider.setValue(default_value)
                    slider.setTickInterval(10)
                slider.setTickPosition(QSlider.TicksBelow)
                slider.valueChanged.connect(self.parameter_value_changed)
                self.method_options_layout.addRow(label, slider)
                self.parameter_widgets[param_name] = slider
            elif isinstance(default_value, bool):
                # Use a checkbox for boolean values
                checkbox = QCheckBox()
                checkbox.setChecked(default_value)
                self.method_options_layout.addRow(label, checkbox)
                self.parameter_widgets[param_name] = checkbox
            elif isinstance(default_value, str):
                if param_name in ["line_color", "line_style"]:
                    # Use a combo box for predefined options
                    combo_box = QComboBox()
                    if param_name == "line_color":
                        combo_box.addItems(["auto", "black", "white"])
                    elif param_name == "line_style":
                        combo_box.addItems(["black_on_white", "white_on_black"])
                    combo_box.setCurrentText(default_value)
                    self.method_options_layout.addRow(label, combo_box)
                    self.parameter_widgets[param_name] = combo_box
                else:
                    line_edit = QLineEdit(default_value)
                    self.method_options_layout.addRow(label, line_edit)
                    self.parameter_widgets[param_name] = line_edit
            else:
                # Fallback for unsupported types
                line_edit = QLineEdit(str(default_value))
                self.method_options_layout.addRow(label, line_edit)
                self.parameter_widgets[param_name] = line_edit

                

    def parameter_value_changed(self, value):
        # Update any dependent UI elements if necessary
        pass


    def add_chalks_colors(self, input_array):
        input_dict = {
            7: '#a3b2d2',
            8: '#d6cec2',
            9: '#bfded8',
            10: '#a9c484',
            11: '#5d937b',
            12: '#a2a6a9',
            13: '#777f8f',
            14: '#eab281',
            15: '#ea7286',
            16: '#f4a4bf',
            17: '#a07ca7',
            18: '#bf796d',
            19: '#f5d1b6',
            20: '#e3e19f',
            21: '#ffdf00',
            22: '#ffbf00',
            23: '#c4b454',
            24: '#f5deb3',
            25: '#f4c430',
            26: '#00ffff',
            27: '#89cff0',
            28: '#4d4dff',
            29: '#00008b',
            30: '#4169e1',
            31: '#006742',
            32: '#4cbb17',
            33: '#2e6f40',
            34: '#2e8b57',
            35: '#c0c0c0',
            36: '#818589',
            37: '#899499',
            38: '#708090',
            39: '#ffa500',
            40: '#ff8c00',
            41: '#d7942d',
            42: '#ff5f1f',
            43: '#cc7722',
            44: '#ff69b4',
            45: '#ff10f0',
            46: '#aa336a',
            47: '#f4b4c4',
            48: '#953553',
            49: '#d8bfd8',
            50: '#7f00ff',
            51: '#800080',
            52: '#ff2400',
            53: '#ff4433',
            54: '#a52a2a',
            55: '#913831',
            56: '#ff0000',
            57: '#3b2219',
            58: '#a16e4b',
            59: '#d4aa78',
            60: '#e6bc98',
            61: '#ffe7d1'
        }

        # Add items from the dictionary to the input array
        for number, hex_value in input_dict.items():
            input_array.append({
                'number': number,
                'hex': hex_value.lstrip('#'),  # Remove '#' from hex
                'boost': 0,  # Set boost to 0
                'threshold': 0  # Set threshold to 0
            })

        return input_array 
        
    def process_image(self):
        if not hasattr(self, 'result_widget'):
            self.setup_result_menu()
            
        if not self.image_path:
            QMessageBox.warning(self, "Error", "No image selected.")
            return

        self.status_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.action_layout.setCurrentIndex(1)  # Show status layout
        self.status_label.setText("Starting processing...")
        # Hide back button
        if self.back_button:
            self.back_button.hide()
        if self.refresh_button:
            self.refresh_button.hide()
        self.canpaste = False
        
        # Collect parameters
        preprocess_flag = self.preprocess_checkbox.isChecked()
        bg_removal_flag = self.bg_removal_checkbox.isChecked()
        custom_filter_flag = self.lab_color_checkbox.isChecked()
        resize_dim = self.resize_slider.value()

        # Build color_key_array based on user selections
        color_key_array = []

        # Collect selected default colors
        for color in self.default_color_key_array:
            color_number = color['number']
            enable_checkbox = self.color_checkboxes[color_number]
            if enable_checkbox.isChecked():
                # Add a copy of the color to the array
                color_key_array.append(color.copy())

        # Determine which color (if any) is marked as RGB
        rgb_color_number = None
        for color_number, rgb_checkbox in self.rgb_checkboxes.items():
            if rgb_checkbox.isChecked():
                rgb_color_number = color_number
                break

        # If an RGB color is selected, replace its number with 5
        if rgb_color_number is not None:
            for color in color_key_array:
                if color['number'] == rgb_color_number:
                    color['number'] = 5
                    break

        # Determine which color (if any) is marked as Blank
        blank_color_num = None
        for color_number, blank_checkbox in self.blank_checkboxes.items():
            if blank_checkbox.isChecked():
                blank_color_num = color_number
                break

        # If a Blank color is selected, replace its number with -1
        if blank_color_num is not None:
            for color in color_key_array:
                if color['number'] == blank_color_num:
                    color['number'] = -1
                    break

        # Update each color in the array with its corresponding slider values
        for color in color_key_array:
            color_number = color['number']
            # Skip RGB (5) and Blank (-1) as they don't need these values
            if color_number in [5, -1]:
                continue
            # If you need to add logic for retrieving slider values, do it here

        # If background removal is checked, add the chalks colors
        if self.bg_removal_checkbox.isChecked():
            print("Adding chalks colors...")
            color_key_array = self.add_chalks_colors(color_key_array)

        # ----- NEW LOGIC HERE -----
        # Check if none of the blank checkboxes is checked:
        #   (i.e. blank_color_num is None)
        if blank_color_num is None:
            # If self.ongrasscheckbox is checked, add a grass color
            if self.ongrasscheckbox.isChecked():
                color_key_array.append({
                    'number': -1,
                    'hex': '77790e',  # Grass color
                    'boost': 0,
                    'threshold': 0
                })

            # If self.oncanvascheckbox is checked, add a canvas color
            if self.oncanvascheckbox.isChecked():
                color_key_array.append({
                    'number': -1,
                    'hex': 'c48e4c',  # Canvas color
                    'boost': 0,
                    'threshold': 0
                })
        # ----- END NEW LOGIC -----

        process_mode = self.processing_combobox.currentText()

        # Collect parameters from parameter widgets
        process_params = {}
        processing_function = processing_method_registry.get(process_mode)
        if not processing_function:
            QMessageBox.warning(self, "Error", "Invalid processing method selected.")
            return
        default_params = getattr(processing_function, 'default_params', {})

        for param_name, default_value in default_params.items():
            widget = self.parameter_widgets.get(param_name)
            if isinstance(widget, QSlider):
                if isinstance(default_value, float):
                    value = widget.value() / 100.0
                else:
                    value = widget.value()
                process_params[param_name] = value
            elif isinstance(widget, QCheckBox):
                process_params[param_name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                process_params[param_name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                if isinstance(default_value, int):
                    try:
                        process_params[param_name] = int(text)
                    except ValueError:
                        process_params[param_name] = default_value
                elif isinstance(default_value, float):
                    try:
                        process_params[param_name] = float(text)
                    except ValueError:
                        process_params[param_name] = default_value
                else:
                    process_params[param_name] = text
            else:
                process_params[param_name] = default_value

        brightness = self.brightness_slider.value() / 100

        # Prepare parameters for image processing
        params = {
            'image_path': self.image_path,
            'remove_bg': bg_removal_flag,
            'preprocess_flag': preprocess_flag,
            'use_lab': custom_filter_flag,
            'brightness': brightness,
            'resize_dim': resize_dim,
            'color_key_array': color_key_array,
            'process_mode': process_mode,
            'process_params': process_params
        }

        # Switch to status and progress view
        self.status_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.action_layout.setCurrentIndex(1) 
        self.status_label.setText("Starting processing...")
        self.progress_bar.setValue(0)

        # Start image processing in a separate thread
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.update_progress)
        self.signals.message.connect(self.update_status)
        self.signals.error.connect(self.show_error)
        self.processing_thread = ImageProcessingThread(params, self.signals)
        self.processing_thread.start()
        self.monitor_thread()

    def monitor_thread(self):
        """
        Monitors the processing thread and updates the UI upon completion.
        """
        if self.processing_thread.is_alive():
            QTimer.singleShot(100, self.monitor_thread)
            return
        
        self.reset_movie()

        self.progress_bar.setValue(100)
        self.status_label.setText("Processing complete!")

        # Paths for PNG and GIF previews
        preview_png_path = exe_path_fs('game_data/stamp_preview/preview.png')
        preview_gif_path = exe_path_fs('game_data/stamp_preview/preview.gif')

        try:
            if self.is_gif:
                print(preview_gif_path)
                self.process_and_display_gif(preview_gif_path)
            elif preview_png_path.exists():
                self.handle_png(preview_png_path)
            else:
                raise FileNotFoundError("Processed image not found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to display result: {str(e)}")
            self.reset_ui_after_failure()
            return


        self.stacked_widget.setCurrentWidget(self.result_widget)
        self.canpaste = True
        # Re-enable the Process button if needed
        self.reset_ui_after_processing()

    def reset_ui_after_failure(self):
        """
        Resets the UI after a processing failure.
        """
        self.action_layout.setCurrentIndex(0)  # Switch back to process button
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Ready")

        # Re-enable the Process button
        self.process_button.setVisible(True)
        self.process_button.setEnabled(True)

        # Show back and refresh buttons
        if self.back_button:
            self.back_button.show()
        if self.refresh_button:
            self.refresh_button.show()
            
    def process_and_display_gif(self, gif_path):
        """
        Processes an input GIF, resizes frames with hard pixel edges (NEAREST),
        and displays the animation on a QLabel using QTimer for manual frame handling.
        """
        try:
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    raise ValueError("Input file is not an animated GIF.")

                frames = []
                durations = []

                # QLabel dimensions
                max_width, max_height = 600, 530

                # Process each frame
                for frame in ImageSequence.Iterator(img):
                    frame = frame.convert("RGBA")  # Preserve transparency
                    scale_factor = min(max_width / frame.width, max_height / frame.height)
                    new_width = int(frame.width * scale_factor)
                    new_height = int(frame.height * scale_factor)

                    # Resize the frame while preserving aspect ratio
                    resized_frame = frame.resize((new_width, new_height), Image.Resampling.NEAREST)

                    # Create a transparent canvas
                    canvas = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))
                    # Center the resized frame on the canvas
                    offset_x = (max_width - new_width) // 2
                    offset_y = (max_height - new_height) // 2
                    canvas.paste(resized_frame, (offset_x, offset_y), resized_frame)

                    # Convert to QPixmap
                    data = canvas.tobytes("raw", "RGBA")
                    qimage = QImage(data, canvas.width, canvas.height, QImage.Format_RGBA8888)
                    pixmap = QPixmap.fromImage(qimage)

                    # Store the QPixmap and duration
                    frames.append(pixmap)
                    durations.append(frame.info.get("duration", 100))  # Default to 100ms if no duration

                if frames:
                    # Set up frame animation with QTimer
                    self.current_frame = 0
                    self.timer = QTimer(self)
                    self.timer.timeout.connect(lambda: self.update_gif_frame2(frames, durations))
                    self.timer.start(durations[0])  # Start with the first frame's duration
                    self.gif_frames = frames
                    self.gif_durations = durations

                    # Display the first frame to initialize the QLabel
                    self.result_image_label.setPixmap(frames[0])
                    self.result_image_label.setAlignment(Qt.AlignCenter)

        except Exception as e:
            print(f"Error processing or displaying GIF: {e}")
            QMessageBox.warning(self, "Error", f"Failed to process and display GIF: {e}")

    def update_gif_frame2(self, frames, durations):
        """
        Updates the QLabel with the next frame in the animation sequence.
        """
        # Update QLabel with the current frame
        self.result_image_label.setPixmap(frames[self.current_frame])

        # Increment the frame index
        self.current_frame = (self.current_frame + 1) % len(frames)

        # Update timer interval for the next frame
        next_duration = durations[self.current_frame]
        self.timer.start(next_duration)



    def handle_png(self, png_path):
        """
        Resizes and displays a PNG in QLabel, preserving aspect ratio without blank space.
        """
        if not os.path.exists(png_path):
            raise FileNotFoundError(f"PNG file not found: {png_path}")

        try:
            with Image.open(png_path).convert("RGBA") as img:


                # QLabel dimensions
                max_width, max_height = 600, 530
                scale_factor = min(max_width / img.width, max_height / img.height)
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)

                # Resize with high-quality scaling
                resized_img = img.resize((new_width, new_height), Image.Resampling.NEAREST)

                # Convert to QPixmap and display
                qimage = ImageQt.ImageQt(resized_img)
                pixmap = QPixmap.fromImage(qimage)
                self.result_image_label.setPixmap(pixmap)
                self.result_image_label.setAlignment(Qt.AlignCenter)
                self.result_image_label.setStyleSheet("background-color: transparent; border: none;")

        except Exception as e:
            print(f"Error displaying PNG: {e}")
            QMessageBox.warning(self, "Error", f"Failed to display PNG: {e}")


    def reset_ui_after_processing(self):
        """
        Resets the UI to the initial state after processing.
        Ensures all animations and temporary data are cleared.
        """
        self.action_layout.setCurrentIndex(0)
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Ready")


        if self.back_button:
            self.back_button.show()
        if self.refresh_button:
            self.refresh_button.show()

            
    def update_progress(self, progress):
        QTimer.singleShot(0, lambda: self.progress_bar.setValue(progress))

    def update_status(self, message):
        QTimer.singleShot(0, lambda: self.status_label.setText(message))

    def show_error(self, message):
        QMessageBox.warning(self, "Error", message)
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error occurred during processing.")
    

    def reset_to_initial_state(self):
        """
        Resets the application state when the user navigates back to the initial menu.
        """
        # Reset image and GIF-related states
        self.image_path = None
        self.image = None
        self.is_gif = False
        self.movie = None

        # Clear image label
        self.image_label.clear()
        self.image_label.setText("Oopsies")
        self.image_label.setStyleSheet("")

        # Reset resize slider to default
        self.resize_slider.setValue(128)
        self.resize_slider.setMaximum(400)

        # Reset color options and related UI elements
        self.reset_color_options()

        # Reset processing flags
        self.preprocess_checkbox.setChecked(True)
        self.lab_color_checkbox.setChecked(True)
        self.oncanvascheckbox.setChecked(False)
        self.ongrasscheckbox.setChecked(False)

        # Ensure all boost and threshold elements are hidden
        for color_number in self.boost_labels:
            self.boost_labels[color_number].setVisible(False)
            self.boost_sliders[color_number].setVisible(False)
            self.threshold_labels[color_number].setVisible(False)
            self.threshold_sliders[color_number].setVisible(False)

        # Show back and refresh buttons
        if self.back_button:
            self.back_button.show()
        if self.refresh_button:
            self.refresh_button.show()

        # Reset status and progress indicators
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Status: Ready")
        self.status_label.setVisible(False)

        # Re-enable the Process button
        self.process_button.setVisible(True)
        self.process_button.setEnabled(True)

        # Reset any additional flags
        self.canpaste = True
        self.delete_mode = False
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()
            self.timer = None

        # Clear the QLabel
        if hasattr(self, 'result_image_label') and self.result_image_label is not None:
            self.result_image_label.clear()

        # Reset the animation-related attributes
        self.gif_frames = None
        self.gif_durations = None
        self.current_frame = None


    def go_to_initial_menu(self, usepreview = False):
        """
        Handles navigation back to the initial menu and resets the application state.
        """
        if usepreview:
            self.display_new_stamp()
        else: 
            self.background_label.setPixmap(self.load_and_display_random_image())

        if not hasattr(self, 'secondary_widget'):
            self.setup_secondary_menu()
        
        self.reset_to_initial_state()
        self.reset_color_options()
        self.stacked_widget.setCurrentIndex(0)  # Switch to the initial menu
        self.canpaste = True
    
    # Define callback function for feedback
    def callback(self, message, center=False):
        """Handles user feedback."""
        print(message)
        if not self.last_message_displayed:
            self.show_floating_message(message, center)
            self.last_message_displayed = message

    def randomize_saved_stamps(self):
        """
        Randomizes the order of entries in the saved_stamps.json file.
        """
        # Use the new AppData directory
        appdata_dir = get_appdata_dir()
        saved_stamps_json = appdata_dir / "saved_stamps.json"

        # Check if JSON file exists
        if not saved_stamps_json.exists():
            return

        # Load JSON entries
        try:
            with open(saved_stamps_json, 'r') as f:
                saved_stamps = json.load(f)
        except Exception as e:
            self.callback("Error loading JSON file")
            return

        # Randomize entries
        randomized_entries = list(saved_stamps.items())
        random.shuffle(randomized_entries)

        # Convert back to dictionary and save
        randomized_dict = dict(randomized_entries)
        with open(saved_stamps_json, 'w') as f:
            json.dump(randomized_dict, f, indent=4)

        self.show_floating_message("Randomized!", True)
        self.repopulate_grid()


# Function to compute hash for a file
    def compute_hash(self, file_path):
        import hashlib
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.callback(f"Error: {e}")
            return None

    # Save Current Function
    def save_current(self, center=False):
        # Reset the last message at the beginning of the operation
        self.last_message_displayed = None

        # Define paths
        appdata_dir = get_appdata_dir()
        current_stamp_dir = exe_path_fs("game_data/current_stamp_data")
        preview_dir = exe_path_fs("game_data/stamp_preview")
        saved_stamps_json = appdata_dir / "saved_stamps.json"
        saved_stamps_dir = appdata_dir / "saved_stamps"

        stamp_path = current_stamp_dir / "stamp.txt"
        frames_path = current_stamp_dir / "frames.txt"


        # Step 1: Read and parse stamp.txt
        if not stamp_path.exists():
            self.callback("Error: stamp.txt not found")
            return
        try:
            with open(stamp_path, 'r') as f:
                first_line = f.readline().strip()
            parts = first_line.split(',')
            is_gif_flag = parts[2]  # Extract the flag at the third position
        except Exception as e:
            self.callback(f"Error: {e}")
            return

        if is_gif_flag not in ["img", "gif"]:
            self.callback("Error: Invalid flag in stamp.txt")
            return

        # Step 2: Compute hash for stamp.txt
        stamp_hash = self.compute_hash(stamp_path)
        if not stamp_hash:
            return

        # Step 3: Load or initialize saved_stamps.json
        appdata_dir.mkdir(parents=True, exist_ok=True)  # Ensure GUI directory exists
        if not saved_stamps_json.exists():
            initialize_saved()
            saved_stamps = {}
        else:
            try:
                with open(saved_stamps_json, 'r') as f:
                    saved_stamps = json.load(f)
            except Exception:
                initialize_saved()
                saved_stamps = {}

        # Step 4: Check if hash already exists
        if stamp_hash in saved_stamps:
            self.callback("already saved dummy", center)
            return
        
        # Step 5: Check for preview file
        preview_path = preview_dir / ("preview.png" if is_gif_flag == "img" else "preview.gif")
        if not preview_path.exists():
            self.callback(f"Error: {preview_path.name} not found")
            return


        # Step 6: Create folder for the new stamp
        stamp_folder = saved_stamps_dir / stamp_hash
        stamp_folder.mkdir(parents=True, exist_ok=True)

        # Step 7: Copy relevant files to the new folder
        try:
            (stamp_folder / "stamp.txt").write_bytes(stamp_path.read_bytes())
            if is_gif_flag == "gif":
                (stamp_folder / "frames.txt").write_bytes(frames_path.read_bytes())
        except Exception as e:
            self.callback(f"Error: {e}")
            return

        # Step 8: Call the `get_preview` function
        try:
            self.get_preview(preview_path, stamp_folder)
        except Exception as e:
            self.callback(f"Error: {e}")
            return

        # Step 9: Update saved_stamps.json
        saved_stamps[stamp_hash] = {"is_gif": (is_gif_flag == "gif")}
        with open(saved_stamps_json, 'w') as f:
            json.dump(saved_stamps, f, indent=4)

        self.callback("saved gif" if is_gif_flag == "gif" else "saved image", center)
        if center:
            self.repopulate_grid()

    def get_preview(self, preview_path, target_folder):
        """
        Process preview images (PNG or GIF), resize them to fit within 128x128 without warping,
        align them to the center, and save as WebP format.
        """

        # 1) If the path does not exist or is a directory, try to find preview.gif or preview.png.
        if (not preview_path.exists()) or preview_path.is_dir():
            # Determine which folder to look into
            folder_to_check = preview_path if preview_path.is_dir() else preview_path.parent
            
            # Possible preview files in that folder
            gif_file = folder_to_check / "preview.gif"
            png_file = folder_to_check / "preview.png"
            webp_file = folder_to_check / "preview.webp"
            
            # Priority: GIF -> PNG -> (if WebP exists, do nothing) -> else error
            if gif_file.exists():
                preview_path = gif_file
            elif png_file.exists():
                preview_path = png_file
            else:
                # If there's already a preview.webp, we won't overwrite/change it.
                if webp_file.exists():
                    # Since you said it already works "perfectly" for WebP,
                    # we just return here or do nothing further.
                    return
                else:
                    raise FileNotFoundError(
                        "No 'preview.gif', 'preview.png', or existing 'preview.webp' found."
                    )

        # 2) From here down, the code is essentially unchanged—just your original logic for PNG/GIF.
        target_file = target_folder / "preview.webp"
        output_size = (128, 128)

        def resize_and_pad_image(img):
            # Calculate aspect ratio to fit within 128x128
            img_ratio = img.width / img.height
            box_ratio = output_size[0] / output_size[1]

            if img_ratio > box_ratio:
                # Image is wider, fit by width
                new_width = output_size[0]
                new_height = int(output_size[0] / img_ratio)
            else:
                # Image is taller or square, fit by height
                new_height = output_size[1]
                new_width = int(output_size[1] * img_ratio)

            # Resize image while maintaining aspect ratio
            img = img.resize((new_width, new_height), Image.NEAREST)

            # Create a transparent canvas
            canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))

            # Calculate position to center the image
            offset_x = (output_size[0] - new_width) // 2
            offset_y = (output_size[1] - new_height) // 2

            # Paste resized image onto the canvas
            canvas.paste(img, (offset_x, offset_y), img)
            return canvas

        # Handle PNG
        if preview_path.suffix.lower() == ".png":
            img = Image.open(preview_path).convert("RGBA")
            resized_img = resize_and_pad_image(img)
            resized_img.save(target_file, format="WEBP", lossless=True)

        # Handle GIF
        elif preview_path.suffix.lower() == ".gif":
            original_gif = Image.open(preview_path)
            frames = []
            durations = []

            # Process each frame
            for frame in ImageSequence.Iterator(original_gif):
                durations.append(frame.info.get("duration", 100))  # Default to 100ms if no duration info
                frames.append(resize_and_pad_image(frame.convert("RGBA")))

            # Save the resized GIF with the original durations
            frames[0].save(
                target_file,
                save_all=True,
                append_images=frames[1:],
                loop=original_gif.info.get("loop", 0),
                duration=durations,
                format="WEBP",
                lossless=True,
            )

        # If it's something else, raise an error (assuming WebP is handled elsewhere just fine).
        else:
            raise FileNotFoundError("Invalid preview format or no preview file found.")

def initialize_saved():
    """
    Initialize the saved stamps directory by cloning data from
    `saved_stamp_initial` into the AppData directory.
    If the directory already exists, it is cleared before cloning.
    """
    appdata_dir = get_appdata_dir()
    
    # Paths for AppData directories
    saved_stamps_dir = appdata_dir / "saved_stamps"
    saved_stamps_json = appdata_dir / "saved_stamps.json"

    # Clear the existing AppData directory
    if saved_stamps_dir.exists():
        shutil.rmtree(saved_stamps_dir)
    if saved_stamps_json.exists():
        saved_stamps_json.unlink()

    # Path to the initial data directory
    initial_dir = exe_path_fs("saved_stamp_initial")

    # Copy contents of the initial directory to the AppData directory
    for item in initial_dir.iterdir():
        if item.is_dir():
            # Copy the saved_stamps directory
            shutil.copytree(item, appdata_dir / item.name)
        elif item.is_file():
            # Copy the saved_stamps.json file
            shutil.copy(item, appdata_dir / item.name)

    print(f"Initialized saved stamps directory in: {appdata_dir}")


def cleanup_saved_stamps():
    """
    Validate and clean up the saved stamps directory and associated JSON file.
    Calls `initialize_saved` if the AppData directory or the saved stamps directory is missing.
    """
    appdata_dir = get_appdata_dir()
    saved_stamps_dir = appdata_dir / "saved_stamps"
    saved_stamps_json = appdata_dir / "saved_stamps.json"

    validated_entries = []     # List to track validated files
    reconstructed_entries = [] # Entries reconstructed from directory
    removed_folders = []       # List to track removed folders

    # If the saved stamps directory is missing, initialize it
    if not saved_stamps_dir.exists():
        print("Saved stamps directory not found. Initializing...")
        initialize_saved()
        return

    # Validate the .json file
    saved_stamps = {}
    json_valid = False  # Track if the JSON is valid
    if saved_stamps_json.exists():
        try:
            with open(saved_stamps_json, 'r') as f:
                saved_stamps = json.load(f)
                if not isinstance(saved_stamps, dict):
                    raise ValueError("JSON is not a dictionary.")
            json_valid = True
        except Exception as e:
            print(f"Corrupted JSON file detected: {e}. Attempting reconstruction.")
    else:
        print("JSON file not found. Attempting reconstruction.")

    # Collect folders in the saved_stamps directory
    actual_folders = {
        folder.name
        for folder in saved_stamps_dir.iterdir()
        if folder.is_dir()
    }

    # Define acceptable preview filenames
    preview_filenames = ["preview.webp", "preview.gif", "preview.png"]

    if not json_valid:
        # JSON is missing or invalid: Reconstruct it from the existing folders
        print("Reconstructing JSON entries...")
        for folder_name in actual_folders:
            folder_path = saved_stamps_dir / folder_name
            stamp_txt_path = folder_path / "stamp.txt"

            # Check if any of the acceptable previews exist
            has_preview = any((folder_path / fname).exists() for fname in preview_filenames)

            if stamp_txt_path.exists() and has_preview:
                # Check for "frames.txt" to determine if it is a GIF
                is_gif = (folder_path / "frames.txt").exists()
                saved_stamps[folder_name] = {"is_gif": is_gif}
                reconstructed_entries.append(folder_name)
            else:
                # Remove invalid folders
                removed_folders.append(folder_name)
                for file in folder_path.iterdir():
                    file.unlink()  # Remove files in the folder
                folder_path.rmdir()  # Remove the folder itself

        # Save the reconstructed JSON
        with open(saved_stamps_json, 'w') as f:
            json.dump(saved_stamps, f, indent=4)
        print("Reconstructed JSON saved.")
    else:
        # JSON is valid: Delete all folders not listed in the JSON
        print("Deleting folders not listed in JSON...")
        valid_hashes = set(saved_stamps.keys())
        for folder_name in actual_folders - valid_hashes:
            folder_path = saved_stamps_dir / folder_name
            removed_folders.append(folder_name)
            for file in folder_path.iterdir():
                file.unlink()  # Remove files in the folder
            folder_path.rmdir()  # Remove the folder itself

        # Validate and clean folders listed in JSON
        for folder_name in valid_hashes:
            folder_path = saved_stamps_dir / folder_name
            if not folder_path.exists():
                continue  # Skip non-existent folders

            stamp_txt_path = folder_path / "stamp.txt"
            # Check if any acceptable preview exists
            has_preview = any((folder_path / fname).exists() for fname in preview_filenames)

            # If the required files exist, mark as validated; otherwise remove
            if stamp_txt_path.exists() and has_preview:
                validated_entries.append(folder_name)
            else:
                # Remove invalid folders
                removed_folders.append(folder_name)
                for file in folder_path.iterdir():
                    file.unlink()  # Remove files in the folder
                folder_path.rmdir()  # Remove the folder itself
                # Remove from JSON
                saved_stamps.pop(folder_name, None)

        # Save the updated JSON
        with open(saved_stamps_json, 'w') as f:
            json.dump(saved_stamps, f, indent=4)

    # Print results
    print("\nValidated Entries:")
    print(validated_entries if validated_entries else "No entries validated.")

    print("\nReconstructed Entries:")
    print(reconstructed_entries if reconstructed_entries else "No entries reconstructed.")

    print("\nRemoved Folders:")
    print(removed_folders if removed_folders else "No folders removed.")
    
def create_default_config():
    """
    Create a default JSON configuration file at the given path.
    """
    config_path = get_config_path()
    
    default_config_data = {
        "open_menu": 16777247, 
        "spawn_stamp": 61, 
        "ctrl_z": 90, 
        "toggle_playback": 45, 
        "gif_ready": True, 
        "chalks": False,
        "host": False,
        "locked_canvas": [],
        "walky_talky_webfish": "nothing new!", 
        "walky_talky_menu": "nothing new!"
    }

    # Write the default configuration to the file
    try:
        with open(config_path, 'w') as file:
            json.dump(default_config_data, file, indent=4)
        print(f"Default configuration file created at {config_path}")
    except Exception as e:
        print(f"Failed to create config file: {e}")
        sys.exit(1)

def set_gif_ready(value):
    """
    Set the "gif_ready" field in the configuration file to the specified value (True or False).
    """
    config_path = get_config_path()
    
    try:
        # Read the existing configuration
        with open(config_path, 'r') as file:
            config_data = json.load(file)
        
        # Modify the "gif_ready" field
        config_data["gif_ready"] = value
        
        # Write the updated configuration back to the file
        with open(config_path, 'w') as file:
            json.dump(config_data, file, indent=4)
        
        print(f"'gif_ready' set to {value} in configuration file.")
    except Exception as e:
        print(f"Failed to update 'gif_ready': {e}")
        sys.exit(1)

def set_gif_ready_true():
    """Set 'gif_ready' to True."""
    set_gif_ready(True)

def set_gif_ready_false():
    """Set 'gif_ready' to False."""
    set_gif_ready(False)


def ipc_server():
    """
    IPC server that listens for incoming messages to bring the main process's window to front.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((IPC_HOST, IPC_PORT))
        except OSError as e:
            print(f"Failed to bind IPC server on {IPC_HOST}:{IPC_PORT}: {e}")
            sys.exit(1)
        s.listen()
        print(f"IPC Server listening on {IPC_HOST}:{IPC_PORT}")
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    message = data.decode().strip()
                    print(f"Received message: '{message}' from {addr}")
                    if message == 'BRING_TO_FRONT':
                        print("Request to bring window to front received.")
                        window.bringfront.emit()
                    elif message == 'EXIT':
                        print("Exit command received. Shutting down IPC server.")
                        sys.exit(0)
                    else:
                        print(f"Unknown message received: '{message}'")
            except Exception as e:
                print(f"Error in IPC server: {e}")


def startup():
    global app_lock

    if LOCK_FILE.is_dir():
        print(f"Lock file path {LOCK_FILE} is a directory. Please delete it and try again.")
        sys.exit(1)

    app_lock = FileLock(str(LOCK_FILE))
    try:
        app_lock.acquire(timeout=0)
        print("No existing instance detected. Running as the main instance.")

    except Timeout:
        print("Another instance is already running. Checking if it's still active.")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((IPC_HOST, IPC_PORT))
                s.sendall(b'BRING_TO_FRONT')
            print("Request sent to bring the existing instance to front. Exiting.")
            sys.exit(0)
        except FileNotFoundError:
            print(f"Lock file {LOCK_FILE} does not exist, but could not acquire lock.")
            sys.exit(1)
        except ValueError:
            print(f"Invalid PID found in lock file {LOCK_FILE}. Removing and retrying.")
            LOCK_FILE.unlink()
            try:
                app_lock.acquire(timeout=0)
                print("Acquired lock after removing invalid lock file. Running as the main instance.")
                with open(LOCK_FILE, 'w') as f:
                    f.write(str(os.getpid()))
            except Exception as e:
                print(f"Failed to acquire lock after removing invalid lock file: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"An error occurred while handling the lock: {e}")
            sys.exit(1)

    server_thread = threading.Thread(target=ipc_server, daemon=True)
    server_thread.start()

    config_path = get_config_path()
    
    appdata_dir = get_appdata_dir()
    saved_stamps_json = appdata_dir / "saved_stamps.json"

    appdata_dir.mkdir(parents=True, exist_ok=True)
    if not saved_stamps_json.exists():
        initialize_saved()

    # Check if the config file exists, create it if it doesn't
    if not config_path.exists():
        print("Config file not found. Creating default configuration...")
        create_default_config()

    load_config()



def load_config():
    """
    Load the configuration file and set the global variable 'has_chalks'
    based on the value of 'chalks' in the JSON.
    """
    global has_chalks
    config_path = get_config_path()
    
    try:
        with open(config_path, 'r') as file:
            config_data = json.load(file)
        
        # Retrieve the value of 'chalks' from the config
        has_chalks = config_data.get('chalks', False)
        
        print(f"'has_chalks' set to {has_chalks}")
    
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Configuration file at {config_path} is not a valid JSON.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)

if __name__ == '__main__':
    startup()

    app = QApplication(sys.argv)

    if sys.platform.startswith('win'):
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"ImageProcessingGUI")

    # Define the icon path and apply the icon
    icon_path = exe_path_str("imagePawcessor/icon.png")

    app_icon = None
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)  # QIcon is safe here after QApplication is created
        app.setWindowIcon(app_icon)
    else:
        print("Warning: icon.png not found in directory.")

    window = MainWindow()
    if app_icon:
        window.setWindowIcon(app_icon)  # Ensure the window gets the icon
    window.show()

    # Start the event loop
    sys.exit(app.exec())
