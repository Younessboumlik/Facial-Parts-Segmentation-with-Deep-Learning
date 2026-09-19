"""
Facial Parts Segmentation - Streamlit demo
U-Net / PSPNet / SegNet trained on the LaPa dataset (11 classes, 256x256).

Model weights are pulled at runtime from the Hugging Face Hub instead of being
committed to git (the three .keras files weigh ~1 GB in total).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------- configuration

# Overridable via .streamlit/secrets.toml (HF_REPO_ID) or the HF_REPO_ID env var.
DEFAULT_REPO_ID = "YounessBoumlik/face-segmentation-models"

IMAGE_H, IMAGE_W = 256, 256
NUM_CLASSES = 11

# Matches the training pipeline: models/<name>_model.keras
MODELS = {
    "U-Net": "unet_model.keras",
    "PSPNet": "pspnet_model.keras",
    "SegNet": "segnet_model.keras",
}

# LaPa label set, in index order.
CLASS_NAMES = [
    "background", "skin", "left eyebrow", "right eyebrow", "left eye",
    "right eye", "nose", "upper lip", "inner mouth", "lower lip", "hair",
]

# RGB colour per class.
PALETTE = np.array([
    (0, 0, 0),         # background
    (204, 164, 133),   # skin
    (76, 153, 0),      # left eyebrow
    (128, 204, 0),     # right eyebrow
    (0, 128, 255),     # left eye
    (0, 204, 255),     # right eye
    (255, 191, 0),     # nose
    (255, 64, 64),     # upper lip
    (153, 0, 76),      # inner mouth
    (255, 128, 170),   # lower lip
    (140, 82, 45),     # hair
], dtype=np.uint8)


def get_repo_id() -> str:
    try:
        if "HF_REPO_ID" in st.secrets:
            return st.secrets["HF_REPO_ID"]
    except Exception:
        # No secrets.toml present - fine, fall through.
        pass
    return os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID)


def get_hf_token():
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------- model loading

@st.cache_resource(max_entries=1, show_spinner=False)
def load_model(model_name: str, repo_id: str):
    """Load one model, keeping at most one in memory.

    max_entries=1 makes Streamlit evict the previously loaded model, which matters
    because SegNet alone is ~530 MB on disk and the three together will not fit in
    a free Streamlit Cloud container.
    """
    import tensorflow as tf
    from huggingface_hub import hf_hub_download

    filename = MODELS[model_name]
    local_path = os.path.join(os.path.dirname(__file__), "models", filename)

    # Prefer a local copy (handy when developing), otherwise fetch from the Hub.
    if not os.path.exists(local_path):
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=get_hf_token(),
        )

    # compile=False skips the custom training metrics (iou, dice_coefficient,
    # precision, recall); inference does not need them.
    return tf.keras.models.load_model(local_path, compile=False)


# ---------------------------------------------------------------- inference

def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Reproduce the training preprocessing exactly.

    Training read images with cv2.imread (BGR channel order) and scaled to [0, 1]
    without converting to RGB, so the app must feed BGR too.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (IMAGE_W, IMAGE_H))
    return (resized / 255.0).astype(np.float32)


def predict_mask(model, pil_image: Image.Image) -> np.ndarray:
    """Return a 256x256 mask of class indices."""
    batch = np.expand_dims(preprocess(pil_image), axis=0)
    prediction = model.predict(batch, verbose=0)[0]
    return np.argmax(prediction, axis=-1).astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def overlay(pil_image: Image.Image, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the colour mask over the original image, at the original resolution."""
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    colored = colorize(mask_full)

    blended = cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0.0)
    # Leave the background class untouched so the face stays readable.
    blended[mask_full == 0] = rgb[mask_full == 0]
    return blended


def class_distribution(mask: np.ndarray):
    counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
    total = counts.sum()
    return [
        {"Class": CLASS_NAMES[i], "Pixels": int(counts[i]), "Share": counts[i] / total}
        for i in range(NUM_CLASSES)
        if counts[i] > 0
    ]


def to_png_bytes(array: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="Facial Parts Segmentation", page_icon=":art:", layout="wide")

st.title("Facial Parts Segmentation")
st.caption(
    "U-Net, PSPNet and SegNet trained on the LaPa dataset - 11 facial classes at 256x256."
)

with st.sidebar:
    st.header("Settings")

    repo_id = st.text_input(
        "Hugging Face model repo",
        value=get_repo_id(),
        help="Repo holding unet_model.keras, pspnet_model.keras and segnet_model.keras.",
    )

    mode = st.radio("Mode", ["Single model", "Compare all three"])

    if mode == "Single model":
        chosen = [st.selectbox("Model", list(MODELS))]
    else:
        chosen = list(MODELS)
        st.info(
            "Models run one at a time and are evicted between runs to stay inside "
            "the memory limit. Expect this to be slow on CPU."
        )

    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)

    st.divider()
    st.subheader("Legend")
    for name, color in zip(CLASS_NAMES, PALETTE):
        swatch = (
            "<span style=\"display:inline-block;width:12px;height:12px;"
            f"background:rgb({color[0]},{color[1]},{color[2]});"
            "border:1px solid #888;margin-right:8px;\"></span>"
        )
        st.markdown(swatch + name, unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a face image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded is None:
    st.info("Upload a portrait to run the segmentation.")
    st.stop()

models_dir = os.path.join(os.path.dirname(__file__), "models")
if not repo_id.strip() and not os.path.isdir(models_dir):
    st.error(
        "No model source configured. Set a Hugging Face repo in the sidebar, "
        "or in .streamlit/secrets.toml as HF_REPO_ID."
    )
    st.stop()

image = Image.open(uploaded)

st.subheader("Input")
st.image(image, width=360)

for model_name in chosen:
    st.divider()
    st.subheader(model_name)

    try:
        with st.spinner(f"Loading {model_name}..."):
            model = load_model(model_name, repo_id)
    except Exception as exc:
        st.error(f"Could not load {model_name}: {exc}")
        continue

    with st.spinner(f"Running {model_name}..."):
        mask = predict_mask(model, image)

    mask_rgb = colorize(mask)
    blend = overlay(image, mask, alpha)

    left, right = st.columns(2)
    with left:
        st.image(mask_rgb, caption="Segmentation mask", use_container_width=True)
        st.download_button(
            "Download mask",
            to_png_bytes(mask_rgb),
            file_name=f"{model_name.lower()}_mask.png",
            mime="image/png",
            key=f"mask-{model_name}",
        )
    with right:
        st.image(blend, caption="Overlay", use_container_width=True)
        st.download_button(
            "Download overlay",
            to_png_bytes(blend),
            file_name=f"{model_name.lower()}_overlay.png",
            mime="image/png",
            key=f"overlay-{model_name}",
        )

    with st.expander("Detected classes"):
        st.dataframe(
            class_distribution(mask),
            column_config={
                "Share": st.column_config.ProgressColumn(
                    "Share", format="%.2f%%", min_value=0.0, max_value=1.0
                )
            },
            hide_index=True,
            use_container_width=True,
        )
