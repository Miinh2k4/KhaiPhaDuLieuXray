import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception, MobileNetV2
import numpy as np
import cv2
from PIL import Image
import io
import os

# =========================
# ⚙️ CẤU HÌNH
# =========================
IMG_SIZE = (224, 224)
MODEL_XCEPTION_PATH = "xception_fixed_v7.keras"      # 👉 sửa đường dẫn nếu cần
MODEL_MOBILENET_PATH = "mobilenetv2_fixed_v1.keras"  # 👉 sửa đường dẫn nếu cần
LAST_CONV_XCEPTION = "block14_sepconv2_act"
LAST_CONV_MOBILENETV2 = "out_relu"

st.set_page_config(page_title="So sánh Xception vs MobileNetV2", layout="wide")

# =========================
# 🧠 HÀM TẠO MODEL "SẠCH"
# =========================
def create_clean_xception(dropout=0.5):
    base = Xception(weights=None, include_top=False, input_shape=IMG_SIZE + (3,), name="xception")
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="clean_xception")

def create_clean_mobilenetv2(dropout=0.5):
    base = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,), name="MobilenetV2")
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="clean_mobilenetv2")

def copy_weights_by_name(src, dst):
    src_map = {l.name: l for l in src.layers}
    for l in dst.layers:
        if l.name in src_map:
            w = src_map[l.name].get_weights()
            if w:
                try: l.set_weights(w)
                except: pass

def get_dropout_rate(model, default=0.5):
    try: return model.get_layer("dropout").rate
    except: return default

# =========================
# 💾 LOAD & REBUILD
# =========================
@st.cache_resource
def load_and_rebuild_model(path, arch):
    if not os.path.exists(path):
        st.error(f"❌ Không tìm thấy file: {path}")
        return None

    try:
        src = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Lỗi khi load model từ {path}: {e}")
        return None

    rate = get_dropout_rate(src)
    if arch == "xception":
        clean = create_clean_xception(rate)
    else:
        clean = create_clean_mobilenetv2(rate)
    copy_weights_by_name(src, clean)
    return clean

# =========================
# 🔥 GRAD-CAM
# =========================
def make_gradcam(img_array, model, arch):
    if arch == "xception":
        base = Xception(weights=None, include_top=False, input_shape=IMG_SIZE + (3,), name="cam_xcep")
        last_conv = LAST_CONV_XCEPTION
        base_name = "xception"
    else:
        base = MobileNetV2(weights=None, include_top=False, input_shape=IMG_SIZE + (3,), name="cam_mnet")
        last_conv = LAST_CONV_MOBILENETV2
        base_name = "MobilenetV2"

    try:
        last_layer = base.get_layer(last_conv)
    except Exception as e:
        st.error(f"Lỗi: không tìm thấy layer {last_conv}: {e}")
        return None

    # Copy weights base
    try:
        src_base = model.get_layer(base_name)
        src_map = {l.name: l for l in src_base.layers}
        for l in base.layers:
            if l.name in src_map:
                w = src_map[l.name].get_weights()
                if w:
                    try: l.set_weights(w)
                    except: pass
    except:
        pass

    extractor = models.Model(inputs=base.input, outputs=last_layer.output)
    cin = tf.keras.Input(shape=extractor.output.shape[1:])
    x = layers.GlobalAveragePooling2D()(cin)
    x = layers.Dropout(get_dropout_rate(model))(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    head = models.Model(cin, x)

    try:
        head.layers[-1].set_weights(model.get_layer("dense").get_weights())
    except:
        pass

    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_out = extractor(img_array, training=False)
        tape.watch(conv_out)
        preds = head(conv_out, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return None
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled, conv_out), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return heatmap / (np.max(heatmap) + 1e-8)

# =========================
# 🖼️ ẢNH & TIỆN ÍCH
# =========================
def preprocess_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.array(img), np.expand_dims(arr, axis=0)

def overlay_heatmap(rgb_np, heatmap):
    hm = cv2.resize(heatmap, IMG_SIZE)
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_np, 0.6, hm_color, 0.4, 0)

# =========================
# 🚀 GIAO DIỆN
# =========================
# --- Chia layout 1/4 : 3/4 ---
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("##  So sánh Grad-CAM\n### Xception vs MobileNetV2")
    st.markdown("Tải lên 1 ảnh X-quang để xem **so sánh trực quan** giữa hai mô hình.")
    uploaded = st.file_uploader(" Chọn ảnh X-quang (JPG/PNG):", type=["jpg", "jpeg", "png"])

with col_right:
    # --- Load model ---
    with st.spinner(" Đang tải mô hình..."):
        xcep_model = load_and_rebuild_model(MODEL_XCEPTION_PATH, "xception")
        mnet_model = load_and_rebuild_model(MODEL_MOBILENET_PATH, "mobilenetv2")

    if uploaded and (xcep_model and mnet_model):
        rgb_np, x = preprocess_image_bytes(uploaded.getvalue())

        # Xception
        p1 = xcep_model.predict(x, verbose=0)[0][0]
        lbl1 = "Pneumonia" if p1 > 0.5 else "Normal"
        prob1 = p1 if p1 > 0.5 else 1 - p1
        hm1 = make_gradcam(x, xcep_model, "xception")
        img_xcep = overlay_heatmap(rgb_np, hm1) if hm1 is not None else rgb_np

        # MobileNetV2
        p2 = mnet_model.predict(x, verbose=0)[0][0]
        lbl2 = "Pneumonia" if p2 > 0.5 else "Normal"
        prob2 = p2 if p2 > 0.5 else 1 - p2
        hm2 = make_gradcam(x, mnet_model, "mobilenetv2")
        img_mnet = overlay_heatmap(rgb_np, hm2) if hm2 is not None else rgb_np

        # HIỂN THỊ 3 ẢNH LỚN TRÊN 1 HÀNG
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(rgb_np, caption="(1) Ảnh Gốc", use_column_width=True)
        with c2:
            st.image(img_xcep, caption=f"(2) Xception — {lbl1} ({prob1:.3f})", use_column_width=True)
        with c3:
            st.image(img_mnet, caption=f"(3) MobileNetV2 — {lbl2} ({prob2:.3f})", use_column_width=True)
    else:
        st.info(" Vui lòng tải ảnh để hiển thị so sánh.")
