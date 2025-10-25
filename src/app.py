import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Thêm các imports cho MobileNetV2 và EfficientNetB0
from tensorflow.keras.applications import Xception, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2

# ============================================================
# Cấu hình & Tham số Toàn cục (Bổ sung cho các mô hình mới)
# ============================================================
BASE_DIR = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INIT = 15
EPOCHS_FINE_TUNE = 5

# --- Cấu hình Xception ---
XCEPTION_MODEL_PATH = "/kaggle/working/xception_chestxray_finetuned.h5"
XCEPTION_LAST_CONV_LAYER_NAME = "block14_sepconv2_act" # Tên layer conv cuối cùng

# --- Cấu hình MobileNetV2 ---
MOBILENET_MODEL_PATH = "/kaggle/working/mobilenetv2_chestxray_finetuned.h5"
MOBILENET_LAST_CONV_LAYER_NAME = "out_relu" # Tên layer conv cuối cùng

# --- Cấu hình EfficientNetB0 ---
EFFICIENTNET_MODEL_PATH = "/kaggle/working/efficientnetb0_chestxray_finetuned.h5"
EFFICIENTNET_LAST_CONV_LAYER_NAME = "top_conv" # Tên layer conv cuối cùng

# ============================================================
# 1. Hàm Chuẩn bị Dữ liệu (GIỮ NGUYÊN)
# ============================================================
def prepare_data_generators(train_dir, val_dir, test_dir, img_size, batch_size):
    """Tạo và cấu hình các ImageDataGenerator cho tập Train, Val, Test."""
    print("--- 2. Chuẩn bị dữ liệu ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return train_gen, val_gen, test_gen

# ============================================================
# 2. Hàm Xây dựng Mô hình (Transfer Learning)
# ============================================================
def build_model_template(base_model_class, model_name, img_size):
    """Template xây dựng mô hình."""
    print(f"--- 3. Xây dựng mô hình {model_name} (Transfer Learning) ---")
    
    # Khởi tạo Base Model
    base_model = base_model_class(
        input_shape=img_size + (3,), include_top=False, weights='imagenet'
    )
    
    # Freeze ban đầu
    base_model.trainable = False

    # Xây dựng mô hình Classification
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name=f"{model_name}_classifier")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model, base_model

def build_xception_model(img_size):
    return build_model_template(Xception, "Xception", img_size)

# Hàm mới cho MobileNetV2
def build_mobilenetv2_model(img_size):
    return build_model_template(MobileNetV2, "MobileNetV2", img_size)

# Hàm mới cho EfficientNetB0
def build_efficientnetb0_model(img_size):
    return build_model_template(EfficientNetB0, "EfficientNetB0", img_size)

# ============================================================
# 3. Hàm Huấn luyện Mô hình (GIỮ NGUYÊN)
# ============================================================
def train_model(model, train_gen, val_gen, epochs, name="Ban đầu"):
    """Thực hiện quá trình huấn luyện mô hình."""
    print(f"\n--- 4. Huấn luyện {name} (EPOCHS={epochs}) ---")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1)
    ]
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks
    )
    return history

def fine_tune_model(model, base_model, train_gen, val_gen, fine_tune_epochs, save_path):
    """Thực hiện quá trình Fine-Tuning."""
    print("\n--- 5. Fine-Tuning ---")
    # Tự động chọn số lớp để unfreeze (ví dụ: 20 lớp cuối cho Xception, 10 lớp cho Mobile/Efficient)
    unfreeze_layers = 20 if "xception" in base_model.name else 10
    print(f"Mở băng {unfreeze_layers} lớp cuối của base_model ({base_model.name}) để fine-tune...")

    # Unfreeze các lớp cuối
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Biên dịch lại với Learning Rate nhỏ hơn
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history_ft = train_model(model, train_gen, val_gen, fine_tune_epochs, name="Fine-Tuning")

    # Lưu mô hình fine-tuned
    model.save(save_path)
    print(f"Đã lưu mô hình fine-tune tại: {save_path}")
    return history_ft

# ============================================================
# 4. Hàm Đánh giá và Vẽ biểu đồ (GIỮ NGUYÊN)
# ============================================================
def evaluate_and_report(model, test_gen, model_name="Mô hình"):
    """Đánh giá chi tiết trên tập Test và hiển thị báo cáo."""
    print(f"\n--- 7. Đánh giá chi tiết trên tập Test ({model_name}) ---")
    
    # Đánh giá cơ bản
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Dự đoán và Báo cáo Phân loại
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
    class_labels = list(test_gen.class_indices.keys())

    # Ma trận nhầm lẫn
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Ma trận nhầm lẫn ({model_name})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.show()

    # Báo cáo Phân loại
    print("\n Báo cáo Phân loại:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

def plot_training_history(history_init, history_ft, model_name="Mô hình"):
    """Vẽ biểu đồ Accuracy và Loss qua các Epoch."""
    print(f"\n--- 6. Vẽ biểu đồ huấn luyện ({model_name}) ---")
    
    # Kết hợp lịch sử từ 2 giai đoạn
    acc = history_init.history['accuracy'] + history_ft.history['accuracy']
    val_acc = history_init.history['val_accuracy'] + history_ft.history['val_accuracy']
    loss = history_init.history['loss'] + history_ft.history['loss']
    val_loss = history_init.history['val_loss'] + history_ft.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Val')
    plt.title(f"Accuracy qua các Epoch ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title(f"Loss qua các Epoch ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.show()

# ============================================================
# 5. Hàm Grad-CAM (Điều chỉnh nhỏ để tương thích tên layers)
# ============================================================
def find_dynamic_layer_name(model, name_part):
    """Tìm tên layer đầu tiên chứa chuỗi 'name_part'."""
    # Các lớp GlobalAveragePooling2D, Dropout, Dense trong Sequential model luôn được đánh số
    # nhưng chúng nằm ngoài base_model.
    for layer in model.layers:
        if name_part in layer.name:
            return layer.name
    raise ValueError(f"Không tìm thấy layer nào chứa '{name_part}' trong tên.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Tính toán Grad-CAM heatmap cho ảnh đầu vào."""
    
    # 1. Lấy tên layer chính xác từ mô hình đã tải
    try:
        pool_name = find_dynamic_layer_name(model, 'global_average_pooling2d')
        dropout_name = find_dynamic_layer_name(model, 'dropout')
        dense_name = find_dynamic_layer_name(model, 'dense')
    except ValueError:
        # Xử lý trường hợp mô hình không có các layer này (chẳng hạn nếu mô hình là base_model)
        raise ValueError("Các lớp 'global_average_pooling2d', 'dropout', 'dense' không tìm thấy trong mô hình.")

    # 2. Lấy base model (Xception, MobileNetV2, hoặc EfficientNetB0)
    # Base model luôn là layer đầu tiên trong Sequential model (index 0)
    base_model = model.layers[0] 

    # 3. Xây model trung gian ...
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    # 4. Chạy forward và tính gradient
    with tf.GradientTape() as tape:
        conv_output = grad_model(img_array)
        tape.watch(conv_output)

        # Chạy qua các lớp classification head
        x = model.get_layer(pool_name)(conv_output)
        x = model.get_layer(dropout_name)(x)
        preds = model.get_layer(dense_name)(x)

        # Chọn output cần tối ưu (lớp Pneumonia - index 0 vì binary classification)
        loss = preds[:, 0]
    
    # 5. Tính toán heatmap ...
    grads = tape.gradient(loss, conv_output)
    
    # Global average pooling của gradient qua các chiều không gian
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân pooled_grads với feature map để có "sức nặng"
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

    # ReLU (chỉ giữ lại các giá trị ảnh hưởng tích cực)
    heatmap = np.maximum(heatmap, 0)
    # Chuẩn hóa
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def display_gradcam(image_path, model, last_conv_layer_name, img_size, model_name):
    """Tải ảnh, tạo Grad-CAM và hiển thị kết quả."""
    print(f"\n--- 8. Grad-CAM Visualization ({model_name}) ---")
    
    # Xử lý ảnh đầu vào
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Dự đoán
    pred = model.predict(img_array, verbose=0)[0][0]
    pred_label = "Pneumonia" if pred > 0.5 else "Normal"
    prob = pred if pred > 0.5 else 1 - pred
    print(f"Dự đoán: {pred_label} (Xác suất: {prob:.3f})")

    # Tạo Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    except Exception as e:
        print(f"Lỗi tạo Grad-CAM: {e}")
        return

    # Hiển thị kết quả
    # (a) Ảnh gốc
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img_orig, img_size)

    # (b) Resize heatmap cho khớp và tạo màu
    heatmap_resized = cv2.resize(heatmap, (img_size[0], img_size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # (c) Chồng ảnh (Superimposed)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_orig)
    plt.title("Test X-ray Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='viridis')
    plt.title(f"CAM ({model_name})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Predicted: {pred_label} | {model_name}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ============================================================
# 6. Hàm Chạy Chính (Main Execution)
# ============================================================
def run_model_workflow(model_builder, model_name, save_path, train_gen, val_gen, test_gen, epochs_init, epochs_fine_tune):
    """Chạy toàn bộ quy trình cho một mô hình cụ thể."""
    print(f"\n################ BẮT ĐẦU QUY TRÌNH CHO MÔ HÌNH: {model_name} ################")
    
    # 2. Xây dựng mô hình
    model, base_model = model_builder(IMG_SIZE)

    # 3. Huấn luyện ban đầu (Transfer Learning)
    history_init = train_model(model, train_gen, val_gen, epochs_init, name=f"Ban đầu ({model_name})")

    # 4. Fine-Tuning
    history_ft = fine_tune_model(
        model, base_model, train_gen, val_gen, epochs_fine_tune, save_path
    )

    # 5. Vẽ biểu đồ huấn luyện
    plot_training_history(history_init, history_ft, model_name)

    # 6. Đánh giá chi tiết
    evaluate_and_report(model, test_gen, model_name)
    
    print(f"\n################ HOÀN THÀNH QUY TRÌNH CHO MÔ HÌNH: {model_name} ################")
    return save_path

def main_training_and_evaluation():
    """Chạy toàn bộ quy trình cho 3 mô hình."""
    print("--- 1. Cấu hình cơ bản ---")
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices('GPU'))
    
    # 1. Chuẩn bị dữ liệu (Chỉ chạy 1 lần)
    train_gen, val_gen, test_gen = prepare_data_generators(
        TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
    )

    model_list = [
        (build_xception_model, "Xception", XCEPTION_MODEL_PATH),
        (build_mobilenetv2_model, "MobileNetV2", MOBILENET_MODEL_PATH),
        (build_efficientnetb0_model, "EfficientNetB0", EFFICIENTNET_MODEL_PATH),
    ]
    
    saved_model_paths = []
    
    # Chạy quy trình cho từng mô hình
    for model_builder, name, path in model_list:
        saved_path = run_model_workflow(
            model_builder, name, path, train_gen, val_gen, test_gen, EPOCHS_INIT, EPOCHS_FINE_TUNE
        )
        saved_model_paths.append((name, saved_path))
        
    print("\n--- Hoàn thành quy trình huấn luyện và đánh giá cho tất cả mô hình ---")
    return saved_model_paths

def main_gradcam_visualization(model_info_list, image_path, img_size):
    """Tải mô hình đã lưu và chạy Grad-CAM cho tất cả mô hình."""
    
    # Map tên mô hình với tên layer conv cuối
    last_conv_map = {
        "Xception": XCEPTION_LAST_CONV_LAYER_NAME,
        "MobileNetV2": MOBILENET_LAST_CONV_LAYER_NAME,
        "EfficientNetB0": EFFICIENTNET_LAST_CONV_LAYER_NAME,
    }
    
    for model_name, model_path in model_info_list:
        print(f"\n--- Bắt đầu Quy trình Grad-CAM cho {model_name} ---")
        
        # Tải mô hình fine-tuned
        try:
            model = tf.keras.models.load_model(model_path)
            last_conv_name = last_conv_map[model_name]
        except Exception as e:
            print(f"Lỗi tải mô hình hoặc tìm tên layer cho {model_name} từ {model_path}: {e}")
            continue

        # Chạy Grad-CAM
        display_gradcam(image_path, model, last_conv_name, img_size, model_name)
        
        print(f"--- Hoàn thành Quy trình Grad-CAM cho {model_name} ---")

# ============================================================
# Thực thi
# ============================================================
if __name__ == '__main__':
    # Chạy quy trình huấn luyện và đánh giá
    # Kết quả trả về là list các tuple (model_name, model_path)
    model_paths = main_training_and_evaluation()

    # Chạy quy trình Grad-CAM
    test_image_path = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg"
    main_gradcam_visualization(model_paths, test_image_path, IMG_SIZE)
