import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2

# ============================================================
# Cấu hình & Tham số Toàn cục (Phải giống với file train)
# ============================================================
BASE_DIR = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Cấu hình Mô hình Đã Lưu (Input của bước này) ---
XCEPTION_MODEL_PATH = "/kaggle/working/xception_chestxray_finetuned.h5"
MOBILENET_MODEL_PATH = "/kaggle/working/mobilenetv2_chestxray_finetuned.h5"
EFFICIENTNET_MODEL_PATH = "/kaggle/working/efficientnetb0_chestxray_finetuned.h5"

# --- Cấu hình Grad-CAM ---
XCEPTION_LAST_CONV_LAYER_NAME = "block14_sepconv2_act"
MOBILENET_LAST_CONV_LAYER_NAME = "out_relu"
EFFICIENTNET_LAST_CONV_LAYER_NAME = "top_conv"
TEST_IMAGE_PATH = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray/test/PNEUMONIA/person11_virus_38.jpeg"

# ============================================================
# 1. Hàm Chuẩn bị Dữ liệu (Chỉ cần tạo test_gen)
# ============================================================
def get_test_generator(test_dir, img_size, batch_size):
    """Tạo ImageDataGenerator cho tập Test."""
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return test_gen

# ============================================================
# 2. Hàm Đánh giá và Vẽ biểu đồ (GIỮ NGUYÊN)
# ============================================================
def evaluate_and_report(model, test_gen, model_name="Mô hình"):
    """Đánh giá chi tiết trên tập Test và hiển thị báo cáo."""
    print(f"\n--- Đánh giá chi tiết trên tập Test ({model_name}) ---")
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

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

# NOTE: Hàm plot_training_history bị loại bỏ vì không có history object được lưu.
# Để thực hiện, cần phải lưu history object trong bước train (file training.py)

# ============================================================
# 3. Hàm Grad-CAM (GIỮ NGUYÊN)
# ============================================================
def find_dynamic_layer_name(model, name_part):
    """Tìm tên layer đầu tiên chứa chuỗi 'name_part'."""
    for layer in model.layers:
        if name_part in layer.name:
            return layer.name
    raise ValueError(f"Không tìm thấy layer nào chứa '{name_part}' trong tên.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Tính toán Grad-CAM heatmap cho ảnh đầu vào."""
    
    try:
        pool_name = find_dynamic_layer_name(model, 'global_average_pooling2d')
        dropout_name = find_dynamic_layer_name(model, 'dropout')
        dense_name = find_dynamic_layer_name(model, 'dense')
    except ValueError as e:
        raise ValueError(f"Lỗi tìm tên lớp trong mô hình: {e}")

    base_model = model.layers[0] 
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_output = grad_model(img_array)
        tape.watch(conv_output)

        x = model.get_layer(pool_name)(conv_output)
        x = model.get_layer(dropout_name)(x)
        preds = model.get_layer(dense_name)(x)

        loss = preds[:, 0]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def display_gradcam(image_path, model, last_conv_layer_name, img_size, model_name):
    """Tải ảnh, tạo Grad-CAM và hiển thị kết quả."""
    print(f"\n--- Grad-CAM Visualization ({model_name}) ---")
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array, verbose=0)[0][0]
    pred_label = "Pneumonia" if pred > 0.5 else "Normal"
    prob = pred if pred > 0.5 else 1 - pred
    print(f"Dự đoán: {pred_label} (Xác suất: {prob:.3f})")

    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    except Exception as e:
        print(f"Lỗi tạo Grad-CAM: {e}")
        return

    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img_orig, img_size)

    heatmap_resized = cv2.resize(heatmap, (img_size[0], img_size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)
    
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

def main_evaluation_gradcam():
    """Tải mô hình đã lưu, đánh giá và chạy Grad-CAM cho tất cả mô hình."""
    
    model_info_list = [
        ("Xception", XCEPTION_MODEL_PATH),
        ("MobileNetV2", MOBILENET_MODEL_PATH),
        ("EfficientNetB0", EFFICIENTNET_MODEL_PATH),
    ]

    last_conv_map = {
        "Xception": XCEPTION_LAST_CONV_LAYER_NAME,
        "MobileNetV2": MOBILENET_LAST_CONV_LAYER_NAME,
        "EfficientNetB0": EFFICIENTNET_LAST_CONV_LAYER_NAME,
    }
    
    # 1. Chuẩn bị Test Generator
    test_gen = get_test_generator(TEST_DIR, IMG_SIZE, BATCH_SIZE)

    for model_name, model_path in model_info_list:
        print(f"\n################ BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ & GRAD-CAM CHO: {model_name} ################")
        
        # 2. Tải mô hình
        try:
            # Cần biên dịch lại mô hình với learning_rate thấp sau khi load
            # để đảm bảo khả năng tương thích khi fine-tuning đã được thực hiện
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            last_conv_name = last_conv_map[model_name]
        except Exception as e:
            print(f" LỖI tải mô hình hoặc tìm tên layer cho {model_name} từ {model_path}: {e}")
            continue

        # 3. Đánh giá
        evaluate_and_report(model, test_gen, model_name)

        # 4. Grad-CAM
        display_gradcam(TEST_IMAGE_PATH, model, last_conv_name, IMG_SIZE, model_name)
        
        print(f"################ HOÀN THÀNH QUY TRÌNH ĐÁNH GIÁ & GRAD-CAM CHO: {model_name} ################")

# ============================================================
# Thực thi
# ============================================================
if __name__ == '__main__':
    main_evaluation_gradcam()
