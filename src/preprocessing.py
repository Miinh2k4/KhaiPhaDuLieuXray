import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, MobileNetV2, EfficientNetB0
import numpy as np

# ============================================================
# Cấu hình & Tham số Toàn cục
# ============================================================
BASE_DIR = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INIT = 15
EPOCHS_FINE_TUNE = 5

# --- Cấu hình Lưu Mô hình ---
XCEPTION_MODEL_PATH = "/kaggle/working/xception_chestxray_finetuned.h5"
MOBILENET_MODEL_PATH = "/kaggle/working/mobilenetv2_chestxray_finetuned.h5"
EFFICIENTNET_MODEL_PATH = "/kaggle/working/efficientnetb0_chestxray_finetuned.h5"

# --- Cấu hình Grad-CAM (Tên layer cuối, được sử dụng trong bước 3) ---
XCEPTION_LAST_CONV_LAYER_NAME = "block14_sepconv2_act"
MOBILENET_LAST_CONV_LAYER_NAME = "out_relu"
EFFICIENTNET_LAST_CONV_LAYER_NAME = "top_conv"

# ============================================================
# Hàm Chuẩn bị Dữ liệu (GIỮ NGUYÊN)
# ============================================================
def prepare_data_generators(train_dir, val_dir, test_dir, img_size, batch_size):
    """Tạo và cấu hình các ImageDataGenerator cho tập Train, Val, Test."""
    print("--- 2. Chuẩn bị dữ liệu ---")
    
    # Kiểm tra đường dẫn
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Thư mục TRAIN không tìm thấy: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Thư mục VAL không tìm thấy: {val_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Thư mục TEST không tìm thấy: {test_dir}")

    # Khởi tạo các Generator (Không cần chạy flow_from_directory trong bước Preprocessing)
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Kích thước ảnh: {img_size}, Batch size: {batch_size}")
    print("Cấu hình Augmentation cho tập Train đã sẵn sàng.")

    # Trả về các datagen để có thể tạo lại flow_from_directory sau này (Nếu cần thiết)
    # Tuy nhiên, trong tập lệnh này ta chỉ trả về một thông báo đơn giản.
    return True

# ============================================================
# Thực thi
# ============================================================
if __name__ == '__main__':
    print("--- 1. Cấu hình cơ bản ---")
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices('GPU'))
    print("Các hằng số toàn cục đã được thiết lập.")
    
    # Chạy Preprocessing
    try:
        prepare_data_generators(TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE)
        print("\n TIỀN XỬ LÝ HOÀN TÀNH. Các đường dẫn dữ liệu và cấu hình đã được kiểm tra.")
    except FileNotFoundError as e:
        print(f"\n LỖI: {e}. Vui lòng kiểm tra đường dẫn BASE_DIR.")
