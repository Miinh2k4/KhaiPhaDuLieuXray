import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

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

# --- Cấu hình Lưu Mô hình (Output của bước này) ---
XCEPTION_MODEL_PATH = "/kaggle/working/xception_chestxray_finetuned.h5"
MOBILENET_MODEL_PATH = "/kaggle/working/mobilenetv2_chestxray_finetuned.h5"
EFFICIENTNET_MODEL_PATH = "/kaggle/working/efficientnetb0_chestxray_finetuned.h5"

# ============================================================
# 1. Hàm Chuẩn bị Dữ liệu (GIỮ NGUYÊN - Cần cho bước Train)
# ============================================================
def prepare_data_generators(train_dir, val_dir, test_dir, img_size, batch_size):
    """Tạo và cấu hình các ImageDataGenerator cho tập Train, Val, Test."""
    print("--- 2. Chuẩn bị dữ liệu ---")
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
        horizontal_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
    )
    # Lưu ý: Không cần test_gen cho bước này, nhưng giữ nguyên để thống nhất
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
    )
    return train_gen, val_gen, test_gen

# ============================================================
# 2. Hàm Xây dựng Mô hình (Transfer Learning - GIỮ NGUYÊN)
# ============================================================
def build_model_template(base_model_class, model_name, img_size):
    """Template xây dựng mô hình."""
    print(f"--- 3. Xây dựng mô hình {model_name} (Transfer Learning) ---")
    
    base_model = base_model_class(
        input_shape=img_size + (3,), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

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

def build_mobilenetv2_model(img_size):
    return build_model_template(MobileNetV2, "MobileNetV2", img_size)

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
    unfreeze_layers = 20 if "xception" in base_model.name else 10
    print(f"Mở băng {unfreeze_layers} lớp cuối của base_model ({base_model.name}) để fine-tune...")

    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history_ft = train_model(model, train_gen, val_gen, fine_tune_epochs, name="Fine-Tuning")

    # LƯU MÔ HÌNH
    model.save(save_path)
    print(f"Đã lưu mô hình fine-tune tại: {save_path}")
    return history_ft

# ============================================================
# 4. Hàm Chạy Chính (Main Execution)
# ============================================================
def run_model_workflow(model_builder, model_name, save_path, train_gen, val_gen, epochs_init, epochs_fine_tune):
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
    
    print(f"\n################ HOÀN THÀNH QUY TRÌNH CHO MÔ HÌNH: {model_name} ################")
    return history_init, history_ft, save_path

def main_training():
    """Chạy toàn bộ quy trình cho 3 mô hình."""
    
    # 1. Chuẩn bị dữ liệu
    # Dù đã có bước tiền xử lý, ta vẫn phải gọi lại để có các generator
    train_gen, val_gen, _ = prepare_data_generators(
        TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
    ) # Bỏ qua test_gen vì không cần thiết cho bước này

    model_list = [
        (build_xception_model, "Xception", XCEPTION_MODEL_PATH),
        (build_mobilenetv2_model, "MobileNetV2", MOBILENET_MODEL_PATH),
        (build_efficientnetb0_model, "EfficientNetB0", EFFICIENTNET_MODEL_PATH),
    ]
    
    saved_model_paths = []
    
    # Chạy quy trình cho từng mô hình
    for model_builder, name, path in model_list:
        # Ở đây ta bỏ qua history_init, history_ft để tiết kiệm bộ nhớ,
        # vì chúng ta không vẽ biểu đồ trong bước này (để dành cho bước 3)
        _, _, saved_path = run_model_workflow(
            model_builder, name, path, train_gen, val_gen, EPOCHS_INIT, EPOCHS_FINE_TUNE
        )
        saved_model_paths.append((name, saved_path))
        
    print("\n--- Hoàn thành quy trình huấn luyện và lưu mô hình cho tất cả mô hình ---")
    return saved_model_paths

# ============================================================
# Thực thi
# ============================================================
if __name__ == '__main__':
    # Chạy quy trình huấn luyện và lưu mô hình
    main_training()
