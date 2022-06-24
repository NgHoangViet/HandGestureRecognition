# HandGestureRecognition
## Dùng cử chỉ tay để tạo điều khiển video: Play/stop, tăng, giảm âm lượng, tua lại, tua nhanh, mở full-screen
# Mô hình học máy
## 1.Chuẩn bị dữ liệu với Tensowflow Dataset API
### Pretrained
### Data: Dữ liệu cắt từ webcam với cử chỉ tay tương ứng 6 điều khiển
### Đưa Data về kiểu TFRecord:TFRecord là định dạng văn bản để lưu trữ một chuỗi các bản ghi nhị phân. Chuyển đổi sang TFRecord có một số lợi thế sau:

    1: Lưu trữ hiệu quả: dữ liệu TFRecord có thể chiếm ít dung lượng hơn dữ liệu gốc; nó cũng có thể được phân dữ liệu thành nhiều tệp. Khi save dưới dạng nhị phân, 
    ta sẽ tiết kiệm nhiều dung lượng.
    2: I/O nhanh: định dạng TFRecord có thể được đọc song song, rất hữu ích khi train trên TPU hoặc nhiều máy chủ.
    Tệp độc lập: dữ liệu TFRecord có thể được đọc từ một nguồn duy nhất. Ví dụ như data ảnh ta sẽ có ảnh nhiều folder, label và các chú thích của ảnh (annotation) ở nơi khác. TFRecord sẽ tóm tắt lại dữ liệu về một chỗ.
### Update Config
### 2. Dùng LabelImg cho Label sign language 
#### item
	name:'Play/Pause'
	id:1

	name:'Volume Up'
	id:2

    name:'Volume Down'
	id:3

	name:'Fast_Forward'
	id:4

	name:'Rewind'
	id:5

	name:'Full_Screen'
	id:6
### 3. Cài đặt Tensorflow Object Detection pipeline configuration
### Model: ssd mobnet
#### Mô hình SSD được chia làm hai giai đoạn:
        -Trích xuất feature map để tăng hiệu quả trong việc phát hiện. Ở đây sử dụng MobileNet.
        -Áp dụng các bộ lọc tích chập để có thể detect được các đối tượng.
![image](https://user-images.githubusercontent.com/94554407/175428471-bbde23aa-754c-45a3-8105-0de0a221910e.png)

#### Config
    pipeline_config.model.ssd.num_classes = 6
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
### 4.Train mô hình
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
#### Load Train Model From Checkpoint
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model) 
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()
### 5. Nhận dạng cử chỉ tay trong thời gian thực

![tf_sample](https://user-images.githubusercontent.com/94554407/174918088-4792b937-a4a4-4e6d-b3fc-de94467593d0.png)

## *Python: Dùng thư viện opencv
## Đếm số lượng ngón tay
### Dùng bộ lọc Gaussian
    #applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
### Thresholdin: Otsu's Binarization method
### Tách ngưỡng với Otsu's method. Ở dạng đơn giản nhất, thuật toán trả về một ngưỡng cường độ duy nhất giúp 
### tách các pixel thành hai lớp nền trắng và nền đen để thực hiện bước sau là contours.
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
### Show thresholded image
    cv2.imshow('Thresholded', thresh1)

![278219141_5201918339831283_3299538251744030163_n](https://user-images.githubusercontent.com/94554407/163492275-bdc46d69-cdac-45ab-ad03-794969a104b9.jpg)


### Find contour with max area
#### Để tìm contour chính xác, chúng ta cần phải nhị phân hóa bức ảnh. Trong opencv, việc tìm một contour là việc tìm một đối tượng có màu trắng trên nền đen

    cnt = max(contours, key=lambda x: cv2.contourArea(x))

### Create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

### Finding convex hull
    hull = cv2.convexHull(cnt)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    sub_counter = 0
### Drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
### Tìm góc giữa các ngón tay để xác định hình dạng của bàn tay tương ứng với 6 cử chỉ tay điều khiển
    for i in range(defects.shape[0]):
        if angle <= 134:
            count_defects += 1
            cv2.circle(crop_img, far, 5, [255, 255, 255], -1)

        elif 146.5 < angle <= 150:
            sub_counter = 1
