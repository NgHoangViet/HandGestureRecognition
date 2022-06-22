# HandGestureRecognition
## Dùng cử chỉ tay để tạo điều khiển video: stop, tăng, giảm âm lượng
# Mô hình học máy
## 1.Chuẩn bị dữ liệu với Tensowflow Dataset API
### Pretrained
### Đưa Data về kiểu TFRecord:TFRecord là định dạng văn bản để lưu trữ một chuỗi các bản ghi nhị phân. Chuyển đổi sang TFRecord có một số lợi thế sau:

    Lưu trữ hiệu quả: dữ liệu TFRecord có thể chiếm ít dung lượng hơn dữ liệu gốc; nó cũng có thể được phân dữ liệu thành nhiều tệp. Khi save dưới dạng nhị phân, 
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
        -Trích xuất feature map (dựa vào mạng cơ sở VGG16) để tăng hiệu quả trong việc phát hiện. Ở đây sử dụng MobileNet.
        -Áp dụng các bộ lọc tích chập để có thể detect được các đối tượng.
#### Config
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
### 4.Train mô hình
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
### 5. Nhận dạng cử chỉ tay trong thời gian thực

![tf_sample](https://user-images.githubusercontent.com/94554407/174918088-4792b937-a4a4-4e6d-b3fc-de94467593d0.png)

## *Python: Dùng thư viện opencv
## Đếm số lượng ngón tay
### Dùng bộ lọc Gaussian
    #applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
### Thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
### Show thresholded image
    cv2.imshow('Thresholded', thresh1)

![278219141_5201918339831283_3299538251744030163_n](https://user-images.githubusercontent.com/94554407/163492275-bdc46d69-cdac-45ab-ad03-794969a104b9.jpg)


### find contour with max area
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

### create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

### finding convex hull
    hull = cv2.convexHull(cnt)

### drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
