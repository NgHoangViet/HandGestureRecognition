# HandGestureRecognition
## Dùng cử chỉ tay để tạo điều khiển video: stop, tăng, giảm âm lượng
## *Python: Dùng thư viện opencv
![](C:\Users\PC\Pictures\1.png)
##Đếm số lượng ngón tay
###Dùng bộ lọc Gaussian
    #applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
###Thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
###Show thresholded image
    cv2.imshow('Thresholded', thresh1)
![](C:\Users\PC\Pictures\278219141_5201918339831283_3299538251744030163_n.jpg)
##Dùng thư viện Mediapipe để xác định và vẽ 21 mốc điểm của bàn tay để xác định vị trí các ngón
![img_2.png](img_2.png)
## *Học máy: Nhận dạng cử chỉ tay với Tensorflow Object Detection (Mô hình SSD)
###1. Thu thập hình ảnh từ webcam và opencv
###2. Dùng LabelImg cho Label sign language 
####item  
	name:'Play/Pause'
	id:1

	name:'Volume Up'
	id:2

    name:'Volume Down'
	id:3

###3. Cài đặt Tensorflow Object Detection pipeline configuration
###Model: ssd mobnet
####Mô hình SSD được chia làm hai giai đoạn:
        -Trích xuất feature map (dựa vào mạng cơ sở VGG16) để tăng hiệu quả trong việc phát hiện. Ở đây sử dụng MobileNet.
        -Áp dụng các bộ lọc tích chập để có thể detect được các đối tượng.
###4.Train mô hình
    pipeline_config.model.ssd.num_classes = 2
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
###5. Nhận dạng cử chỉ tay với opencv
 
