from ultralytics import YOLO

model = YOLO('yolov8l-cls.pt')  # will auto-download a fresh file

model.train(
    data=r"E:\XII IBCP\AI\ML SA\Processed_Dataset",  # your dataset path
    epochs=30,
    imgsz=224,
    batch=16
)
