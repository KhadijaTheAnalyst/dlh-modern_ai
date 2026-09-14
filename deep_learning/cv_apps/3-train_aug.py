from ultralytics import YOLO

def train_with_augmentation(data, model_path="yolov8n.pt", aug=None, custom_albu=None,
                           epochs=50, imgsz=640, batch=16, save=True, plots=True, verbose=False):
    model_obj = YOLO(model_path)

    train_config = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'save': save,
        'plots': plots,
        'verbose': verbose,
    }

    results = model_obj.train(**train_config)

    return model_obj, results
