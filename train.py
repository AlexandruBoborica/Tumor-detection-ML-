from ultralytics import YOLO

def train_bmshare():
    model = YOLO('yolov8m.pt')

    model.train(
        data='brats_data.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        lr0=0.001,
        device=0,
        workers=4,
        project='runs/train',
        name='brats_yolov8m_advanced_on_brats',
        exist_ok=True,
        augment=True,          
        patience=15,           
        optimizer='SGD',     
        save=True,        
        verbose=True,
    )

if __name__ == '__main__':
    train_bmshare()
