from ultralytics import YOLO

def resume_bmshare():
    model = YOLO('runs/train/brats_yolov8m_advanced_on_brats/weights/last.pt')  # path to last checkpoint

    model.train(
        resume=True  # this flag resumes from checkpoint automatically
    )

if __name__ == '__main__':
    resume_bmshare()
