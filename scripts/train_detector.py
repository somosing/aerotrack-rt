from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
        project="results/training",
        name="yolo26n_visdrone_640_e30",
        pretrained=True,
        plots=True,
        save=True,
    )


if __name__ == "__main__":
    main()
