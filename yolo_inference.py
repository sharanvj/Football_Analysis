from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('Input_Videos/08fd33_4.mp4',save=True, stream = True )
print(results[0])
print("==============================================")
for box in results[0].boxes:
    print(box)