from PIL import Image
from ultralytics import YOLO
import torch

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on 'bus.jpg'
# results = model([f'test{i}.png' for i in range(3)], device="cuda:3")  # results list
results = model.predict(['test.png'], device=torch.device("cuda:2"), max_det=1, classes=[0])

# Visualize the results
for i, r in enumerate(results):
    print(r.boxes.xywhn)
    # print("Boxes", r.boxes)  # print the Boxes object containing the detection bounding boxes
    # print("Masks", r.masks)  # print the Masks object containing the detected instance masks
    # print("Keypoints", r.keypoints)  # print the Keypoints object containing the detected keypoints
    # print("Probs", r.probs)  # print the Probs object containing the detected class probabilities
    # print("Obb", r.obb)  # print the OBB object containing the oriented detection bounding boxes

    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    # r.show()

    # Save results to disk
    r.save(filename=f'results{i}.jpg')