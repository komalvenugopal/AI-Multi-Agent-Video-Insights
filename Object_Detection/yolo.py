from ultralytics import YOLO
from PIL import Image

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")



# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("image2.png")
print(results[0].tojson())


for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")