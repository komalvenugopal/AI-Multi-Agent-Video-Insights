import cv2, os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context


def get_frame_features(frame, model, transform):
    # Convert OpenCV image (BGR numpy array) to RGB PIL Image
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_tensor = transform(frame).unsqueeze(0)  # Transform and add batch dimension
    with torch.no_grad():
        features = model(frame_tensor)
    return features

def extract_keyframes_dl(video_path, output_dir, threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    prev_features = get_frame_features(prev_frame, model, transform)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features = get_frame_features(frame, model, transform)
        similarity = torch.nn.functional.cosine_similarity(prev_features, features, dim=1)

        if similarity.item() < threshold:
            # Save the frame as an image file
            frame_path = os.path.join(output_dir, f"keyframe_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"Saved keyframe {frame_count} at {frame_path}")

        prev_features = features
        frame_count += 1

    cap.release()

# Example usage
video_path = "sources/bahubali/chunks/videos/video_20_40.mp4"
output_dir = "sources/bahubali/chunks/images"
extract_keyframes_dl(video_path, output_dir)

