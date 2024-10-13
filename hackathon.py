import asyncio
import os
import requests
from moviepy.editor import VideoFileClip
from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import ssl

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"
ssl._create_default_https_context = ssl._create_stdlib_context

class VideoProcessingWorkflow:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.setup_directories()
        
    def setup_directories(self):
        directories = [self.output_dir, 
                       os.path.join(self.output_dir, "videos"),
                       os.path.join(self.output_dir, "audios"),
                       os.path.join(self.output_dir, "transcripts"),
                       os.path.join(self.output_dir, "images")]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    async def process_subclip(self, start_time, end_time, subclip_path, audio_path):
        try:
            with VideoFileClip(self.video_path).subclip(start_time, end_time) as clip:
                clip.write_videofile(subclip_path, codec='mpeg4')
                clip.audio.write_audiofile(audio_path)
                transcript = query(audio_path)
                transcript_path = os.path.join(self.output_dir, "transcripts", f"transcript_{start_time}_{end_time}.txt")
                with open(transcript_path, 'w') as f:
                    f.write(transcript['text'])
                print(f"Processed and transcribed video and audio from {start_time} to {end_time} seconds.")
                
                # Extract keyframes
                images_dir = os.path.join(self.output_dir, "images", f"video_{start_time}_{end_time}")
                os.makedirs(images_dir, exist_ok=True)
                self.extract_keyframes_dl(subclip_path, images_dir)
        except Exception as e:
            print(f"Error processing media from {start_time} to {end_time} seconds: {str(e)}")

    async def split_video_and_audio(self):
        try:
            with VideoFileClip(self.video_path) as clip:
                duration = int(clip.duration)
                tasks = []
                for start_time in range(0, duration, 20):
                    end_time = min(start_time + 20, duration)
                    subclip_path = os.path.join(self.output_dir, "videos", f"video_{start_time}_{end_time}.mp4")
                    audio_path = os.path.join(self.output_dir, "audios", f"audio_{start_time}_{end_time}.mp3")
                    task = asyncio.create_task(self.process_subclip(start_time, end_time, subclip_path, audio_path))
                    tasks.append(task)
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error opening video file: {str(e)}")

    def get_frame_features(self, frame, model, transform):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            features = model(frame_tensor)
        return features

    def extract_keyframes_dl(self, video_path, output_dir, threshold=0.3):
        print("Generating keyframes...")
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()

        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()

        frame_count = 0
        prev_features = self.get_frame_features(prev_frame, model, transform)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = self.get_frame_features(frame, model, transform)
            similarity = torch.nn.functional.cosine_similarity(prev_features, features, dim=1)

            if similarity.item() < threshold:
                frame_path = os.path.join(output_dir, f"keyframe_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"Saved keyframe {frame_count} at {frame_path}")

            prev_features = features
            frame_count += 1

        cap.release()

def download_video(url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_output_path = os.path.join(output_path, 'movie.mp4')

    if os.path.exists(full_output_path):
        os.remove(full_output_path)
        print("Existing file removed.")

    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)
 
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename='movie.mp4')

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

if __name__ == "__main__":
    source = "sources/bahubali/"
    video_path = source + "movie.mp4"
    output_dir = source + "chunks"
    
    print("YOUTUBE: Download started")
    download_video('https://www.youtube.com/watch?v=7z1bv8CtQxs', source)
    
    workflow = VideoProcessingWorkflow(video_path, output_dir)
    asyncio.run(workflow.split_video_and_audio())