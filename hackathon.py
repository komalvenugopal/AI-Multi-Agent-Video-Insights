import asyncio
import os
import requests
from moviepy.editor import VideoFileClip
from bs4 import BeautifulSoup
from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import ssl

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"
ssl._create_default_https_context = ssl._create_stdlib_context
class VideoProcessingWorkflow:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "videos")):
            os.makedirs(os.path.join(output_dir, "videos"))
        if not os.path.exists(os.path.join(output_dir, "audios")):
            os.makedirs(os.path.join(output_dir, "audios"))
        if not os.path.exists(os.path.join(output_dir, "transcripts")):
            os.makedirs(os.path.join(output_dir, "transcripts"))

        
    async def process_subclip(self, start_time, end_time, subclip_path, audio_path):
        """Process a video subclip asynchronously."""
        try:
            with VideoFileClip(self.video_path).subclip(start_time, end_time) as clip:
                clip.write_videofile(subclip_path, codec='mpeg4')
                clip.audio.write_audiofile(audio_path)
                # After saving the audio, transcribe it using the Hugging Face API
                transcript = query(audio_path)  # Assuming the API returns text directly
                transcript_path = os.path.join(self.output_dir, "transcripts", f"transcript_{start_time}_{end_time}.txt")
                with open(transcript_path, 'w') as f:
                    f.write(transcript['text'])
                print(f"Processed and transcribed video and audio from {start_time} to {end_time} seconds.")
        except Exception as e:
            print(f"Error processing media from {start_time} to {end_time} seconds: {str(e)}")

    def transcribe_audio(self, audio_path, transcript_path):
        """Transcribe audio using Whisper and save the transcript."""
        result = self.model.transcribe(audio_path)
        with open(transcript_path, 'w') as file:
            file.write(result["text"])

    async def split_video_and_audio(self):
        """Asynchronously splits video into 20-second clips and extracts audio."""
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

def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define the full path for the video
    full_output_path = os.path.join(output_path, 'movie.mp4')

    # Check if the file already exists and remove it
    if os.path.exists(full_output_path):
        os.remove(full_output_path)
        print("Existing file removed.")

    yt = YouTube(url, on_progress_callback = on_progress)
    print(yt.title)
 
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename='movie.mp4')


if __name__ == "__main__":
    source = "sources/bahubali/"
    video_path = source + "movie.mp4"
    output_dir = source + "chunks"
    print("YOUTUBE: Download started")
    download_video('https://www.youtube.com/watch?v=7z1bv8CtQxs',source)
    
    # audio to text
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
    headers = {"Authorization": "Bearer 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"}


    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    workflow = VideoProcessingWorkflow(video_path, output_dir)
    asyncio.run(workflow.split_video_and_audio())
