import asyncio
import os
from moviepy.editor import VideoFileClip

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

    async def process_subclip(self, start_time, end_time, subclip_path, audio_path):
        """Process a video subclip asynchronously."""
        try:
            clip = VideoFileClip(self.video_path).subclip(start_time, end_time)
            clip.write_videofile(subclip_path, codec='mpeg4')  # Synchronous blocking operation for video
            clip.audio.write_audiofile(audio_path)  # Synchronous blocking operation for audio
            print(f"Processed video and audio from {start_time} to {end_time} seconds.")
        except Exception as e:
            print(f"Error processing media from {start_time} to {end_time} seconds: {str(e)}")
        finally:
            clip.close()

    async def split_video_and_audio(self):
        """Asynchronously splits video into 20-second clips and extracts audio."""
        try:
            clip = VideoFileClip(self.video_path)
            duration = int(clip.duration)
            tasks = []
            for start_time in range(0, duration, 20):
                end_time = min(start_time + 20, duration)
                subclip_path = os.path.join(self.output_dir, "videos", f"video_{start_time}_{end_time}.mp4")
                audio_path = os.path.join(self.output_dir, "audios", f"audio_{start_time}_{end_time}.mp3")
                # Schedule the processing of each subclip
                task = asyncio.create_task(self.process_subclip(start_time, end_time, subclip_path, audio_path))
                tasks.append(task)
            # Wait for all scheduled tasks to complete
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error opening video file: {str(e)}")
        finally:
            clip.close()

if __name__ == "__main__":
    source = "sources/bahubali/"
    video_path = source + "video.mp4"
    output_dir = source + "chunks"
    workflow = VideoProcessingWorkflow(video_path, output_dir)
    asyncio.run(workflow.split_video_and_audio())
    