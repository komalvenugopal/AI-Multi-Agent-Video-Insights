from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
import getpass
import os
import base64
import requests

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

folder_path = '/Users/rithik/Desktop/hackathon/ragathon/chunks/images'
image_paths = os.listdir(folder_path)
print(image_paths)

# Getting the base64 string
base64_images = [encode_image(os.path.join(folder_path,image_path)) for image_path in image_paths]
print(len(base64_images))

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}

payload = {
  "model": "gpt-4o",
  "messages": [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "These are the key frames images which indicate change of scenes when extracted from the set of image frames. Give me an good overall description summary of what is happening in the video so that whenever I search for it in the future, i can know what is happening here."
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            for base64_image in base64_images
        ]
    }],
    "max_tokens":300,
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_dict = response.json()
content = response_dict['choices'][0]['message']['content']