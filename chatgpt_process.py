import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-gECJbZO1y4sk3NoUB1SbT3BlbkFJ5QdYttqNjmS1hSsGQIJY'
from openai import OpenAI
import json
from tqdm.auto import tqdm
import base64
import requests
import torch
import re
import tiktoken
import pandas as pd
import io
from PIL import Image
import argparse
from pathlib import Path
from chatgpt_utils import PROMPT_ADD_VER1, PROMPT_SELECT_VER1, PROMPT_ADD_VER2

PROMPT = PROMPT_ADD_VER2

batch_body = """
    {{
    "custom_id":"{custom_id}",
    "method":"POST",
    "url":"/v1/chat/completions",
    "body": {payload}
    }}
"""
batch_body = re.sub("\s\s+" , " ", batch_body.replace("\n", ""))

payload = """
    {{
    "model": "gpt-4o",
    "messages": [
        {{
        "role": "user",
        "content": [
            {{
            "type": "text",
            "text": "{prompt}"
            }},
            {{
            "type": "image_url",
            "image_url": {{
                "url": "data:image/jpeg;base64,{base64_image}"
            }}
            }}
        ]
        }}
    ],
    "max_tokens": 3000
    }}
"""
payload =  re.sub("\s\s+" , " ", payload.replace("\n", ""))

def encode_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((512, 512))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return encoded_string

def prepare_data(df, start=None, end=None):
    image_paths = df["img_path"].to_numpy()
    output = []
    for idx, path in tqdm(enumerate(image_paths), total=len(image_paths)):
        base64_image = encode_image(path)
        current_payload = payload.format(prompt=PROMPT, base64_image=base64_image)
        output += [batch_body.format(custom_id=idx, payload=current_payload)]

    if (start is None) or (end is None):
        return output
    else:
        return output[start:end]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    PROMPT =  re.sub("\s\s+" , " ", PROMPT.replace("\n", "\\n"))
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(PROMPT)

    df = pd.read_csv(args.df_path)
    output = prepare_data(df, start=args.start, end=args.end)
    input_file_path = args.input_path
    os.makedirs(Path(input_file_path).parent, exist_ok=True)
    with open(input_file_path, 'w', encoding='utf-8') as f:
        for line in output:
            f.write(line+'\n')

    print(output[0][:1000])
    print(len(output))

    # exit()

    # send requests
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    print("batch_input_file_id: ", batch_input_file_id)

    current_batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": Path(args.df_path).stem
        }
    )
    batch_id = current_batch.id
    print("batch_id: ", batch_id)

    import time
    while True:
        try:
            time.sleep(30)
            output_file_id = client.batches.retrieve(batch_id).output_file_id
            content = client.files.content(output_file_id)

            output_file_path = args.output_path
            with open(output_file_path, 'w') as f:
                f.write(content.content.decode("utf-8"))
            
            print("Retrieved outputs")
            break
        
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in 30 seconds...")