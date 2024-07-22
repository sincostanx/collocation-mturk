from litellm import completion
import glob
import os
from pathlib import Path
import pandas as pd
from natsort import natsorted
from tqdm.auto import tqdm

def prepare_message(question, img_path):
    return [{
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"""
                Answer the following questions using the given image with "yes" or "no" only.
                1. {question}
                """
            },
            {
                "type": "image_url",
                "image_url": {"url": f"http://10.204.100.119/worameth/DiffSpatial2/{img_path}"}           
            },
        ],
    }]

# Answer the following questions using the given image. List the objects' names (separated by ",") for the 1st and 2nd questions on separate lines without explanations. Each object should be fully visible after editing and not already included in the original image.

#     Suggest 5 objects that can be placed in only one location.

#     Suggest 5 objects that can be placed in multiple specific locations.

# Answer "None" if inserting any objects is implausible.

# ############

# Answer the following questions using the given image. List the objects' names for the 1st and 2nd questions in the 1st and 2nd lines (separated by commas) before providing the full explanation below.

# 1. Suggest 5 objects that can be edited into this image. Each object should be fully visible after editing and can be placed in only one location. It should not already be included in the original image.

# 2. Suggest 5 objects that can be edited into this image. Each object should be fully visible after editing and can be placed in multiple locations, but not just anywhere. It should not already be included in the original image.

# You may answer "None" if it is implausible to insert any objects.

# ############

# Answer the following questions using the given image. List the objects' names for the 1st and 2nd questions in the 1st and 2nd lines (separated by commas) without explanations. Each object should be fully visible after editing and not already included in the original image.

# 1. Suggest 5 objects that can be edited into this image and placed in only one location
# 2. Suggest 5 objects that can be edited into this image and placed in multiple locations, but not just anywhere

# Answer "None" if it is implausible to insert any objects.

QUESTIONS = [
    "Is there at least one focal point or subject in this image?",
    "Does the image depict something abstract?",
    "Was the image captured with a bird's-eye view?",
    "Was the image captured with a close-up shot?",
    "Was the image captured with a macro shot?",
    "Is the camera angle looking up at the sky?",
]

llm_model="llava:7b-v1.6"
# paths = natsorted(list(glob.glob("collocation-mturk/data_preprocessed/**/*")))

import pickle
with open("collocation-mturk/missing_paths.pickle", "rb") as f:
    paths = pickle.load(f)

print(f"Total: {len(paths)}")

df = []
for i, path in tqdm(enumerate(paths), total=len(paths)):
    result = {"id": i, "source": Path(path).parts[-2], "img_path": path}
    for idx, question in enumerate(QUESTIONS):
        messages = prepare_message(question, path)
        response = completion(
            model=f"ollama/{llm_model}", 
            messages=messages, 
            api_base='http://10.204.100.191:11434',
            max_tokens=5000,
            temperature = 0.2,
            # top_p = 0.1
        )
        result[idx] = response.choices[0].message.content
    df.append(result)

    if i % 100 == 0:
        temp_df = pd.DataFrame(df)
        temp_df.to_csv("./collocation-mturk/llava_result2.csv", index=False)

temp_df = pd.DataFrame(df)
temp_df.to_csv("./collocation-mturk/llava_result2.csv", index=False)