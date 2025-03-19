import json
import pandas as pd
import os
from pathlib import Path
import numpy as np
import argparse

def extract_data(outputs):
    df = []
    for output in outputs:
        df.append({
            "id": output["id"],
            "custom_id": output["custom_id"],
            "status_code": output["response"]["status_code"],
            "model": output["response"]["body"]["model"],
            "prompt_tokens": output["response"]["body"]["usage"]["prompt_tokens"],
            "completion_tokens": output["response"]["body"]["usage"]["completion_tokens"],
            "answer": output["response"]["body"]["choices"][0]["message"]["content"],
        })

    return pd.DataFrame(df)

def parse_answer_select(text):
    return text.lower().split(', ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=int, required=True)
    args = parser.parse_args()

    # outputs = []
    # for path in paths:
    #     outputs += ([json.loads(i) for i in open(path).readlines()])
    outputs = [json.loads(i) for i in open(args.input_path).readlines()]
    df = extract_data(outputs)

    name_mapping = lambda x: f'answer_{x % 5}'
    parsed_columns = df['answer'].apply(parse_answer_select)
    parsed_df = pd.DataFrame(parsed_columns.tolist(), columns=[name_mapping(i) for i in range(5)])
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    input_df = pd.read_csv(args.df_path)
    input_df["custom_id"] = input_df.index
    df = input_df.join(df, on="custom_id", how="inner", lsuffix="_left", rsuffix="_right")
    columns_to_drop = [col for col in df.columns if col.endswith('_left')]
    df.drop(columns=columns_to_drop, inplace=True)
    df.columns = [col.replace('_right', '') for col in df.columns]

    save_path = args.save_path
    os.makedirs(Path(save_path).parent, exist_ok=True)
    df.to_csv(save_path, index=False)