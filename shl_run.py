import os
import re
from openai import OpenAI
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import argparse
from config.prompt_config import PromptConfigHAR
import time
import base64
from PIL import Image

CLIENT = None
PROMPT_CONFIG_HAR = None
DATA_FOLDER = None

def save_result(batch_results: list, result_save_path: str):
    batch_df = pd.DataFrame(batch_results)
    file_exists = os.path.isfile(result_save_path)
    batch_df.to_csv(result_save_path, index=False, header=not file_exists, mode="a")


def time_text_classify(model, input_repre, result_save_path, data_num, location, freq):
    """ Input representations: time_text, time_text_fewshot, time_text_description """
    data = pd.read_csv(os.path.join(DATA_FOLDER, f'{location}_Motion.txt'), sep=' ', header=None)
    labels_df = pd.read_csv(os.path.join(DATA_FOLDER, f'{location}_IMU_labels.csv'), index_col='chunk_index')
    sampled_indices = []
    fewshot_dict = {}

    # For simplicity, only classify four labels: Still, Walking, Run, Car
    for label in ['Still', 'Walking', 'Run', 'Car']:
        filtered_indices = labels_df[labels_df['label_name'] == label].index[:data_num].tolist()
        sampled_indices.extend(filtered_indices)
        if input_repre == "time_text_fewshot":
            fewshot_idx = labels_df[labels_df['label_name'] == label].index[data_num]
            start_index = data.index[data[0] == labels_df.loc[fewshot_idx, 'start_timestamp']]
            start_index = start_index.item()
            end_index = data.index[data[0] == labels_df.loc[fewshot_idx, 'end_timestamp']]
            end_index = end_index.item()
            data_segment = data.iloc[start_index:end_index+1:(100 // freq), 1:10] # IMU data
            formatted_data = data_segment.apply(lambda col: ' '.join(col.astype(str)), axis=0)
            fewshot_dict[label] = formatted_data

    # Iterate over each row in the data
    batch_results = [] # Write into result file for each 10 results
    
    for idx in sampled_indices:  # DURATION seconds for each segment
        start_timestamp, end_timestamp = labels_df.loc[idx, 'start_timestamp'], labels_df.loc[idx, 'end_timestamp']
        start_index = data.index[data[0] == start_timestamp]
        start_index = start_index.item()
        end_index = data.index[data[0] == end_timestamp]
        end_index = end_index.item()

        data_segment = data.iloc[start_index:end_index+1:(100 // freq), 1:10] # IMU data. Use 100 // freq to downsample to `freq`. Use `1:10` to extract IMU data in column 2-10.
        formatted_data = data_segment.apply(lambda col: ' '.join(col.astype(str)), axis=0)
        
        prompt = ""
        if input_repre == "time_text":
            prompt = PROMPT_CONFIG_HAR.get_time_text_prompt(imu_data=formatted_data, model=model)
        elif input_repre == "time_text_fewshot":
            prompt = PROMPT_CONFIG_HAR.get_time_text_fewshot_prompt(fewshot_data_dict=fewshot_dict, imu_data=formatted_data, model=model)
        elif input_repre == "time_text_description":
            # Get description before classification for time_text_description
            prompt_desc = PROMPT_CONFIG_HAR.get_time_text_description_step1_prompt(imu_data=formatted_data)

            retry_cnt = 0
            err = ""
            while retry_cnt < 3:
                try:
                    imu_desc = CLIENT.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_desc}],
                    )
                    imu_desc = imu_desc.choices[0].message.content
                    break
                except Exception as e:
                    err = e
                    print(f"Trail {retry_cnt} failed: {e}. Retry...")
                    retry_cnt += 1
                    time.sleep(2)
            if retry_cnt == 3:
                raise RuntimeError(f"Failed after 3 trials: {err}")
            
            prompt = PROMPT_CONFIG_HAR.get_time_text_description_step2_prompt(imu_desc=imu_desc, imu_data=formatted_data, model=model)
        else:
            raise ValueError(f"{input_repre} is not supported in function `time_text_classify()`")

        # print(f"prompt for {input_repre}: {prompt}")
        # input("Enter Press...")

        if model == "deepseek-reasoner":
            # use raw text response format because deepseek-reasoner doesn't support json output
            retry_cnt = 0
            err = ""
            while retry_cnt < 3:
                try:
                    response = CLIENT.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except Exception as e:
                    err = e
                    print(f"Trail {retry_cnt} failed: {e}. Retry...")
                    retry_cnt += 1
                    time.sleep(2)
            if retry_cnt == 3:
                raise RuntimeError(f"Failed after 3 trials: {err}")
            
            result = response.choices[0].message.content
            match = re.search(r'<<([^<>]+)>>', result)
            predicted_label = match.group(1) if match else ""
            analysis = result
        else:
            response = CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content
            try:
                result = json.loads(result)
                predicted_label = result.get('label', "")
                analysis = result.get('analysis', result)
            except Exception as e:
                print(f"ERROR: {filename_prefix} result json parsing failed: {e}")
                predicted_label = ""
                analysis = f"ERROR: {e}. Raw Response: {result}"

        # print("Raw result:", result)

        true_label = labels_df.loc[idx, 'label_name']
        batch_results.append({
            'chunk_index': idx,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'analysis': analysis,
            **({'description': imu_desc} if input_repre == "time_text_description" else {})
        })

        print(f"Chunk index: {idx}, Timestamp range: {start_timestamp}-{end_timestamp}, Predicted label: {predicted_label}, True label: {true_label}")
        # input("Press Enter to continue...")

        # Write to results file for every 10 data segment
        if len(batch_results) >= 5:
            save_result(batch_results, result_save_path)
            batch_results = []

    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def time_plot_classify(model, input_repre, result_save_path, data_num, location, freq):
    """ Input Representations: "time_plot", "time_plot_fewshot", "time_plot_env" """
    data = pd.read_csv(os.path.join(DATA_FOLDER, f'{location}_Motion.txt'), sep=' ', header=None)
    labels_df = pd.read_csv(os.path.join(DATA_FOLDER, f'{location}_IMU_labels.csv'), index_col='chunk_index')
    sampled_indices = []
    fewshot_dict = {}

    # For simplicity, only classify four labels: Still, Walking, Run, Car
    for label in ['Still', 'Walking', 'Run', 'Car']:
        filtered_indices = labels_df[labels_df['label_name'] == label].index[:data_num].tolist()
        sampled_indices.extend(filtered_indices)
        if input_repre == "time_plot_fewshot":
            fewshot_idx = labels_df[labels_df['label_name'] == label].index[data_num]
            fewshot_dict[label] = os.path.join(DATA_FOLDER, f'{fewshot_idx}_{location}_IMU_plot.png')

    # print("Sampled indices:\n", sampled_indices)
    # print("fewshot_dict:", fewshot_dict)
    # input("Press Enter...")

    # Iterate over each row in the data
    batch_results = [] # Write into result file for each 10 results
    
    for idx in sampled_indices:  # DURATION seconds for each segment
        img_path = os.path.join(DATA_FOLDER, f'{idx}_{location}_IMU_plot.png')
        base64_image = encode_image(img_path)
        
        prompt = ""
        if input_repre == "time_plot":
            prompt = PROMPT_CONFIG_HAR.get_time_plot_prompt()
            response = CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }],
            )
        elif input_repre == "time_plot_fewshot":
            # Get fewshot examples for each class
            still_sample_img = encode_image(fewshot_dict['Still'])
            walking_sample_img = encode_image(fewshot_dict['Walking'])
            run_sample_img = encode_image(fewshot_dict['Run'])
            car_sample_img = encode_image(fewshot_dict['Car'])
            
            prompt = PROMPT_CONFIG_HAR.get_time_plot_fewshot_prompt()
            response = CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{still_sample_img}"},
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{walking_sample_img}"},
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{run_sample_img}"},
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{car_sample_img}"},
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                    },
                                ]
                            }
                        ],
            )
        elif input_repre == "time_plot_env":
            # Get environment photo
            file_path_video = os.path.join(DATA_FOLDER, f'{idx}_video.png')
            base64_image_env = encode_image(file_path_video)
            
            prompt = PROMPT_CONFIG_HAR.get_time_plot_env_prompt()
            response = CLIENT.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_image_env}"},
                                    },
                                ]
                            }
                        ],
            )
        else:
            raise ValueError(f"{input_repre} is not supported in function `time_plot_classify()`")

        # print(f"PROMPT FOR {input_repre}: {prompt}")
        # input("Enter Press...")

        result = response.choices[0].message.content
        try:
            result = json.loads(result)
            predicted_label = result.get('label', "")
            analysis = result.get('analysis', result)
        except Exception as e:
            print(f"ERROR: {filename_prefix} result json parsing failed: {e}")
            predicted_label = ""
            analysis = f"ERROR: {e}. Raw Response: {result}"

        # print("Raw result:", result)

        true_label = labels_df.loc[idx, 'label_name']
        batch_results.append({
            'chunk_index': idx,
            'start_timestamp': labels_df.loc[idx, 'start_timestamp'],
            'end_timestamp': labels_df.loc[idx, 'end_timestamp'],
            'true_label': true_label,
            'predicted_label': predicted_label,
            'analysis': analysis
        })

        print(f"Chunk index: {idx}, Timestamp range: {labels_df.loc[idx, 'start_timestamp']}-{labels_df.loc[idx, 'end_timestamp']}, Predicted label: {predicted_label}, True label: {true_label}")
        # input("Press Enter to continue...")

        # Write to results file for every 10 data segment
        if len(batch_results) >= 5:
            save_result(batch_results, result_save_path)
            batch_results = []

    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=["gpt-4o", "o1-2024-12-17", "deepseek-reasoner"], required=True, help="Model (LLM) for this task")
    parser.add_argument('-i', '--input',
        choices=[
            "time_text",
            "time_text_fewshot",
            "time_text_description",
            "time_plot",
            "time_plot_fewshot",
            "time_plot_env"
        ], 
        required=True, help="Input representation"
    )
    parser.add_argument('-dn', '--data-num', type=int, default=30, help="Number of test data for each class")
    parser.add_argument('-df', '--data-folder', type=str, default='./datasets/SHL_processed/User1/220617/Torso_video/', help="Data folder path.")
    parser.add_argument('-l', '--location', type=str, default='Torso', help="Location of IMU data collection smartphone.")
    parser.add_argument('-f', '--frequency', type=int, help="sample Frequency (unit: Hz). Default to 10 if --input is `time_text_fewshot`, otherwise default to 100.")
    parser.add_argument('-r', '--result-save-filename', type=str, default=None, help="File name to save results.")
    args = parser.parse_args()

    DATA_FOLDER = args.data_folder
    # Set default sample frequency
    if args.frequency is None:
        if args.input in ["time_text_fewshot", "time_text_description"]:
            args.frequency = 10
        else:
            args.frequency = 100

    if args.model == "deepseek-reasoner":
        CLIENT = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        CLIENT = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    PROMPT_CONFIG_HAR = PromptConfigHAR(location=args.location, freq=args.frequency)
    model_abbr = {
        "gpt-4o": "gpt4o",
        "o1-2024-12-17": "gpto1",
        "deepseek-reasoner": "dsr1"
    }
    if args.result_save_filename is None:
        args.result_save_filename = f"TEST_results_{model_abbr[args.model]}_User1_220617_{args.data_num*4}_{args.location}_{args.input}_4class.csv"
    result_save_path = os.path.join("./results/HAR", args.result_save_filename)
    
    ## debug
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    ########
    
    if args.input in ["time_text", "time_text_fewshot", "time_text_description"]:
        time_text_classify(model=args.model, input_repre=args.input, result_save_path=result_save_path, data_num=args.data_num, location=args.location, freq=args.frequency)
    elif args.input in ["time_plot", "time_plot_fewshot", "time_plot_env"]:
        time_plot_classify(model=args.model, input_repre=args.input, result_save_path=result_save_path, data_num=args.data_num, location=args.location, freq=args.frequency)
    else:
        raise ValueError(f"{args.input} is not supported")
