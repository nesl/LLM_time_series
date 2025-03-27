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

client = None
prompt_config_har = None

def save_result(batch_results: list, result_save_path: str):
    batch_df = pd.DataFrame(batch_results)
    file_exists = os.path.isfile(result_save_path)
    batch_df.to_csv(result_save_path, index=False, header=not file_exists, mode="a")


def time_text_classify(model, input_repre, result_save_path, data_num, location, freq):
    """ Input representations: time_text, time_text_fewshot, time_text_description """
    data = pd.read_csv(f'./datasets/SHLDataset_preview_v1/User1/220617/{location}_Motion.txt', sep=' ', header=None)
    labels_df = pd.read_csv(f'./datasets/SHL_processed/User1/220617/{location}_video/{location}_IMU_labels.csv', index_col='chunk_index')
    # no2label = ['Null', 'Still', 'Walking', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
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

    ### TEMP: start from 60
    sampled_indices = sampled_indices[90:]
    print(sampled_indices)
    # input("Press Enter...")
    #######################

    # print("Sampled indices:\n", sampled_indices)
    # print("fewshot_dict:", fewshot_dict)
    # input("Press Enter...")

    # Iterate over each row in the data
    batch_results = [] # Write into result file for each 10 results
    
    for idx in sampled_indices:  # DURATION seconds for each segment
        start_timestamp, end_timestamp = labels_df.loc[idx, 'start_timestamp'], labels_df.loc[idx, 'end_timestamp']
        start_index = data.index[data[0] == start_timestamp]
        start_index = start_index.item()
        end_index = data.index[data[0] == end_timestamp]
        end_index = end_index.item()

        """ !!! Modify shl_process.py!!! """
        # if data.loc[start_index:end_index+1, 'label'].nunique() > 1:
        #     print(f"Skipping segment {start_index}-{end_index} due to multiple labels")
        #     continue
        
        data_segment = data.iloc[start_index:end_index+1:(100 // freq), 1:10] # IMU data. Use 100 // freq to downsample to `freq`. Use `1:10` to extract IMU data in column 2-10.
        formatted_data = data_segment.apply(lambda col: ' '.join(col.astype(str)), axis=0)
        
        prompt = ""
        if input_repre == "time_text":
            prompt = prompt_config_har.get_time_text_prompt(imu_data=formatted_data, model=model)
        elif input_repre == "time_text_fewshot":
            prompt = prompt_config_har.get_time_text_fewshot_prompt(fewshot_data_dict=fewshot_dict, imu_data=formatted_data, model=model)
        elif input_repre == "time_text_description":
            # Get description before classification for time_text_description
            prompt_desc = prompt_config_har.get_time_text_description_step1_prompt(imu_data=formatted_data)
            # print("prompt for description:", prompt_desc)
            # input("Press Enter...")
            retry_cnt = 0
            err = ""
            while retry_cnt < 3:
                try:
                    imu_desc = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_desc}],
                    )
                    imu_desc = imu_desc.choices[0].message.content
                    break
                    # print("DESCRIPTION:", imu_desc)
                    # input("Press Enter...")
                except Exception as e:
                    err = e
                    print(f"Trail {retry_cnt} failed: {e}. Retry...")
                    retry_cnt += 1
                    time.sleep(2)
            if retry_cnt == 3:
                raise RuntimeError(f"Failed after 3 trials: {err}")
            
            prompt = prompt_config_har.get_time_text_description_step2_prompt(imu_desc=imu_desc, imu_data=formatted_data, model=model)
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
                    response = client.chat.completions.create(
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
            response = client.chat.completions.create(
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

        ## debug
        # batch_results_keys = set()
        # for entry in batch_results:
        #     for key in entry:
        #         batch_results_keys.add(key)
        # print("batch_results:", batch_results_keys)
        ########

        print(f"Chunk index: {idx}, Timestamp range: {start_timestamp}-{end_timestamp}, Predicted label: {predicted_label}, True label: {true_label}")
        # input("Press Enter to continue...")

        # Write to results file for every 10 data segment
        if len(batch_results) >= 5:
            save_result(batch_results, result_save_path)
            batch_results = []

    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)


def time_plot_classify(model, input_repre, result_save_path, data_num, location, freq):
    """ Input Representations: "time_plot", "time_plot_few_shot", "time_plot_env" """
    pass

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=["gpt-4o", "o1-2024-12-17", "deepseek-reasoner"], required=True, help="Model (LLM) for this task")
    parser.add_argument('-i', '--input',
        choices=[
            "time_text",
            "time_text_fewshot",
            "time_text_description",
            "time_plot",
            "time_plot_few_shot",
            "time_plot_env"
        ], 
        required=True, help="Input representation. Choose from ['time_text', 'time_text_fewshot, 'time_text_description', 'time_plot', 'time_plot_few_shot', 'time_plot_env']"
    )
    parser.add_argument('-dm', '--data-num', type=int, default=30, help="Number of test data for each class")
    parser.add_argument('-l', '--location', type=str, default='Torso', help="Location of IMU data collection smartphone.")
    parser.add_argument('-f', '--frequency', type=int, help="sample Frequency (unit: Hz). Default to 10 if --input is `time_text_fewshot`, otherwise default to 100.")
    parser.add_argument('-r', '--result-save-filename', type=str, default=None, help="File name to save results.")
    args = parser.parse_args()

    # Set default sample frequency
    if args.frequency is None:
        if args.input in ["time_text_fewshot", "time_text_description"]:
            args.frequency = 10
        else:
            args.frequency = 100

    if args.model == "deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    prompt_config_har = PromptConfigHAR(location=args.location, freq=args.frequency)
    model_abbr = {
        "gpt-4o": "gpt4o",
        "o1-2024-12-17": "gpto1",
        "deepseek-reasoner": "dsr1"
    }
    if args.result_save_filename is None:
        args.result_save_filename = f"results_{model_abbr[args.model]}_User1_220617_{args.data_num*4}_{args.location}_{args.input}_4class.csv"
    result_save_path = os.path.join("./results/HAR", args.result_save_filename)
    
    ## debug
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    ########
    
    if args.input in ["time_text", "time_text_fewshot", "time_text_description"]:
        time_text_classify(model=args.model, input_repre=args.input, result_save_path=result_save_path, data_num=args.data_num, location=args.location, freq=args.frequency)
    elif args.input in ["time_plot", "time_plot_few_shot", "time_plot_env"]:
        time_plot_classify(model=args.model, input_repre=args.input, result_save_path=result_save_path, location=args.location, freq=args.freq)
    else:
        raise ValueError(f"{args.input} is not supported")
