import os
import re
from typing import Dict, Union, Tuple, Callable, Any
from openai import OpenAI
import pandas as pd
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
import base64
from PIL import Image
import json
import argparse
from config.prompt_config import PromptConfigDTMF

# OpenAI API Key (Ensure you set this in your environment variables)
CLIENT = None
PROMPT_CONFIG_DTMF = PromptConfigDTMF()
DATA_FOLDER = "/home/yihan/code/datasets/dtmf/"

def save_result(batch_results: list, result_save_path: str):
    batch_df = pd.DataFrame(batch_results)
    file_exists = os.path.isfile(result_save_path)
    batch_df.to_csv(result_save_path, index=False, header=not file_exists, mode="a")

def freq_text(args):
    if args.guide:
        args.result_save_filename += "_guide"
    result_save_path = os.path.join("./results/dtmf", f"{args.result_save_filename}.csv")
    print(f"Result file will be saved in file {result_save_path}")

    batch_results = []  # Save results in batch
    df_test_data = pd.read_csv(os.path.join(DATA_FOLDER, "../single_digit_plot_filename.csv")) # Testing dataset. Each data contains a single key. 20 samples for each class.
    for filename_prefix in df_test_data["filename_prefix"]:
        pattern = re.compile(r"^(.{6})_tone(\d+)$")
        match = pattern.match(filename_prefix)
        if match:
            phone_number = match.group(1)
            tone_index = int(match.group(2))
            true_number = phone_number[tone_index - 1]
        else:
            raise ValueError(f"Cannot match digit from{filename_prefix}")
        
        # Construct Frequencies Template
        file_path = os.path.join(DATA_FOLDER, f"{phone_number}_freqs.csv")
        df = pd.read_csv(file_path)
        df = df[(df["Tone"] == tone_index - 1) & (df["Frequency (Hz)"] >= 0) & (df["Frequency (Hz)"] <= 2000)]
        freq_mag_pairs = [f"{row['Frequency (Hz)']:.2f}:{row['Magnitude']:.2f}" for _, row in df.iterrows()]
        freq_mag_text = ", ".join(freq_mag_pairs)

        prompt = PROMPT_CONFIG_DTMF.get_freq_text_prompt(freq_mag_text=freq_mag_text, model=args.model, enable_guide=args.guide)

        if args.model == "deepseek-reasoner":
            # DeepSeek-R1 doesn't support Json Output. Prompt the model output the key in the format of <<KEY>> at the first word, followed by detailed analysis.
            response = CLIENT.chat.completions.create(
                model=args.model,
                messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }],
            )
            rsp = response.choices[0].message.content
            match = re.search(r"<<([\d\s*#]+)>>(?:, (\[[\d.,\s]+\]))?", rsp)
            result = {
                "key": match.group(1) if match else "",
                "frequencies": match.group(2) if match and match.group(2) else [],
                "analysis": rsp
            }
        else:
            response = CLIENT.chat.completions.create(
                    model=args.model,
                    response_format={"type": "json_object"},
                    messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }],
                )

            result = response.choices[0].message.content
            try:
                result = json.loads(result)
            except Exception as e:
                print(f"{filename_prefix} result json parsing failed: {e}")
                result = {
                    "analysis": f"ERROR: {e}. Raw Response: {result}",
                    "frequencies": [],
                    "key": None
                }

        # Print and store results
        # print("RESULT:", result)
        # print(f"Filename prefix: {filename_prefix}, Predicted frquencies: {result.get('frequencies', [])}, Predicted number: {result.get('key', None)}, True number: {true_number}")

        batch_results.append({
            "filename_prefix": filename_prefix,
            "true_number": true_number,
            "predicted_number": result.get("key", None),
            "frequencies": result.get("frequencies", []),
            "raw_response": result.get("analysis", "No analysis responded")
        })

        # Write results to file every 10 entries
        if len(batch_results) >= 10:
            save_result(batch_results, result_save_path)
            batch_results = []
    
    # Save remain results
    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)

def freq_plot(args):
    if args.model == "deepseek-reasoner":
        raise ValueError("DeepSeek-R1 does not support image input. Please use GPT-4o or GPT-o1 instead.")
    
    if args.grid:
        args.result_save_filename += "_grid"
    if args.guide:
        args.result_save_filename += "_guide"
    result_save_path = os.path.join("./results/dtmf", f"{args.result_save_filename}.csv")
    print(f"Result file will be saved in file {result_save_path}")

    df = pd.read_csv(os.path.join(DATA_FOLDER, "../single_digit_plot_filename.csv"))
    batch_results = []  # Save results in batch

    img_suffix = "freqs_grid" if args.grid else "freqs"
    for filename_prefix in df["filename_prefix"]:
        img_path = os.path.join(DATA_FOLDER, f"{filename_prefix}_{img_suffix}.png")
        pattern = re.compile(r"^(.{6})_tone(\d+)$")
        match = pattern.match(filename_prefix)
        if match:
            phone_number = match.group(1)
            tone_index = match.group(2)
            true_number = phone_number[int(tone_index) - 1]
        else:
            raise ValueError(f"Cannot match key from{filename_prefix}")
        
        # # Digit only
        # if not true_number.isdigit():
        #     print(f"{filename_prefix} is {true_number}, not a digit. Skip...")
        #     continue
        
        # Construct the LLM prompt
        prompt = PROMPT_CONFIG_DTMF.get_freq_plot_prompt(enable_guide=args.guide)

        # Call LLM
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            response = CLIENT.chat.completions.create(
                model=args.model,
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
                            ]
                        }],
            )

            result = response.choices[0].message.content
            result = json.loads(result)
            # print("RESULT:", result)
            # print(f"Filename prefix: {filename_prefix}, Predicted number: {result.get('key', None)}, Frequencies: {result.get('frequencies', [])}, True number: {true_number}")

            batch_results.append({
                "filename_prefix": filename_prefix,
                "true_number": true_number,
                "predicted_number": result.get("key", None),
                "frequencies": result.get("frequencies", []),
                "raw_response": result.get("analysis", "No analysis responded")
            })

            # Write results to file every 10 entries
            if len(batch_results) >= 10:
                save_result(batch_results, result_save_path)
                batch_results = []
    
    # Save remain results (if any)
    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)

def freq_pair(args):
    if args.map:
        args.result_save_filename += "_map"
    result_save_path = os.path.join("./results/dtmf", f"{args.result_save_filename}.csv")
    print(f"Result file will be saved in file {result_save_path}")

    batch_results = []  # Save results in batch

    df_test_data = pd.read_csv(os.path.join(DATA_FOLDER, "../single_digit_plot_filename.csv"))
    DTMF_FREQUENCIES = {
        "1": [697, 1209], "2": [697, 1336], "3": [697, 1477],
        "4": [770, 1209], "5": [770, 1336], "6": [770, 1477],
        "7": [852, 1209], "8": [852, 1336], "9": [852, 1477],
        "*": [941, 1209], "0": [941, 1336], "#": [941, 1477]
    }
    for filename_prefix in df_test_data["filename_prefix"]:
        pattern = re.compile(r"^(.{6})_tone(\d+)$")
        match = pattern.match(filename_prefix)
        if match:
            phone_number = match.group(1)
            tone_index = int(match.group(2))
            true_number = phone_number[tone_index - 1]
        else:
            raise ValueError(f"Cannot match digit from{filename_prefix}")
        
        # Construct Frequencies Pair Template
        if args.noise_type == "noise":
            file_path = os.path.join(DATA_FOLDER, f"{phone_number}_freqs_pair.csv")
            df = pd.read_csv(file_path)
            row = df.iloc[tone_index - 1]  # Adjust for zero-based index
            freq_pair = f"{row['Freq1']} Hz, {row['Freq2']} Hz"
            
        else:
            freq_pair = f"{DTMF_FREQUENCIES[true_number][0]} Hz, {DTMF_FREQUENCIES[true_number][1]} Hz"
        
        prompt = PROMPT_CONFIG_DTMF.get_freq_pair_prompt(freq_pair=freq_pair, model=args.model, enable_map=args.map)

        # Call LLM
        if args.model == "deepseek-reasoner":
            response = CLIENT.chat.completions.create(
                model=args.model,
                messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }],
            )
            rsp = response.choices[0].message.content
            match = re.search(r"<<([\d\s*#]+)>>", rsp)
            result = {
                "key": match.group(1) if match else "",
                "analysis": rsp
            }
        else:
            response = CLIENT.chat.completions.create(
                    model=args.model,
                    response_format={"type": "json_object"},
                    messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }],
                )

            result = response.choices[0].message.content
            try:
                result = json.loads(result)
            except Exception as e:
                print(f"{filename_prefix} result json parsing failed: {e}")
                result = {"analysis": f"ERROR: {e}. Raw Response: {result}", "key": None}

        # Print and store results
        # print("RESULT:", result)
        # print(f"Filename prefix: {filename_prefix}, Predicted number: {result.get('key', None)}, True number: {true_number}")

        batch_results.append({
            "filename_prefix": filename_prefix,
            "true_number": true_number,
            "predicted_number": result.get("key", None),
            "raw_response": result.get("analysis", "No analysis responded")
        })

        # Write results to file every 10 entries
        if len(batch_results) >= 10:
            save_result(batch_results, result_save_path)
            batch_results = []
    
    # Save remain results (if any)
    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)

if __name__ == "__main__":
    # shared parameters (inherited by main and all subparsers)
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-m', '--model', required=True, choices=["gpt-4o", "o1-2024-12-17", "deepseek-reasoner"], help="Model (LLM) for this task")
    base_parser.add_argument('-n', '--noise-type', required=True, choices=['noise', 'clean'], help="Choose to input tones with or with out noise. Choose from ['noise', 'clean']")
    base_parser.add_argument('-r', '--result-save-filename', type=str, default=None, help="File name to save results.")

    # main parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # freq_text
    parser_text = subparsers.add_parser("freq_text", parents=[base_parser], help="Run experiments for frequency-magnitude series in raw text")
    parser_text.add_argument('-g', '--guide', action='store_true', help="Add step-bystep guidance or not")
    parser_text.set_defaults(func=freq_text)

    # freq_plot
    parser_plot = subparsers.add_parser("freq_plot", parents=[base_parser], help="Run experiments for frequency-magnitude series line plots")
    parser_plot.add_argument('-g', '--guide', action='store_true', help="Add step-bystep guidance or not")
    parser_plot.add_argument('-gr', '--grid', action='store_true', help="Whether to input plots with grid or not")
    parser_plot.set_defaults(func=freq_plot)

    # freq_pair
    parser_pair =subparsers.add_parser("freq_pair", parents=[base_parser], help="Run experiments given low- and high-frequency pair")
    parser_pair.add_argument('-map', '--map', action='store_true', help="Add true DTMF freqwuency-key map or not")
    parser_pair.set_defaults(func=freq_pair)

    args = parser.parse_args()
    
    DATA_FOLDER = os.path.join(DATA_FOLDER, f"dtmf_{args.noise_type}_12")

    if args.model == "deepseek-reasoner":
        CLIENT = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        CLIENT = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    PROMPT_CONFIG_DTMF = PromptConfigDTMF()
    model_abbr = {
        "gpt-4o": "gpt4o",
        "o1-2024-12-17": "gpto1",
        "deepseek-reasoner": "dsr1"
    }
    if args.result_save_filename is None:
        args.result_save_filename = f"TEST_results_{model_abbr[args.model]}_{args.noise_type}_{args.func.__name__}"
    print("args.result_save_filename:", args.result_save_filename)
    args.func(args)
