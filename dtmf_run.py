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
client = None
prompt_config_dtmf = PromptConfigDTMF()
data_folder = "/home/yihan/code/datasets/dtmf/"

def save_result(batch_results: list, result_save_path: str):
    batch_df = pd.DataFrame(batch_results)
    file_exists = os.path.isfile(result_save_path)
    batch_df.to_csv(result_save_path, index=False, header=not file_exists, mode="a")

def freq_text(args):
    if args.result_save_filename is None:
        # if args.grid:
        #     args.result_save_filename += "_grid"
        # if args.map:
        #     args.result_save_filename += "_map"
        if args.guide:
            args.result_save_filename += "_guide"
        args.result_save_filename += ".csv"
    result_save_path = os.path.join("./results/HAR", args.result_save_filename)
    print(f"Result file will be saved in file {args.result_save_filename}")

    batch_results = []  # Write results to file every 10 entries
    df_test_data = pd.read_csv(os.path.join(data_folder, "../single_digit_plot_filename.csv")) # Testing dataset. Each data contains a single key. 20 samples for each class.
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
        file_path = os.path.join(data_folder, f"{phone_number}_freqs.csv")
        df = pd.read_csv(file_path)
        df = df[(df["Tone"] == tone_index - 1) & (df["Frequency (Hz)"] >= 0) & (df["Frequency (Hz)"] <= 2000)]
        freq_mag_pairs = [f"{row['Frequency (Hz)']:.2f}:{row['Magnitude']:.2f}" for _, row in df.iterrows()]
        freq_mag_text = ", ".join(freq_mag_pairs)

        prompt = prompt_config_dtmf.get_freq_text_prompt(freq_mag_text=freq_mag_text, model=args.model, enable_guide=args.guide)
        print(f"PROMPT: {prompt}")
        input("Press Enter...")

        if model == "deepseek-reasoner":
            # DeepSeek-R1 doesn't support Json Output. Prompt the model output the key in the format of <<KEY>> at the first word, followed by detailed analysis.

            # # Prompt (unguide)
            # prompt = f"""
            # You are an expertise in dual tone multi-frequency decoding. You are given a series of frequency components with corresponding amplitudes in the format of "frequency1:magnitude1, frequency2:magnitude2, ...". The frequency series represents a key (0-9, * or #). Please identify the key it refers to. Here is the frequency series:  {freq_mag_text} \n \
            # Please provide your recognized key in the format of <<KEY>> at the first word of your response, followed by your detailed analysis. For example, if you recognized the provided frequency series as key "6", then the first word of your response should be <<6>>. 
            # """

            # # Prompt (guide)
            # prompt = f"""
            # You are an expertise in dual tone multi-frequency decoding. You are given a series of frequency components with corresponding amplitudes in the format of "frequency1:magnitude1, frequency2:magnitude2, ...". The frequency series represents a key (0-9, * or #). Please first recognize the frequencies with high magnitudes in this series. Then, identify the key it refers to. Here is the frequency series:  {freq_mag_text} \n \
            # Please provide your recognized key in the format of <<KEY>> at the first word of your response, followed by a list of high-magnitude frequencies as floating-point numbers rounded to two decimal places, and then followed by your detailed analysis. For example, if you detected the magnitude of frequencies 770Hz, 1477Hz are high and recognized the provided frequency series as key "6", then your response should begin with "<<6>>, [770.00, 1477.00]", followed by deyailed analysis.
            # """
            response = client.chat.completions.create(
                model=model,
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
            # # Prompt (unguide)
            # prompt = f"""
            # You are an expertise in dual tone multi-frequency decoding. You are given a series of frequency components with corresponding amplitudes in the format of "frequency1:magnitude1, frequency2:magnitude2, ...". The frequency series represents a key (0-9, * or #). Please identify the key it refers to. Here is the frequency series:  {freq_mag_text} \n \
            # Please provide your answer in the following JSON structure: \n \
            # {{ \n \
            #     "key": "the recognized key", \n \
            #     "analysis": "a detailed explanation of your analysis process." \n \
            # }} \n \
            # """

            # # Prompt (guide)
            # prompt = f"""
            # You are an expertise in dual tone multi-frequency decoding. You are given a series of frequency components with corresponding amplitudes in the format of "frequency1:magnitude1, frequency2:magnitude2, ...". The frequency series represents a key (0-9, * or #). Please first recognize the frequencies with high magnitudes in this series. Then, identify the key it refers to. Here is the frequency series:  {freq_mag_text} \n \
            # Please provide your answer in the following JSON structure: \n \
            # {{ \n \
            #     "frequencies": [list of high-magnitude frequencies as floating-point numbers rounded to two decimal places], \n \
            #     "key": "the recognized key", \n \
            #     "analysis": "a detailed explanation of your analysis process." \n \
            # }} \n \
            # """
            
            response = client.chat.completions.create(
                    model=model,
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
        # print(result)
        print(f"Filename prefix: {filename_prefix}, Predicted frquencies: {result.get('frequencies', [])}, Predicted number: {result.get('key', None)}, True number: {true_number}")
        # input("Press Enter...")

        batch_results.append({
            "filename_prefix": filename_prefix,
            "true_number": true_number,
            "predicted_number": result.get("key", None),
            "frequencies": result.get("frequencies", []),
            "raw_response": result.get("analysis", "No analysis responded")
        })
        # input("Press Enter...")

        # Write results to file every 10 entries
        if len(batch_results) >= 10:
            save_result(batch_results, result_save_path)
            batch_results = []
    
    # Save remain results
    if len(batch_results) > 0:
        save_result(batch_results, result_save_path)

def freq_plot(model, result_save_path):
    if args.result_save_filename is None:
        if args.grid:
            args.result_save_filename += "_grid"
        # if args.map:
        #     args.result_save_filename += "_map"
        if args.guide:
            args.result_save_filename += "_guide"
        args.result_save_filename += ".csv"
    df = pd.read_csv(os.path.join(data_folder, "../single_digit_plot_filename.csv"))
    batch_results = []  # Write results to file every 10 entries

    test_cnt = 0
    for filename_prefix in df["filename_prefix"]:
        img_path = os.path.join(data_folder, f"{filename_prefix}_freqs_grid.png")
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
        prompt = prompt_config_dtmf.get_freq_plot_prompt(enable_guide=args.guide)
        print(f"PROMPT: {prompt}")
        input("Press Enter...")

        # Call LLM
        # result, predict_number, match_cnt = call_llm(model=model, true_number=true_number, prompt=prompt, image_path=img_path, result_pattern=r'<<([^<>]+)>>')

        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            response = client.chat.completions.create(
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
                            ]
                        }],
            )

            result = response.choices[0].message.content
            result = json.loads(result)
            # print(result)
            print(f"Filename prefix: {filename_prefix}, Predicted number: {result.get('key', None)}, Frequencies: {result.get('frequencies', [])}, True number: {true_number}")
            # input("Press Enter...")

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

            # input("Press Enter...")
    # Save remain results
    save_result(batch_results, result_save_path)

def freq_pair(model, result_save_path):
    if args.result_save_filename is None:
        # if args.grid:
        #     args.result_save_filename += "_grid"
        if args.map:
            args.result_save_filename += "_map"
        # if args.guide:
        #     args.result_save_filename += "_guide"
        args.result_save_filename += ".csv"
    batch_results = []  # Write results to file every 10 entries

    test_cnt = 0
    df_test_data = pd.read_csv(os.path.join(data_folder, "../single_digit_plot_filename.csv"))
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
            file_path = os.path.join(data_folder, f"{phone_number}_freqs_clean.csv")
            df = pd.read_csv(file_path)
            row = df.iloc[tone_index - 1]  # Adjust for zero-based index
            freq_pair = f"{row['Freq1']} Hz, {row['Freq2']} Hz"
            
        else:
            freq_pair = f"{DTMF_FREQUENCIES[true_number][0]} Hz, {DTMF_FREQUENCIES[true_number][1]} Hz"
        
        print(f"freq_pair: {freq_pair}")
        input("Press Enter...")
        
        prompt = prompt_config_dtmf.get_freq_pair_prompt(freq_pair=freq_pair, enable_map=args.map)
        # # Prompt (no map)
        # prompt = f"""
        # You are an expertise in dual tone multi-frequency decoding. You are given a frequency pair of a tone which represents a key (0-9, * or #). Please identify the key it refers to. Here is the frequency pair: {freq_pair}
        # Please provide your answer in the following JSON structure:
        # {{
        #     "key": "the recognized key",
        #     "analysis": "a detailed explanation of your analysis process."
        # }}
        # """

        # # Prompt (with map)
        # prompt = f"""
        # You are an expertise in dual tone multi-frequency decoding. You are given a frequency pair of a tone. This tone represents a key (0-9, * or #). Here is the frequency pair: {freq_pair}.
        # Please identify the key it refers to using the map below.
        # | Low Frequency (Hz)  | High Frequency (Hz)| Key  |
        # |---------------------|--------------------|------|
        # | 697                 | 1209               | 1    |
        # | 697                 | 1336               | 2    |
        # | 697                 | 1477               | 3    |
        # | 770                 | 1209               | 4    |
        # | 770                 | 1336               | 5    |
        # | 770                 | 1477               | 6    |
        # | 852                 | 1209               | 7    |
        # | 852                 | 1336               | 8    |
        # | 852                 | 1477               | 9    |
        # | 941                 | 1209               | *    |
        # | 941                 | 1336               | 0    |
        # | 941                 | 1477               | #    |
        # Please provide your answer in the following JSON structure:
        # {{
        #     "key": "the recognized key",
        #     "analysis": "a detailed explanation of your analysis process."
        # }}
        # """
        # print("PROMPT:", prompt)

        # Call LLM
        response = client.chat.completions.create(
                model=model,
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
        # print(result)
        print(f"Filename prefix: {filename_prefix}, Predicted number: {result.get('key', None)}, True number: {true_number}")
        # input("Press Enter...")

        batch_results.append({
            "filename_prefix": filename_prefix,
            "true_number": true_number,
            "predicted_number": result.get("key", None),
            "raw_response": result.get("analysis", "No analysis responded")
        })
        # input("Press Enter...")

        # Write results to file every 10 entries
        if len(batch_results) >= 10:
            save_result(batch_results, result_save_path)
            batch_results = []
    
    # Save remain results
    save_result(batch_results, result_save_path)

if __name__ == "__main__":
    # --- 1. 定义父解析器：所有函数都要有的参数 ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-m', '--model', choices=["gpt-4o", "o1-2024-12-17", "deepseek-reasoner"], help="Model (LLM) for this task")
    parent_parser.add_argument('-n', '--noise-type', required=True, choices=['noise', 'clean'], help="Choose to input tones with or with out noise. Choose from ['noise', 'clean']")
    parent_parser.add_argument('-r', '--result-save-filename', type=str, default=None, help="File name to save results.")

    # main parser and shared parameters
    parser = argparse.ArgumentParser(description="CLI with shared args for all functions")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # freq_text
    parser_text = subparsers.add_parser("freq_text", parents=[parent_parser], help="Run experiments for frequency-magnitude series in raw text")
    parser_text.add_argument('-g', '--guide', action='store_true', help="Add step-bystep guidance or not")
    parser_text.set_defaults(func=func1)

    # freq_plot
    parser_plot = subparsers.add_parser("freq_plot", parents=[parent_parser], help="Run experiments for frequency-magnitude series line plots")
    parser_text.add_argument('-g', '--guide', action='store_true', help="Add step-bystep guidance or not")
    parser_text.add_argument('-gr', '--grid', action='store_true', help="Whether to input plots with grid or not")
    parser_plot.set_defaults(func=func2)

    # freq_pair
    parser_pair =subparsers.add_parser("freq_pair", parents=[parent_parser], help="Run experiments given low- and high-frequency pair")
    args = parser.parse_args()
    data_folder = os.path.join(data_folder, f"dtmf_{args.noise_type}_12")
    model_abbr = {
        "gpt-4o": "gpt4o",
        "o1-2024-12-17": "gpto1",
        "deepseek-reasoner": "dsr1"
    }

    if args.model == "deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )if args.model == "deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    if args.result_save_filename is None:
        args.result_save_filename = f"results_{model_abbr[args.model]}_{args.noise_type}_{args.func}"
    
    args.func(args)
