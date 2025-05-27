import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import argparse
import re

result_folder = "./results/dtmf/"

def cal_acc_overall(result_save_filename):
    ''' Calculate accuracy of predicted keys. Support multiple keys in one response. '''
    df = pd.read_csv(os.path.join(result_folder, f"results_{result_save_filename}.csv"))

    total_digits = 0
    valid_pred_digits = 0
    correct_digits = 0

    true_digits = []
    pred_digits = []

    digit_labels = [str(i) for i in range(1, 10)] + ['*', '0', '#']
    extended_labels = digit_labels + ["Null"]

    null_count = 0  # Count of invalid predicted key
    unmatch_len_cnt = 0 # Count of wrong length of keys (i.e, return less or more keys than groud truth)

    for _, row in df.iterrows():
        true_number = str(row["true_number"])
        predicted_number = re.sub(r'\s+', '', str(row["predicted_number"]))

        # Calculate invalid prediction count before aligning predicted-Number and true_number
        for ch in predicted_number:
            if ch not in digit_labels:
                # print(f"row{_}, ch={ch}, predicted_number={predicted_number}")
                # print(row["raw_response"])
                null_count += 1
                break
        
        if len(true_number) != len(predicted_number):
            unmatch_len_cnt += 1
            predicted_number = predicted_number[:len(true_number)]
            predicted_number = predicted_number.ljust(len(true_number))
        correct_digits += sum(1 for a, b in zip(true_number, predicted_number) if a == b)
        total_digits += len(true_number)

        true_digits.extend(list(true_number))

        # Process invalid predicted keys
        for ch in predicted_number:
            if ch in digit_labels:
                pred_digits.append(ch)
            else:
                pred_digits.append("Null")

    # input("Press Enter")
    accuracy = correct_digits / total_digits
    null_ratio = null_count / total_digits

    print(f"{unmatch_len_cnt} responses length unmatch among {len(df)} testing data")
    print(f"Total digits: {total_digits}, Invalid ('Null') predicted digits count: {null_count} ({null_ratio:.2%})")
    print(f"{args.result_save_filename} DTMF decoding accuracy: {accuracy:.3f}")

    conf_matrix = confusion_matrix(true_digits, pred_digits, labels=extended_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=extended_labels, yticklabels=extended_labels, linewidths=0.5)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Acc: {accuracy:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f'conf_matrix_{args.result_save_filename}_overall.png'))



def cal_acc_freqs_detect(result_save_filename, err_tolerance):
    '''
    Calculate the accuracy of low and high frequency detection.
    Also generate confusion matrices and frequency error plots.
        err_tolerance: bias no more than err_tolerance will be count as correct detection
    '''
    # DTMF standard frequencies
    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477]
    DTMF_FREQUENCIES = {
        "1": [697, 1209], "2": [697, 1336], "3": [697, 1477],
        "4": [770, 1209], "5": [770, 1336], "6": [770, 1477],
        "7": [852, 1209], "8": [852, 1336], "9": [852, 1477],
        "*": [941, 1209], "0": [941, 1336], "#": [941, 1477]
    }

    def map_to_nearest_dtmf(freq, freq_set):
        return min(freq_set, key=lambda x: abs(x - freq))

    df = pd.read_csv(os.path.join(result_folder, f"results_{result_save_filename}.csv"))
    df["frequencies"] = df["frequencies"].apply(ast.literal_eval) # convert string-form list to true python list
    df["low_freq"] = df["frequencies"].apply(lambda x: min(x))
    df["high_freq"] = df["frequencies"].apply(lambda x: max(x))

    correct_low_freqs = 0
    correct_high_freqs = 0
    total_freqs = len(df)

    true_low_freqs = []
    true_high_freqs = []
    pred_low_freqs = []
    pred_high_freqs = []
    low_freq_errors = []
    high_freq_errors = []

    for idx, row in df.iterrows():
        if "clean" in result_save_filename:
            true_low_freq, true_high_freq = DTMF_FREQUENCIES[row["true_number"]]
        else:
            filename_prefix = row['filename_prefix']
            match = re.match(r"^(.{6})_tone(\d+)$", filename_prefix)
            phone_number = match.group(1)
            tone_index = int(match.group(2))
            true_number = phone_number[tone_index - 1]
            freqs_df = pd.read_csv(f"./datasets/dtmf/dtmf_noise_12/{phone_number}_freqs_clean.csv")
            freqs_row = freqs_df.iloc[tone_index - 1]
            true_low_freq, true_high_freq = int(freqs_row['Freq1']), int(freqs_row['Freq2'])

        pred_low = row["low_freq"]
        pred_high = row["high_freq"]

        # compute errors
        low_freq_errors.append(abs(pred_low - true_low_freq))
        high_freq_errors.append(abs(pred_high - true_high_freq))

        # check correctness
        if abs(pred_low - true_low_freq) <= err_tolerance:
            correct_low_freqs += 1
            mapped_low = str(map_to_nearest_dtmf(pred_low, LOW_FREQS))
        else:
            mapped_low = "null"
        if abs(pred_high - true_high_freq) <= err_tolerance:
            correct_high_freqs += 1
            mapped_high = str(map_to_nearest_dtmf(pred_high, HIGH_FREQS))
        else:
            mapped_high = "null"

        true_low_freqs.append(str(map_to_nearest_dtmf(true_low_freq, LOW_FREQS)))
        true_high_freqs.append(str(map_to_nearest_dtmf(true_high_freq, HIGH_FREQS)))
        pred_low_freqs.append(mapped_low)
        pred_high_freqs.append(mapped_high)

    low_freq_acc = correct_low_freqs / total_freqs
    high_freq_acc = correct_high_freqs / total_freqs
    total_freq_acc = (correct_low_freqs + correct_high_freqs) / (2 * total_freqs)

    print(f"Low Frequency Accuracy: {low_freq_acc:.3f}")
    print(f"High Frequency Accuracy: {high_freq_acc:.3f}")
    print(f"Total Frequency Accuracy: {total_freq_acc:.3f}")

    # Remove None before building confusion matrix
    valid_low = [(t, p) for t, p in zip(true_low_freqs, pred_low_freqs) if p is not None]
    valid_high = [(t, p) for t, p in zip(true_high_freqs, pred_high_freqs) if p is not None]
    true_low_valid, pred_low_valid = zip(*valid_low) if valid_low else ([], [])
    true_high_valid, pred_high_valid = zip(*valid_high) if valid_high else ([], [])

    # Confusion Matrix - Low Frequency
    all_low_labels = [str(f) for f in LOW_FREQS] + ['null']
    cm_low = confusion_matrix(true_low_freqs, pred_low_freqs, labels=all_low_labels)
    df_cm_low = pd.DataFrame(cm_low, index=all_low_labels, columns=all_low_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm_low, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.xlabel("Detected Low Freq")
    plt.ylabel("True Low Freq")
    plt.title(f"Confusion Matrix(±{err_tolerance}Hz) (Low Freq Acc: {low_freq_acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"conf_matrix_{result_save_filename}_low_freq_{err_tolerance}Hz.png"))
    plt.close()

    # Confusion Matrix - High Frequency
    all_high_labels = [str(f) for f in HIGH_FREQS] + ['null']
    cm_high = confusion_matrix(true_high_freqs, pred_high_freqs, labels=all_high_labels)
    df_cm_high = pd.DataFrame(cm_high, index=all_high_labels, columns=all_high_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm_high, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.xlabel("Detected High Freq")
    plt.ylabel("True High Freq")
    plt.title(f"Confusion Matrix(±{err_tolerance}Hz) (High Freq Acc: {high_freq_acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"conf_matrix_{result_save_filename}_high_freq_{err_tolerance}Hz.png"))
    plt.close()

    # Frequency error statistics
    low_mean = np.mean(low_freq_errors)
    low_std = np.std(low_freq_errors)
    high_mean = np.mean(high_freq_errors)
    high_std = np.std(high_freq_errors)

    # Frequency error distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(low_freq_errors, bins=20, kde=True)
    plt.title(f"Low Freq Errors (Mean: {low_mean:.2f} Hz, Std: {low_std:.2f} Hz)")
    plt.xlabel("Absolute Error (Hz)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.histplot(high_freq_errors, bins=20, kde=True, color='orange')
    plt.title(f"High Freq Errors (Mean: {high_mean:.2f} Hz, Std: {high_std:.2f} Hz)")
    plt.xlabel("Absolute Error (Hz)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"freq_error_dist_{result_save_filename}.png"))
    plt.close()

    # Define standard DTMF frequencies (low, high) mapped to keys
    cal_acc_freq2key


def cal_acc_freq2key(result_save_filename, err_tolerance):
    # Define standard DTMF frequencies (low, high) mapped to keys
    REV_DTMF_FREQUENCIES = {
        (697, 1209): "1", (697, 1336): "2", (697, 1477): "3",
        (770, 1209): "4", (770, 1336): "5", (770, 1477): "6",
        (852, 1209): "7", (852, 1336): "8", (852, 1477): "9",
        (941, 1209): "*", (941, 1336): "0", (941, 1477): "#"
    }

    LOW_FREQS = [697, 770, 852, 941]
    HIGH_FREQS = [1209, 1336, 1477]
    digit_labels = [str(i) for i in range(1, 10)] + ['*', '0', '#']
    df = pd.read_csv(os.path.join(result_folder, f"results_{result_save_filename}.csv"))
    df["frequencies"] = df["frequencies"].apply(ast.literal_eval) # convert string-form list to true python list
    df["low_freq"] = df["frequencies"].apply(lambda x: min(x))
    df["high_freq"] = df["frequencies"].apply(lambda x: max(x))

    def map_to_nearest_dtmf(freq, freq_set):
        return min(freq_set, key=lambda x: abs(x - freq))

    # Map detected low/high frequencies to nearest DTMF standard frequency
    df["low_freq_mapped"] = df["low_freq"].apply(lambda f: map_to_nearest_dtmf(f, LOW_FREQS))
    df["high_freq_mapped"] = df["high_freq"].apply(lambda f: map_to_nearest_dtmf(f, HIGH_FREQS))

    # Map (low, high) freq pair to DTMF key if within tolerance
    def match_key(row):
        for (std_low, std_high), key in REV_DTMF_FREQUENCIES.items():
            if abs(row["low_freq"] - std_low) <= err_tolerance and abs(row["high_freq"] - std_high) <= err_tolerance:
                return key
        return None

    df["mapped_key"] = df.apply(match_key, axis=1)
    valid_df = df[df["mapped_key"].notnull()].copy()

    print(f"Valid frequency pairs (within ±{err_tolerance}Hz): {len(valid_df)} / {len(df)}")
    acc = accuracy_score(valid_df["mapped_key"], valid_df["predicted_number"])
    print(f"Freq→Key Accuracy (±{err_tolerance}Hz): {acc:.3f}")

    # Confusion Matrix: Freq-pair → Predicted Key
    cm = confusion_matrix(valid_df["mapped_key"], valid_df["predicted_number"], labels=digit_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                xticklabels=digit_labels, yticklabels=digit_labels)
    plt.xlabel("Predicted Key")
    plt.ylabel("Key Mapped from Detected Freq")
    plt.title(f"Confusion Matrix(±{err_tolerance}Hz) (Freq→Key Accuracy: {acc:.3f})\n{len(valid_df)} valid")
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"conf_matrix_{result_save_filename}_freq2key_{err_tolerance}Hz.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result-save-filename', type=str, required=True, help="Results file name stem. A prefix of \"result_\" and a file extension of \".csv\" will be automatically added.")
    parser.add_argument('-e', '--err-tolerance', type=int, help="Error tolerance range for frequency detection")
    parser.add_argument('--no-detail-acc', dest='detail_acc', action="store_false", help="Calculate step-by-step accuracies or not. If guidance is included, step-by-step accuracies will be calculated by default, set --no-detail-acc to disable this feature. Otherwise only overall accuracy will be calculated, regardless of whether this argument is setted or not.")
    args = parser.parse_args()

    if args.err_tolerance is None:
        args.err_tolerance = 15 if "freq_plot" in args.result_save_filename else 5

    # args.result_save_filename = "dsr1_noise_freq_text"
    cal_acc_overall(args.result_save_filename)
    if args.detail_acc and "guide" in args.result_save_filename:
        cal_acc_freqs_detect(args.result_save_filename, args.err_tolerance)
        cal_acc_freq2key(args.result_save_filename, args.err_tolerance)
