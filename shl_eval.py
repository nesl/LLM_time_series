import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--result-save-filename', type=str, required=True, help="Results file name stem. A prefix of "result_" and a file extension of ".csv" will be automatically added. This filename will also be used to generate the confusion matrix file name.") # e.g., args.result_save_filename = "gpto1_User1_220617_120_Torso_env_only_4class"
args = parser.parse_args()

file_path = f"./results/HAR/results_{args.result_save_filename}.csv"
df = pd.read_csv(file_path)

class_labels = ['Still', 'Walking', 'Run', 'Car']

def extract_prediction(pred_label_str):
    for label in class_labels:
        if label in pred_label_str:
            return label
    return "Null"

df["predicted_category"] = df["predicted_label"].apply(extract_prediction)

df["correct"] = df.apply(lambda row: row["true_label"] in row["predicted_label"], axis=1)

accuracy = df["correct"].mean()

conf_matrix = confusion_matrix(df["true_label"], df["predicted_category"], labels=class_labels + ["Null"])
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels + ["Null"], columns=class_labels + ["Null"])

print(f"Acc: {accuracy:.3f}")
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Acc: {accuracy:.3f})")
plt.show()
plt.savefig(f'./results/HAR/conf_matrix_{args.result_save_filename}.png')