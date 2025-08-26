import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = r"E:\bunkerKrunker\results_epoch_train1"

all_results = []

for exp in os.listdir(RESULTS_DIR):
    exp_path = os.path.join(RESULTS_DIR, exp, "results.csv")
    if os.path.exists(exp_path):
        df = pd.read_csv(exp_path)
        # Take last epoch metrics
        final_metrics = df.iloc[-1].to_dict()
        final_metrics["experiment"] = exp
        all_results.append(final_metrics)

# Combine
results_df = pd.DataFrame(all_results)
print(results_df)

# Save
results_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

# Plot mAP vs epochs for each model
for metric in ["metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]:
    plt.figure()
    for model in ["yolov8n", "yolov8s"]:
        subset = results_df[results_df["experiment"].str.contains(model)]
        subset = subset.sort_values("epoch")
        plt.plot(subset["epoch"], subset[metric], label=model)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Epochs")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"{metric.replace('/', '_')}.png"))
