# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import glob
import pandas as pd
from multi_layer_nn import MultiLayerNN
from optimizer import Adam
from mnist import load_mnist
from functions import relu

def run_experiments():
    # 実験設定（全パターンでepochs=200、学習率(lr)=0.01に統一）
    experiments = [
        # 1層のパターン
        # {"hidden_sizes": [4],    "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [8],    "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [16],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [32],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [64],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [128],  "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [256],  "epochs": 200, "batch_size": 128, "lr": 0.01},
        # {"hidden_sizes": [512],  "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [1024], "epochs": 200, "batch_size": 128, "lr": 0.01},

        # 2層のパターン
        {"hidden_sizes": [4, 4],       "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [8, 8],       "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [16, 16],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [32, 32],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [64, 64],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [128, 128],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [256, 256],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [512, 512],   "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [1024, 1024], "epochs": 200, "batch_size": 128, "lr": 0.01},

        # 3層のパターン
        {"hidden_sizes": [4, 4, 4],          "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [8, 8, 8],          "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [16, 16, 16],       "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [32, 32, 32],       "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [64, 64, 64],       "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [128, 128, 128],    "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [256, 256, 256],    "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [512, 512, 512],    "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [1024, 1024, 1024], "epochs": 200, "batch_size": 128, "lr": 0.01},

        # 4層のパターン
        {"hidden_sizes": [4, 4, 4, 4],             "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [8, 8, 8, 8],             "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [16, 16, 16, 16],         "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [32, 32, 32, 32],         "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [64, 64, 64, 64],         "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [128, 128, 128, 128],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [256, 256, 256, 256],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [512, 512, 512, 512],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [1024, 1024, 1024, 1024],"epochs": 200, "batch_size": 128, "lr": 0.01},

        # 追加パターン
        {"hidden_sizes": [4, 32, 256, 2048], "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [2048, 256, 32, 4], "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [329, 329, 329, 329],     "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [32] * 32,          "epochs": 200, "batch_size": 128, "lr": 0.01},
        {"hidden_sizes": [1024] * 8,          "epochs": 200, "batch_size": 128, "lr": 0.01},
    ]

    all_results = []
    for idx, config in enumerate(experiments, 1):
        # ---- 今回実行中の実験をコンソールに表示 ----
        layer_str = "-".join(map(str, config["hidden_sizes"]))
        print(f"\n===== Running experiment: {layer_str} =====")
        (X_train, T_train), (X_test, T_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
        train_ratio = config.get("train_ratio", 1.0)
        if train_ratio < 1.0:
            N = int(len(X_train) * train_ratio)
            X_train, T_train = X_train[:N], T_train[:N]

        net = MultiLayerNN(
            input_size=784,
            hidden_sizes=config["hidden_sizes"],
            output_size=10
        )
        optimizer = Adam(lr=config["lr"])
        train_acc_list, test_acc_list = [], []
        iter_per_epoch = max(1, X_train.shape[0] // config["batch_size"])
        start_time = time.time()

        for epoch in range(config["epochs"]):
            for _ in range(iter_per_epoch):
                batch_mask = np.random.choice(X_train.shape[0], config["batch_size"])
                x_batch = X_train[batch_mask]
                t_batch = T_train[batch_mask]
                grads = net.gradient(x_batch, t_batch)
                optimizer.update(net.params, grads)
            train_acc = net.accuracy(X_train, T_train)
            test_acc = net.accuracy(X_test, T_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Epoch {}/{} - Train: {:.4f}, Test: {:.4f}".format(epoch+1, config["epochs"], train_acc, test_acc))  

        # 層構成を文字列化
        layer_str = "-".join(map(str, config["hidden_sizes"]))
        # 画像ファイル名を "[32]" などの形式に変更
        filename = f"[{layer_str}].png"
        title = "Accuracy: {}".format(str(config["hidden_sizes"]))
        elapsed = time.time() - start_time

        epochs_range = np.arange(1, config["epochs"]+1)
        plt.figure()
        plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
        plt.plot(epochs_range, test_acc_list, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)

        ax = plt.gca()
        ax.set_ylim(0.8, 1.0)
        ax.set_yticks(np.arange(0.8, 1.01, 0.05)) # Major ticks: 0.50, 0.55, ... , 1.00
        ax.set_yticks(np.arange(0.8, 1.01, 0.01), minor=True) # Minor ticks: 0.01刻み
        ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)

        plt.figtext(0.01, 0.02, "Execution time: {:.2f} s".format(elapsed), ha='left', va='bottom')
        plt.savefig(filename)
        print("✅ Saved: {} ({:.2f}秒)".format(filename, elapsed))
        plt.close()

        # ---- 各実験ごとのログをCSVに保存 ----
        per_exp_csv = f"{layer_str}_log.csv"
        with open(per_exp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Training Accuracy", "Test Accuracy"])
            for ep, (tr_acc, te_acc) in enumerate(zip(train_acc_list, test_acc_list), start=1):
                writer.writerow([ep, f"{tr_acc:.4f}", f"{te_acc:.4f}"])
        print(f"✅ Saved per-experiment log: {per_exp_csv}")

        # CSV用の結果をリストに蓄積
        final_train_acc = train_acc_list[-1]
        final_test_acc = test_acc_list[-1]
        avg_train_acc = np.mean(train_acc_list)
        avg_test_acc = np.mean(test_acc_list)
        abs_error = abs(avg_train_acc - avg_test_acc)
        pattern_str = f"[{layer_str}]"
        all_results.append([idx, pattern_str, final_train_acc, final_test_acc, avg_train_acc, avg_test_acc, abs_error, elapsed])

    # 全実験結果を1つのCSVファイルに出力
    csv_filename = "experiment_all_log.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Experiment Index", "Config", "Final Train Acc", "Final Test Acc", "Avg Train Acc", "Avg Test Acc", "Abs Error", "Execution Time"])
        for row in all_results:
            writer.writerow(row)
    print("✅ CSV log saved: {}".format(csv_filename))
    print("✅ All experiments completed")

    # ---- 全実験パターンを層数ごとにまとめて可視化 ----
    history = {1: {}, 2: {}, 3: {}, 4: {}}
    for filepath in glob.glob("*_log.csv"):
        if filepath == "experiment_all_log.csv": continue
        df = pd.read_csv(filepath)
        if "Test Accuracy" not in df.columns: continue
        layer_str = os.path.basename(filepath)[:-8]
        dims = list(map(int, layer_str.split('-')))
        layers = len(dims)
        history[layers][layer_str] = df["Test Accuracy"].values

    # カラー設定：ユニット数ごとに同じ色を割り当て
    color_map = {'4':'purple','16':'blue','64':'green','256':'orange','1024':'red'}
    fig, axes = plt.subplots(2,2, figsize=(12,8), sharex=True, sharey=True)
    layer_axes = {1:axes[0,0], 2:axes[0,1], 3:axes[1,0], 4:axes[1,1]}
    for layers, ax in layer_axes.items():
        for key in sorted(history[layers].keys(), key=lambda k:int(k.split('-')[0])):
            series = history[layers][key]
            base = key.split('-')[0]
            c = color_map.get(base,None)
            ax.plot(series, label=key, color=c)
        ax.set_title(f"Test Accuracy by Epoch ({layers} Hidden Layer{'s' if layers>1 else ''})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0.8,1.0)
        ax.set_yticks(np.arange(0.8,1.01,0.05))
        ax.set_yticks(np.arange(0.8,1.01,0.01), minor=True)
        ax.grid(which='minor', axis='y', linestyle=':', linewidth=0.5)
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("comparison_by_layer_and_units.png")
    print("✅ Saved comparison plot: comparison_by_layer_and_units.png")

    # Additional groups...
    patterns1 = ["4-32-256-2048","2048-256-32-4","329-329-329-329"]
    plt.figure(figsize=(8,6))
    for key in patterns1:
        series = history.get(len(key.split('-')),{}).get(key)
        if series is not None: plt.plot(series,label=key)
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title("Comparison Group 1: Unbalanced Four-Layer Patterns")
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig("comparison_group1.png"); print("✅ Saved comparison_group1.png"); plt.close()

    patterns2 = ["32-"*31+"32","1024","512-512","256-256-256-256"]
    plt.figure(figsize=(8,6))
    for key in patterns2:
        series = history.get(len(key.split('-')),{}).get(key)
        if series is not None: plt.plot(series,label=key)
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title("Comparison Group 2: Mixed-Depth Small Patterns")
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig("comparison_group2.png"); print("✅ Saved comparison_group2.png"); plt.close()

    patterns3 = [
        "1024-1024-1024-1024-1024-1024-1024-1024",
        "1024","1024-1024","1024-1024-1024","1024-1024-1024-1024"
    ]
    plt.figure(figsize=(8,6))
    for key in patterns3:
        series = history.get(len(key.split('-')),{}).get(key)
        if series is not None: plt.plot(series,label=key)
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.title("Comparison Group 3: 1024-Unit Variations")
    plt.legend(loc='lower right'); plt.grid(True)
    plt.savefig("comparison_group3.png"); print("✅ Saved comparison_group3.png"); plt.close()

if __name__ == '__main__':
    run_experiments()