import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import defaultdict

def format_config(config_str):
    # 角括弧を削除
    config_str = config_str.strip('[]')
    # ハイフンで分割
    units = config_str.split('-')
    # 同じ数字が連続する回数をカウント
    current_unit = units[0]
    count = 1
    formatted_parts = []
    
    for i in range(1, len(units)):
        if units[i] == current_unit:
            count += 1
        else:
            if count > 1:
                formatted_parts.append(f"{current_unit}*{count}")
            else:
                formatted_parts.append(current_unit)
            current_unit = units[i]
            count = 1
    
    # 最後のグループを追加
    if count > 1:
        formatted_parts.append(f"{current_unit}*{count}")
    else:
        formatted_parts.append(current_unit)
    
    return '-'.join(formatted_parts)

def get_sort_key(config_str):
    # 特殊なパターン（ハイフンで区切られた異なる数字）の場合は最後に
    if '-' in config_str and '*' not in config_str:
        return (999, 0)  # 最大の層数として扱う
    
    # 通常のパターン（例：32*4）の場合
    if '*' in config_str:
        unit, layers = config_str.split('*')
        return (int(layers), int(unit))
    
    # 単一層の場合
    return (1, int(config_str))

# CSVファイルを読み込む
df = pd.read_csv('experiment_all_log.csv')

# 必要な列を選択
selected_columns = ['Config', 'Final Train Acc', 'Final Test Acc', 'Avg Train Acc', 'Avg Test Acc', 'Execution Time']
result_df = df[selected_columns]

# 実行時間を分単位に変換
result_df['Execution Time (min)'] = result_df['Execution Time'] / 60

# 結果を表示
print("\n実験結果のまとめ:")
print("=" * 100)
print(result_df.to_string(index=False))

# 結果をCSVファイルとして保存
result_df.to_csv('experiment_summary.csv', index=False)

# グラフの作成
plt.figure(figsize=(12, 6))
sns.barplot(data=result_df, x='Config', y='Final Test Acc')
plt.xticks(rotation=45, ha='right')
plt.title('Final Test Accuracy for Each Configuration')
plt.tight_layout()
plt.savefig('test_accuracy_comparison.png')
plt.close()

# すべての *_log.csv ファイルを集計（experiment_all_log.csvは除外）
log_files = [f for f in glob.glob("*_log.csv") if f != "experiment_all_log.csv"]
summary = []

for log_file in log_files:
    df_log = pd.read_csv(log_file)
    config = log_file.replace('_log.csv', '')
    final_train_acc = df_log['Training Accuracy'].iloc[-1]
    final_test_acc = df_log['Test Accuracy'].iloc[-1]
    avg_train_acc = df_log['Training Accuracy'].mean()
    avg_test_acc = df_log['Test Accuracy'].mean()
    # 実行時間はexperiment_all_log.csvのみなので、ここでは空欄
    summary.append({
        'Config': format_config(config),
        'Final Train Acc': final_train_acc,
        'Final Test Acc': final_test_acc,
        'Avg Train Acc': avg_train_acc,
        'Avg Test Acc': avg_test_acc,
        'Execution Time': ''
    })

# DataFrame化
summary_df = pd.DataFrame(summary)

# 既存のexperiment_all_log.csvのデータも追加
if os.path.exists('experiment_all_log.csv'):
    exp_all_df = pd.read_csv('experiment_all_log.csv')
    exp_all_df = exp_all_df.rename(columns={'Config': 'Config',
                                            'Final Train Acc': 'Final Train Acc',
                                            'Final Test Acc': 'Final Test Acc',
                                            'Avg Train Acc': 'Avg Train Acc',
                                            'Avg Test Acc': 'Avg Test Acc',
                                            'Execution Time': 'Execution Time'})
    exp_all_df = exp_all_df[['Config', 'Final Train Acc', 'Final Test Acc', 'Avg Train Acc', 'Avg Test Acc', 'Execution Time']]
    # Configの表記を変更
    exp_all_df['Config'] = exp_all_df['Config'].apply(format_config)
    summary_df = pd.concat([summary_df, exp_all_df], ignore_index=True)

# ソートキーを計算してソート
summary_df['sort_key'] = summary_df['Config'].apply(get_sort_key)
summary_df = summary_df.sort_values('sort_key')
summary_df = summary_df.drop('sort_key', axis=1)

# 結果を表示
print("\n全ログファイルの集計結果:")
print("=" * 100)
print(summary_df.to_string(index=False))

# CSVとして保存
summary_df.to_csv('all_experiment_summary.csv', index=False)

# 層数ごとの精度推移グラフを作成

# 層数ごとにファイルを分類
layer_files = defaultdict(list)
for log_file in log_files:
    config_raw = log_file.replace('_log.csv', '')
    # 角括弧付き表記
    config_bracket = f'[{config_raw.replace("-", "-")}]'
    # 層数をカウント
    units = config_raw.split('-')
    layer_count = len(units)
    layer_files[layer_count].append((log_file, config_raw))

for layer_count, files in sorted(layer_files.items()):
    plt.figure(figsize=(10, 6))  # 幅を12から8に変更
    for log_file, config_raw in sorted(files, key=lambda x: int(x[1].split('-')[0])):
        # 層数4の場合、すべてのユニット数が異なるパターンや[329-329-329-329]は除外
        if layer_count == 4:
            units = config_raw.split('-')
            # すべてのユニット数が異なる場合、または全て329の場合はスキップ
            if len(set(units)) == 4 or all(u == '329' for u in units):
                continue
        df_log = pd.read_csv(log_file)
        legend_label = f'[{config_raw}]'
        plt.plot(df_log['Epoch'], df_log['Test Accuracy'], label=legend_label)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f'Accuracy Progress for {layer_count} Layer{"s" if layer_count > 1 else ""} Networks')
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')  # 凡例を右側中央に配置
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{layer_count}_layer_accuracy_progress.png', bbox_inches='tight')  # bbox_inches='tight'を追加
    plt.close() 