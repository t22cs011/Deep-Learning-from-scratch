```mermaid
graph TD
    subgraph "実験全体 (各ネットワーク構成で繰り返し)"
        direction TB
        A["実験開始:<br>ある1つのネットワーク構成"];
        A --> OuterLoopStart{"エポック処理<br>(200回繰り返す)"};
    end

    %% ここからが「1エポック」の範囲の開始点です
    subgraph "1エポックの処理"
        direction LR
        OuterLoopStart --> B["(ii) 1エポックで<br>(ほぼ)全データを一巡<br><b>(約468イテレーション実行)</b>"];
        B --> LoopControl{"ミニバッチ処理<br><b>(ここが1イテレーション)</b>"};
    end

    %% 「LoopControl」から「IterationEnd」までが「1イテレーション」の処理内容です
    subgraph "ミニバッチ処理 (1イテレーション)"
        direction TB
        LoopControl --> C["(i) 訓練データをシャッフルし、<br>128個のミニバッチに分割<br>(実際はランダムサンプリング)"];
        C --> D["(iii) 各ミニバッチに対し<br>勾配を計算"];
        D --> E["パラメータ更新"];
        E --> IterationEnd{"ミニバッチ処理 完了<br><b>(1イテレーション 完了)</b>"};
    end
    %% ここまでが「1イテレーション」の処理内容です

    IterationEnd -.-> B; %% 次のイテレーションへ (Bに戻り、再度LoopControlへ)
    %% 「B」に戻り、再び「LoopControl」から始まるイテレーションを繰り返す。
    %% これを約468回繰り返すと「1エポック」が完了します。

    OuterLoopStart -- 200エポック完了 --> F["実験終了 /<br>次の構成へ"];
    %% 「OuterLoopStart」から始まり、200回「1エポックの処理」を繰り返した全体が200エポック分の処理です。