# ネットワークシミュレーター

このプロジェクトは、様々なネットワークトポロジ上でデータ伝播やその他のプロセスをシミュレートするためのツール群です。

## スクリプトの役割

このプロジェクトは、責務の異なる4つの主要なPythonスクリプトで構成されています。

| スクリプト | 役割 |
| :--- | :--- |
| `nwsim.py` | **コア・シミュレータ。** 単一のパラメータ設定でシミュレーションを実行します。 |
| `make_network_for_nwsim.py` | **グラフ生成スクリプト。** シミュレーションに使用するグラフファイル (`node.csv`, `edge.csv`) を生成します。 |
| `run_sweep.py` | **単一グラフ用スイープ実行スクリプト。** 既存のグラフファイルに対し、定義された全戦略のパラメータスイープを並列実行します。 |
| `run_experiment.py` | **大規模自動実験スクリプト。** グラフ生成からパラメータスイープまで、一連の実験全体を自動で実行する最上位のスクリプトです。 |

## 主な機能と変更点

- **リンクごとのパラメータ設定**
  - `edge.csv` ファイルに `connect_rate` と `disconnect_rate` の列を追加することで、リンク（エッジ）ごとに個別の接続・切断率を設定できるようになりました。
  - `make_network_for_nwsim.py` は、これらの値をランダムに割り当ててグラフを生成できます。
  - `nwsim.py` は、`edge.csv` に個別レートの定義がない場合、コマンドライン引数で与えられたグローバル設定値をデフォルトとして使用します（下位互換性）。

- **実験の完全自動化 (`run_experiment.py`)**
  - `run_experiment.py` を実行するだけで、グラフ生成から大規模なパラメータスイープまでを単一のコマンドで実行できます。
  - `--num_graphs` で生成数を、`--topology_type` でトポロジを指定（または全トポロジを対象に）できます。
  - `make_network_for_nwsim.py` の引数（例: `--num_nodes`）をそのまま渡すことで、生成されるグラフの特性も制御可能です。

- **単一グラフでのスイープ実行 (`run_sweep.py`)**
  - ご自身で用意した、あるいは過去に生成した特定のグラフファイルに対して、全パラメータのスイープ（総当たり実験）を実行できます。

- **並列処理による高速化**
  - `run_experiment.py` と `run_sweep.py` は、内部で複数のシミュレーションを並列で実行します。
  - スクリプト上部の `MAX_WORKERS` 変数を調整することで、並列度（使用するCPUコア数）を変更できます。

---

## 使い方

### 1. 大規模な自動実験 (`run_experiment.py`)

グラフ生成からパラメータスイープまでを完全に自動化する場合に使用します。

**コマンド例:**

- **全トポロジについて、それぞれ2個ずつグラフを生成して実験**
  ```bash
  python run_experiment.py --num_graphs 2
  ```

- **`grid` トポロジに限定し、5個のグラフを生成して実験**
  ```bash
  python run_experiment.py --num_graphs 5 --topology_type grid
  ```

- **グラフ生成のパラメータも指定**
  ```bash
  python run_experiment.py --num_graphs 5 --topology_type barabasi_albert --num_nodes 200 --barabasi_m 3
  ```

### 2. 既存グラフでのスイープ実行 (`run_sweep.py`)

ご自身で用意した、あるいは過去に生成した特定のグラフファイルに対して、定義された全パラメータのスイープ（総当たり実験）を並列実行する場合に使用します。

**グラフファイルの配置:**
グラフファイル（`node.csv` と `edge.csv`）は、プロジェクト内のどこに置いても構いません。実行時に、コマンドライン引数でそのファイルの**絶対パス**または**相対パス**を指定します。

**コマンド例:**

```bash
# 例1: プロジェクト内のサブディレクトリに置いた場合 (相対パス)
# (例: nksim/my_graph_data/my_nodes.csv)
python run_sweep.py --node_file my_graph_data/my_nodes.csv --edge_file my_graph_data/my_edges.csv --output_base_dir my_custom_graph_result

# 例2: 絶対パスで指定する場合 (Windowsの例)
python run_sweep.py --node_file C:\Users\YourUser\Desktop\my_graphs\nodes.csv --edge_file C:\Users\YourUser\Desktop\my_graphs\edges.csv --output_base_dir my_desktop_result

# 例3: 絶対パスで指定する場合 (Linux/macOSの例)
python run_sweep.py --node_file /home/youruser/my_graphs/nodes.csv --edge_file /home/youruser/my_graphs/edges.csv --output_base_dir /home/youruser/my_results
```

**引数の説明:**

*   `--node_file`: （必須）ご自身で用意したノードファイルのパス。
*   `--edge_file`: （必須）ご自身で用意したエッジファイルのパス。
*   `--output_base_dir`: （任意）シミュレーション結果を保存するディレクトリ名。指定しない場合は `sweep_result` という名前のディレクトリが作成され、その中に結果が保存されます。


### 3. グラフの生成 (`make_network_for_nwsim.py`)

シミュレーションに使うグラフファイルを手動で生成する場合に使用します。

**コマンド例:**

```bash
python make_network_for_nwsim.py --num_nodes 50 --topology_type grid --node_output_path my_node.csv --edge_output_path my_edge.csv
```

### 4. 単一シミュレーションの実行 (`nwsim.py`)

特定のパラメータで1回だけシミュレーションを実行する場合に使用します。

**コマンド例:**

```bash
python nwsim.py --node_file node.csv --edge_file edge.csv --strategy fixed_wait_duration_strategy --routing_strategy dijkstra
```

---

## 出力ディレクトリ構造

`run_experiment.py` を実行すると、結果は `graph/` と `result/` の2つのトップレベルディレクトリに保存されます。

```
.
├── graph/  <-- 生成されたすべてのグラフデータ
│   └── {topology_type}/
│       ├── graph_0/
│       │   ├── node.csv
│       │   └── edge.csv
│       └── graph_1/
│           └── ...
│
└── result/ <-- すべてのシミュレーション結果
    └── {topology_type}/
        └── graph_{index}/
            ├── dijk/  <-- ルーティング戦略ごとのディレクトリ
            │   ├── fixed_wait/  <-- 待機戦略ごとのディレクトリ
            │   │   ├── summary_dijk_fixed_p0.5.csv  <-- パラメータ値を含むファイル名
            │   │   └── ...
            │   └── ...
            └── reli/
                └── ...
```
