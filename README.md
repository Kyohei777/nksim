# ネットワークシミュレーター

このプロジェクトは、様々なネットワークトポロジ上でデータ伝播やその他のプロセスをシミュレートするためのツール群です。

## スクリプトの実行関係

各スクリプトは以下の階層関係で実行されます。

```
[実行フロー1: 全トポロジで実行]
run_all_topology_simulations.py
    |
    +-- (1) make_network_for_nwsim.py を呼び出し (ネットワーク生成)
    |
    +-- (2) run_custom_sweep.py を呼び出し (スイープ実行)
            |
            +-- nwsim.py を繰り返し呼び出し (シミュレーション本体)

[実行フロー2: 単一設定で実行]
run_simulation_python.py
    |
    +-- run_custom_sweep.py をインポートして利用
            |
            +-- nwsim.py を繰り返し呼び出し (シミュレーション本体)
```

---

## スクリプト詳細

### `nwsim.py` (コアエンジン)

シミュレーションを実行するコアプログラムです。多数のコマンドライン引数を受け取り、振る舞いを細かく制御できます。

- **役割**: コマンドライン引数で与えられた設定に基づき、単一のシミュレーションを実行し、結果を指定されたCSVファイルに出力します。
- **主なコマンドライン引数**:
    - **ファイルパス**: `--node_file`, `--edge_file`, `--output_base_dir`, `--summary_filename`
    - **シミュレーション基本設定**: `--packets`, `--lambda_rate` (パケット生成レート), `--max_sim_time`, `--buffer_size`, `--ttl` (生存時間)
    - **リンク変動設定**: `--connect_rate`, `--disconnect_rate`, `--link_seed` (乱数シード)
    - **ルーティング戦略**: `--routing_strategy` (`dijkstra` または `reliable`)
    - **待機戦略**: `--strategy` (`fixed_wait_duration_strategy`, `no_wait_strategy` など)
    - **各戦略のパラメータ**: `--base_wait_time`, `--dynamic_factor`, `--ratio_factor`
    - **その他**: `--src_node`, `--dst_node`, `--debug_packet_ids` など多数。

---

### `run_custom_sweep.py` (パラメータスイープ実行)

`nwsim.py`を繰り返し呼び出し、パラメータの総当たり実験（スイープ）を行う中心的なスクリプトです。

- **役割**: 自身が受け取ったネットワーク情報（ノード、エッジファイル等）と、ソースコード内にハードコーディングされたパラメータ範囲を基に、多数の`nwsim.py`プロセスを並列実行します。
- **受け取るコマンドライン引数**:
    - `node_file`: ノード情報が記載されたCSVファイルのパス。
    - `edge_file`: エッジ情報が記載されたCSVファイルのパス。
    - `source_node`: シミュレーションの始点ノードID。
    - `dest_node`: シミュレーションの終点ノードID。
    - `output_dir`: 結果を出力するルートディレクトリ。
- **直接編集が必要な設定（ハードコーディング）**:
    - `RATE_SWEEP_VALUES`: `connect_rate`と`disconnect_rate`として試行する値のリスト。(`np.arange`で定義)
    - `STRATEGIES_WITH_PARAMS`: 実験対象とする待機戦略と、その戦略が持つパラメータ（例: `dynamic_factor`）の試行範囲を定義した辞書。
    - `NUM_PROCESSES`: 並列実行するプロセス数。

---

### `run_all_topology_simulations.py` (統合実行)

複数のネットワークトポロジを対象に、設計からシミュレーションまでを一括で行う最高レベルのオーケストレーションスクリプトです。

- **役割**: ソースコード内で定義された複数のネットワークトポロジ設定を順番に処理します。各トポロジに対して`make_network_for_nwsim.py`でネットワークを生成し、その後`run_custom_sweep.py`を呼び出してパラメータスイープを実行します。
- **受け取るコマンドライン引数**: なし。すべての設定はソースコード内で完結します。
- **直接編集が必要な設定（ハードコーディング）**:
    - `TOPOLOGIES_TO_SIMULATE`: 生成とシミュレーションの対象とするネットワークトポロジの種類と、その生成に必要なパラメータを定義した辞書。（例: `"random": {"PROB_EDGE_RANDOM": 0.2}`）

---

### `make_network_for_nwsim.py` (ネットワーク生成)

シミュレーションに使用するネットワークを定義したCSVファイルを生成します。

- **役割**: コマンドライン引数で指定されたトポロジの種類やノード数に基づき、`node.csv`と`edge.csv`を生成します。
- **主なコマンドライン引数**:
    - `--topology_type`: `random`, `grid`, `barabasi_albert` などのトポロジ種別。
    - `--num_nodes`: 生成するノード数。
    - `--node_output_path`, `--edge_output_path`: 出力ファイルパス。
    - **トポロジごとの引数**: `--probability_of_edge` (random用), `--barabasi_m` (barabasi_albert用) など。

---

### `run_simulation_python.py` (単一実行ラッパー)

特定のネットワークファイルに対して、`run_custom_sweep.py`で定義されたパラメータスイープを実行するための簡単なラッパーです。

- **役割**: ソースコード内で指定されたCSVファイル（ノード、エッジ、始点・終点）を読み込み、その情報を`run_custom_sweep.py`の`main`関数に渡して実行します。
- **受け取るコマンドライン引数**: なし。
- **直接編集が必要な設定（ハードコーディング）**:
    - `NODE_FILE`: 使用するノードファイルのパス。
    - `EDGE_FILE`: 使用するエッジファイルのパス。
    - `ENDPOINTS_FILE`: 始点・終点ノードが書かれたファイルのパス。
