
# PerFedRec++: 自己教師あり事前学習によるパーソナライズされた連合レコメンデーションの強化

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2305.06622.pdf)](https://arxiv.org/pdf/2305.06622.pdf)

このプロジェクトは、**PerFedRec++** モデルのPyTorch実装です。PerFedRec++は、パーソナライズされた連合レコメンデーションシステムを強化するために提案された新しいフレームワークです。

## 概要 (Introduction)

今日のデジタル世界では、AmazonやYouTubeのようなプラットフォームで「次に何を見るか」「何を買うか」を提案してくれる**レコメンデーションシステム**が非常に重要です。しかし、これらのシステムがユーザーの好みや行動に関する大量のデータを収集・利用するにつれて、**ユーザープライバシー**に関する深刻な懸念が高まっています。GDPRやCCPAのような規制も、このプライバシー保護の必要性から生まれました。

このプライバシーの課題に対処するため、**連合学習 (Federated Learning, FL)** という有望なパラダイムが登場しました。連合学習は、生のユーザーデータを中央サーバーに送信するのではなく、**モデルの更新情報のみを交換**することでユーザーデータのプライバシーを保護します。レコメンデーションシステムと連合学習を組み合わせたものが**連合レコメンデーションシステム**です。

現在の連合レコメンデーションシステムは、いくつかの大きな課題に直面しています:
1.  **異質性とパーソナライゼーション**: ユーザーの属性やローカルデータの多様性のため、個々のユーザーに合わせたモデル（パーソナライズされたモデル）が必要です。
2.  **モデル性能の劣化**: 擬似アイテムラベリングや差分プライバシーなど、プライバシー保護のためのメカニズムがモデルの性能を低下させる可能性があります。
3.  **通信のボトルネック**: 標準的な連合レコメンデーションアルゴリズムは、サーバーとデバイス間の通信オーバーヘッドが高くなる傾向があります。

**PerFedRec++** は、これらの課題を同時に解決するために提案された新しいフレームワークです。

## PerFedRec++の仕組み (How PerFedRec++ Works)

PerFedRec++は、主に以下の3つの主要なモジュールで構成されています。

### 1. 自己教師あり事前学習モジュール (Self-Supervised Pre-Training Module)

*   **データ拡張の利用**: 連合レコメンデーションシステムのプライバシー保護メカニズム（例: クライアント選択、擬似アイテムラベリング、差分プライバシー）自体が、システムに関する異なる「ビュー」を生成すると捉えられます。これは、グラフ学習におけるノードドロップアウトやエッジ摂動、ノイズ注入といったデータ拡張の手法に似ています。
*   **対照学習 (Contrastive Learning)**: 生成されたこれら2つの拡張されたグラフビューを**対照タスク**として使用し、**自己教師ありグラフ学習**によってモデルを事前学習させます。具体的には、対照損失 (contrastive loss) を用いて、同じノードの異なるビュー間の類似度を最大化します。
*   **効果**: この事前学習により、埋め込みの表現がより統一され、モデルの性能が向上します。また、連合学習の初期状態を改善することで、モデルの収束が速くなり、全体の訓練にかかる時間と**通信負荷を軽減**します。

### 2. ユーザーサイドのローカルレコメンデーションネットワーク (User-Side Local Recommendation Network)

*   **埋め込み層**: 各ユーザーとアイテムは、それぞれ密な埋め込みベクトルに変換されます。事前学習された埋め込みが初期値として使われます。
*   **ローカルGNNモジュール (Local Graph Neural Network)**: 各ユーザーは自身のプライベートなユーザー・アイテムインタラクション情報（履歴データ）を用いて**ローカルGNN**を訓練します。アイテム埋め込みはサーバーを通じて共有されますが、ユーザー埋め込みはプライバシー保護のためローカルに保持されます。
*   **パーソナライズされた予測**: 最終的に、各ユーザーは**グローバルな連合モデル**、**クラスタレベルの連合モデル**、そして**自身のファインチューンされたローカルモデル**を組み合わせて、パーソナライズされたレコメンデーションモデルを獲得します。
*   **プライバシー保護**: 勾配を保護するため、**ローカル差分プライバシー (Local Differential Privacy, LDP)** が適用されます。これは、勾配にランダムなノイズを加えることで実現されます。

### 3. サーバーサイドのクラスタリングに基づく連合 (Server-Side Clustering-Based Federation)

*   **ユーザーのクラスタリング**: サーバーは、各ユーザーの学習された埋め込み (表現) に基づいて、ユーザーを類似した**K個のグループにクラスタリング**します。K-meansのような一般的なクラスタリング手法が利用されます。
*   **モデルの集約**: サーバーは、参加するすべてのユーザーからの重み付けされた合計によって**グローバルモデル**を統合します。さらに、各クラスタ内のユーザーからの重み付けされた合計によって、**クラスタレベルのモデル**も集約します。これらのモデルは、パーソナライズされたレコメンデーションのために各ユーザーに提供されます。
*   **ユーザー選択**: 通信コストが高いシナリオでは、各クラスタ内でクラスタサイズに比例してランダムにユーザーを選択する**クラスタベースのユーザー選択メカニズム**も導入されています。

## 実験結果 (Experimental Results)

PerFedRec++は、3つの実世界データセット（**Yelp、Amazon-Kindle、Gowalla**）で広範な実験を行い、その有効性と効率性を検証しました。

*   **優れた性能**: PerFedRec++は、既存の最先端の連合レコメンデーションシステムと比較して、すべてのデータセットで**優れた性能**を達成しました。特に、先行研究であるPerFedRecと比較して、RecallとNDCGで平均11.92%と11.68%の相対的な改善が見られました。
*   **自己教師あり事前学習とパーソナライゼーションの寄与**: アブレーションスタディにより、自己教師あり事前学習とパーソナライゼーションモジュールの両方が、全体の性能向上に大きく貢献していることが示されました。
*   **訓練効率の向上**: 自己教師あり事前学習は、より良い初期状態を提供することで、連合訓練の**収束を速め**、通信オーバーヘッドを削減することが確認されました。

## リポジトリの使い方 (Repository Usage)

このリポジトリは、PerFedRec++モデルを動かすためのコードと設定ファイルを提供しています。

### プロジェクト構造 (Project Structure)

```
.
├── LICENSE
├── PerFedRec++
│   ├── SELFRec.py
│   ├── base
│   │   ├── graph_recommender.py
│   │   └── recommender.py
│   ├── conf
│   │   ├── FedGNN.conf
│   │   ├── FedMF.conf
│   │   ├── LightGCN.conf
│   │   ├── MF.conf
│   │   ├── PerFedRec.conf
│   │   ├── PerFedRec_plus.conf  <- PerFedRec++の設定ファイル
│   │   ├── SGL.conf
│   │   ├── SimGCL.conf
│   │   └── XSimGCL.conf
│   ├── data
│   │   ├── augmentor.py
│   │   ├── data.py
│   │   ├── graph.py
│   │   ├── loader.py
│   │   └── ui_graph.py
│   ├── dataset
│   │   ├── yelp_test             <- Yelpデータセットの例
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   └── valid.txt
│   ├── main.py                   <- 実行スクリプト
│   ├── model
│   │   └── graph
│   │       ├── FedGNN.py
│   │       ├── FedMF.py
│   │       ├── LightGCN.py
│   │       ├── MF.py
│   │       ├── PerFedRec.py
│   │       ├── PerFedRec_plus.py <- PerFedRec++のモデル実装
│   │       ├── SGL.py
│   │       ├── SimGCL.py
│   │       └── XSimGCL.py
│   ├── result_vis
│   │   ├── result_vis.py
│   │   ├── result_vis_ali.py
│   │   ├── result_vis_clu.py
│   │   └── result_vis_uni.py
│   └── util
│       ├── algorithm.py
│       ├── conf.py
│       ├── evaluation.py
│       ├── logger.py
│       ├── loss_torch.py
│       └── sampler.py
├── README.md
└── fig
    ├── fig1.png
    └── fig2.png
```


### 実行方法 (How to Run)

`main.py`スクリプトを使用して、モデルの訓練と評価を実行できます。

**コマンドライン引数**:

*   `--model`: 使用するモデル名を指定します。`PerFedRec_plus`を指定してください。
    *   利用可能なグラフベースラインモデル：`LightGCN`, `MF`, `FedGNN`, `FedMF`, `PerFedRec`, `PerFedRec_plus`。
    *   利用可能な自己教師ありグラフモデル：`SGL`, `SimGCL`, `XSimGCL`。
*   `--dataset`: 使用するデータセットを指定します。`kindle`、`yelp`、`gowalla`のいずれかです。
    *   内部では、`kindle`は`kindle_test`に、`yelp`は`yelp_test`に変換されます。
*   `--emb`: 埋め込みのサイズを指定します。デフォルトは`64`です。
*   `--pretrain_epoch`: 事前学習のエポック数を指定します。デフォルトは`5`です。
*   `--noise_scale`: ノイズのスケール（差分プライバシー関連）を指定します。デフォルトは`0.1`です。
*   `--clip_value`: 勾配クリッピングの値（差分プライバシー関連）を指定します。デフォルトは`0.5`です。
*   `--pretrain_noise`: 事前学習時のノイズのスケールを指定します。デフォルトは`0.1`です。
*   `--pretrain_nclient`: 事前学習時に参加するクライアントの数。デフォルトは`256`です。

**実行例**:

```bash
# YelpデータセットでPerFedRec++モデルを実行
python main.py --model PerFedRec_plus --dataset yelp

# Amazon-KindleデータセットでPerFedRec++モデルを実行し、埋め込みサイズを128に設定
python main.py --model PerFedRec_plus --dataset kindle --emb 128

# GowallaデータセットでPerFedRec++モデルを実行し、事前学習エポック数を10に設定
python main.py --model PerFedRec_plus --dataset gowalla --pretrain_epoch 10
```

これにより、指定されたモデルが訓練され、結果が`./resultsX/`ディレクトリ（設定ファイル内の`output.setup`で指定）に出力されます。

## 引用 (Citation)

もしPerFedRec++があなたの研究やアプリケーションで役立った場合、以下の論文を引用してください:

```bibtex
@article{luo2023perfedrec++,
  title={PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training},
  author={Luo, Sichun and Xiao, Yuanzhang and Zhang, Xinyi and Liu, Yang and Ding, Wenbo and Song, Linqi},
  journal={arXiv preprint arXiv:2305.06622},
  year={2023}
}
```


## 謝辞 (Acknowledgement)

このプロジェクトは、[SELFRec](https://github.com/Coder-Yu/SELFRec) プロジェクトに感謝しています。

## ライセンス (License)

このプロジェクトは、**GNU General Public License v3** の下でリリースされています。このライセンスは、ソフトウェアのコピー、配布、変更の自由を保証し、変更されたバージョンも元のバージョンと同じ自由を保持することを目的としています。詳細については、`LICENSE`ファイルを参照してください。

---