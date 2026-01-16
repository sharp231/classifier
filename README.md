
下記がプロジェクトの解説。

````
# End-to-End Image Classification App (CIFAR-10)

画像認識モデルの学習からWebアプリケーションとしてのデプロイまでを、End-to-Endで実装したプロジェクトです。
ユーザーがアップロードした画像をAIが解析し、10種類のクラス（飛行機、猫、犬など）に分類します。

## 🚀 プロジェクト概要
機械学習モデルを単に作成するだけでなく、実際のWebサービスとして稼働させるためのアーキテクチャ設計と実装を行いました。
入力画像の揺らぎ（回転、アスペクト比）吸収や、推論精度の安定化（TTA）など、実運用を想定した工夫を取り入れています。

* **デモ:** [RenderでのデプロイURLをここに記載]
* **開発期間:** [2ヶ月]
* **担当領域:** 企画・要件定義 / モデル構築 / Web実装 / デプロイ (Full Cycle)

## ✨ 技術的なこだわり (Key Features)

### 1. 推論精度の安定化 (Test Time Augmentation)
単一の推論ではなく、入力画像を複数の倍率（1.0, 0.8, 0.65）でクロップして推論し、その平均値を取るアンサンブル処理を実装しています。これにより、被写体の大きさや位置ズレに強い判定を実現しました。

### 2. 堅牢な前処理パイプライン
ユーザーが投稿する多様な画像に対応するための処理を実装しています。
* **EXIF自動補正:** スマホ写真の回転情報を読み取り、正しい向きに補正。
* **Center Crop:** 単純なリサイズによる画像の歪みを防ぐため、アスペクト比を維持したまま中央を切り抜く処理を採用。

### 3. モダンなUI/UX
* **Tailwind CSS:** ユーティリティファーストなCSSフレームワークを採用し、レスポンシブなデザインを構築。
* **インタラクション:** ドラッグ＆ドロップ対応のアップロード領域と、解析中のローディング表示による体感速度の向上。

### 4. 安定性を重視したアーキテクチャ
* **遅延ロード (Lazy Loading):** サーバー起動時のメモリ負荷を抑えるため、モデルのロードを初回リクエスト時（または明示的なタイミング）に行う設計を採用。
* **エラーハンドリング:** 画像形式エラーやモデルロード失敗時に、サーバーをクラッシュさせずに適切なメッセージを返す安全設計。

## 🛠 技術スタック (Tech Stack)

| Category | Technology |
| :--- | :--- |
| **ML / DL** | Python 3, TensorFlow, Keras (CNN Model) |
| **Backend** | Flask (Web API & Model Serving) |
| **Frontend** | HTML5, Tailwind CSS, JavaScript (jQuery) |
| **Image Proc** | Pillow (PIL), NumPy |
| **Infrastructure** | Render (PaaS) |

## 📂 ディレクトリ構成

```text
.
├── classifier.py      # Flaskアプリケーションエントリーポイント（推論API）
├── train_cifar10.py   # モデル学習用スクリプト
├── karas.py           # 前処理ロジック（EXIF補正等）
├── image_classifier.h5 # 学習済みモデル（バイナリ）
├── templates/         # フロントエンド（HTML）
│   ├── index.html     # アップロード画面
│   ├── result.html    # 結果表示画面
│   └── layout.html    # 共通レイアウト
└── static/            # 静的ファイル（CSS, 画像保存先）

````

## ⚙️ ローカルでの実行方法

```bash
# 1. リポジトリのクローン
git clone [repository_url]
cd [project_name]

# 2. 依存ライブラリのインストール
pip install -r requirements.txt

# 3. モデルの学習（学習済みモデルがない場合）
python train_cifar10.py

# 4. アプリケーションの起動
python classifier.py

```

ブラウザで `http://127.0.0.1:5000` にアクセスしてください。

## 🔍 今後の展望 (Future Roadmap)

- GitHub Actions を用いた CI/CD パイプラインの構築
- フィードバック機能による継続的なデータ収集と再学習ループ（MLOps）の構築
- 物体検出モデル（YOLO 等）へのリプレイス検討
- Flask をフロント/BFF、FastAPI をバックエンドは、「Python だけで完結させつつ、それぞれのフレームワークの得意分野を活かせる非常に良いアーキテクチャです。
  この構成では、
  Flask (BFF): ユーザーとの対話、HTML 表示、API へのつなぎ込みを担当。
  FastAPI (Backend): 重い計算（AI 推論）、TTA（精度向上処理）を担当。

````

---

### 補足：`requirements.txt` について

上記の `README.md` を機能させるために、以下の内容で `requirements.txt` というファイルをプロジェクトのルートに作成して一緒に保存してください。（Render へのデプロイ時にも必須となります）

**requirements.txt**

```text
Flask
tensorflow
# tensorflow-cpu  # ※Renderの無料枠等で容量制限が厳しい場合はこちらを検討
Pillow
numpy
gunicorn
werkzeug

```
````
