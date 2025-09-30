# 🎓 CEFR対応英文生成ワークフロー

`english_text_generation_workflow.py` は、CEFR-J に基づき読みやすさ指標を算出しつつ、2K〜7K の 6 段階難易度で英語読解テキスト（物語文・説明文）を自動生成するリファレンス実装です。語彙レベルと文構造の両面から制約を与え、各テキストに日本語の自由記述式設問を付与し、評価指標とともに CSV へ保存します。

## 🧩 システム概要
- **ReadabilityAnalyzer** が Excel 由来の学習データと CEFR-J 語彙リストから統計的プロファイルを構築し、spaCy と wordfreq/textstat を使って 8 指標を算出します。
- CEFR-J リストに存在しない内容語（固有名詞を除く）は自動的に C レベル相当とみなし、最大語彙ランク制約（2K〜7K）との乖離を把握します。
- **PromptBasedGenerator** が難易度バンド（2K〜7K）の語彙上限・話題・分量をプロンプトに織り込み、本文と日本語設問2問を JSON 形式で生成します。
- 各テキストは英語180〜220語に統一し、語彙帯と定量指標の乖離をレポートします。
- **IntegratedTextGenerator** が 6 バンド × 2 種（物語文・説明文）の 12 本を一括生成し、メトリクス評価と CSV 保存を行います。
- 生成直後に同じ指標で再評価し、ターゲットとの乖離（distance score）を返すため、品質のトレースが容易です。

## 🗂️ ディレクトリと関連ファイル
- `english_text_generation_workflow.py` — 本READMEの対象となる統合ワークフロー。
- `CVLA3_*.xlsx` — レベル別統計量を含む学習データ（必須）。
- `CEFR-J Wordlist Ver1.6.xlsx` — 語彙レベル参照（推奨）。
- `output_manager.py` — 生成物の保存ユーティリティ。
- `api_setup_guide.md` / `implementation_guide.md` — 補助ドキュメント。

## ⚙️ セットアップ
1. Python 3.9 以降を用意し、仮想環境を作成（任意）。
2. 依存ライブラリをインストール：
   ```bash
   pip install pandas numpy openai textstat spacy scikit-learn joblib wordfreq openpyxl python-dotenv
   python -m spacy download en_core_web_sm
   ```
3. プロジェクト直下の `.env` に API キーを記載（`python-dotenv` が読み込みます）：
   ```bash
   echo "OPENAI_API_KEY=sk-..." > .env
   ```
   ※ `.env` を使わずに環境変数 `export OPENAI_API_KEY='sk-...'` を直接設定しても動作します。
4. `CVLA3_*.xlsx`（レベル統計データ）と `CEFR-J Wordlist Ver1.6.xlsx` を `english_text_generation_workflow.py` から参照できる場所に置きます。

## 🚀 クイックスタート
```python
import os
from dotenv import load_dotenv
from english_text_generation_workflow import IntegratedTextGenerator

load_dotenv()

# システム初期化
generator = IntegratedTextGenerator(
    excel_path='CVLA3_20250912133649_3373.xlsx',
    api_key=os.getenv('OPENAI_API_KEY'),
    cefr_wordlist_path='CEFR-J Wordlist Ver1.6.xlsx'
)

# 例: 5Kバンドの物語文を単発生成
result = generator.generate(
    target_level='B1.2',
    topic='A teen journalist covering a local mystery',
    method='prompt',
    text_type='narrative',
    band_label='5K',
    max_rank=5000,
    word_count=210
)

print(result['text'])
print(result['questions'])
print(result['evaluation'])
```

## 🧠 プロンプト戦略（PromptBasedGenerator）
- レベルごとの**スタイルガイドライン**と**定量指標ターゲット**をプロンプトに埋め込み、CEFR-J 語彙を優先するよう圧力をかけます。
- 生成前に LLM にアウトライン思考を促し、最終段で本文と日本語設問2問を含む JSON を返させるワークフローで安定性を確保します。
- テキストタイプ（`narrative` / `expository`）専用の説明文を追加し、体裁を統制します。

## 🔁 反復改善ワークフロー（IterativeRefinementGenerator）
> ※ 標準の12本生成では使用しませんが、個別改善フローを検証する際に活用できます。
1. ざっくりした初期テキストを生成。
2. `ReadabilityAnalyzer` で 8 指標を再計算。
3. 目標プロファイルとの偏差から改善指示を組み立て、LLM に改稿させます。
4. 乖離スコアがしきい値（既定 0.1）を切るまで繰り返します。

## 🏗️ テンプレートベース生成
> ※ 簡易デモや A1 領域の固定文例を作る際に利用できます（標準バッチでは未使用）。
- A1 レベル向けに、語彙を穴埋めするだけで最低限の品質を担保できるテンプレートを収録。
- `content_dict` を渡せば、キャラクターや感情を簡単に差し替え可能。`IntegratedTextGenerator` がレベルに応じて自動利用します。

## 📐 評価指標
| 指標 | 説明 |
|------|------|
| `AvrDiff` | 内容語の平均 CEFR 難易度 (A1=1 … C2=6) |
| `BperA` | A レベル語に対する B レベル語の比率 |
| `CVV1` | 動詞の多様性指標 |
| `AvrFreqRank` | 語の頻度ランク（低いほど一般的） |
| `ARI` | Automated Readability Index |
| `VperSent` | 文あたりの動詞数 |
| `POStypes` | 文中に出現する品詞タイプ数 |
| `LenNP` | 名詞句の平均長 |


## 📦 出力ファイル
- `texts/` ディレクトリ: 各バンド・タイプごとの `.txt` と `.json` を格納（本文と日本語設問をパッケージ化）。
- `reading_band_set_texts.csv`: 欄 `level`（2K〜7K）, `type`（narrative/expository）, 英文本文 `text`, 日本語の自由記述式設問 `question1`, `question2` を収録。
- `reading_band_set_metrics.csv`: 各テキストの語数・文数・CEFR-J 指標 (`metric_*`)・ターゲットとの差 (`delta_*`)・語彙逸脱数 (`out_of_band_count`) などを一覧化。
- セッションフォルダ全体が ZIP 化されるため、必要に応じて他環境へ転送できます。

評価結果にはターゲット平均との差分 (`delta_from_target`) と正規化距離 (`distance_score`) が含まれるため、改善余地を定量的に把握できます。

## 📄 スクリプトの実行例
```bash
python english_text_generation_workflow.py
```
- 依存関係のインストールガイドが表示された後、生成セット数（標準12本×セット数）を尋ねられます。
- 指定したセット数だけ 2K〜7K × 物語/説明 のテキストを出力します（各テキストは180〜220語に揃えます）。
- 本文と日本語設問2問がコンソールに出力され、`outputs/session_*/reading_band_set_texts.csv` にテキスト、`*_metrics.csv` に指標が保存されます。
- セッションフォルダ全体が ZIP にまとめられるため、成果物一式を容易に共有できます。

## 🧾 データとセキュリティ上の注意
- Excel データには CEFR レベルごとの統計量が含まれるため、アクセス権限を管理してください。
- API キーは `.env` などで安全に保存し、バージョン管理に含めないでください。

## 🔄 今後の拡張案
1. `MLBasedGenerator` で実際の特徴抽出・推論パイプラインを整備し、レベル推定を生成と独立に評価する。
2. A2 以上向けテンプレートや、語彙制御付きのテンプレート自動生成を追加する。
3. 乖離スコアを利用した自動停止条件やログ収集を `output_manager.py` と統合する。

## 📝 ライセンスと貢献
- ライセンス: MIT
- 改善アイデアやバグ報告は Issue / PR でお寄せください。

---

**Happy Level-Controlled Writing! ✨**
