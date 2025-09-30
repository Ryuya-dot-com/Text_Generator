# 🚀 ハイブリッド型英文生成システム - APIセットアップガイド

## 📌 クイックスタート（5分で開始）

### オプション1: OpenAI GPT-4

**料金**: 約$0.03-0.06/1000トークン  
**品質**: 最高  
**速度**: 高速

```bash
# 1. APIキーを取得
# https://platform.openai.com/api-keys にアクセス
# "Create new secret key"をクリック

# 2. 環境変数に設定
export OPENAI_API_KEY='sk-...'  # Mac/Linux
set OPENAI_API_KEY=sk-...       # Windows

# 3. ライブラリインストール
pip install openai pandas scikit-learn textstat spacy

# 4. 実行
python hybrid_approach_setup.py
```

### オプション2: Google Gemini

**料金**: 無料（60リクエスト/分まで）  
**品質**: 良好  
**速度**: 高速

```bash
# 1. APIキーを取得
# https://makersuite.google.com/app/apikey にアクセス
# "Create API Key"をクリック

# 2. 環境変数に設定
export GOOGLE_API_KEY='...'

# 3. ライブラリインストール
pip install google-generativeai pandas scikit-learn textstat spacy

# 4. 実行
python hybrid_approach_setup.py
```

### オプション3: Claude API

**料金**: 約$0.015-0.075/1000トークン  
**品質**: 最高  
**速度**: 高速

```bash
# 1. APIキーを取得
# https://console.anthropic.com/ にアクセス
# API keysセクションで新規作成

# 2. 環境変数に設定
export ANTHROPIC_API_KEY='sk-ant-...'

# 3. ライブラリインストール
pip install anthropic pandas scikit-learn textstat spacy

# 4. 実行
python hybrid_approach_setup.py
```

## 🎯 即座に試せるコード

```python
# quick_test.py
import os
from hybrid_approach_setup import HybridTextGenerator

# APIキーを直接指定（テスト用）
generator = HybridTextGenerator(
    excel_path='CVLA3_20250912133649_3373.xlsx',
    llm_provider='openai',  # または 'gemini', 'claude'
    api_key='your-api-key-here'
)

# テキスト生成
result = generator.generate_with_validation(
    target_level='B1.1',
    topic='Climate Change',
    word_count=200
)

print(f"生成テキスト: {result['text']}")
print(f"予測レベル: {result['actual_level']}")
print(f"成功: {result['success']}")
```

## 📊 API比較表

| プロバイダー | 料金（1000語あたり） | 品質 | 速度 | 無料枠 | 特徴 |
|------------|-------------------|------|------|--------|------|
| **OpenAI GPT-4** | $0.30-0.60 | ★★★★★ | 高速 | なし | 最高品質、安定性抜群 |
| **GPT-3.5-turbo** | $0.015-0.02 | ★★★★ | 超高速 | なし | コスパ最良 |
| **Claude 3 Opus** | $0.15-0.75 | ★★★★★ | 高速 | なし | 長文に強い |
| **Claude 3 Sonnet** | $0.03-0.15 | ★★★★ | 高速 | なし | バランス型 |
| **Google Gemini** | 無料～$0.35 | ★★★★ | 高速 | 60/分 | 無料枠が魅力 |
| **ローカルLLM** | 無料 | ★★★ | 遅い | 無制限 | プライバシー重視 |

## 🔧 詳細セットアップ

### ステップ1: 基本環境の準備

```bash
# Python仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 基本ライブラリをインストール
pip install pandas numpy scikit-learn openpyxl
pip install textstat spacy
python -m spacy download en_core_web_sm
```

### ステップ2: APIの選択と設定

#### A. OpenAI（最も簡単・推奨）

1. **アカウント作成**
   - https://platform.openai.com/signup
   - クレジットカード登録が必要

2. **APIキー取得**
   ```
   1. https://platform.openai.com/api-keys
   2. "+ Create new secret key"をクリック
   3. キーをコピー（sk-...で始まる文字列）
   ```

3. **料金設定**
   - 使用量上限を設定: https://platform.openai.com/account/limits
   - 推奨: 月$10-20の上限設定

4. **Pythonコードでの使用**
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'sk-...'
   
   # または.envファイルを使用
   # OPENAI_API_KEY=sk-...
   ```

#### B. Google Gemini（無料で開始）

1. **Google AIアカウント**
   - https://makersuite.google.com/
   - Googleアカウントでログイン

2. **APIキー取得**
   ```
   1. "Get API key"をクリック
   2. "Create API key in new project"
   3. キーをコピー
   ```

3. **制限事項**
   - 無料: 60リクエスト/分
   - 有料プランで制限解除可能

#### C. Claude（高品質な出力）

1. **Anthropicアカウント**
   - https://console.anthropic.com/
   - メールで登録

2. **APIキー取得**
   ```
   1. Settingsメニュー
   2. API Keys
   3. "Create Key"
   ```

3. **特徴**
   - 長文生成に強い
   - 指示への忠実性が高い

### ステップ3: 環境変数の永続設定

#### Mac/Linux
```bash
# ~/.bashrc または ~/.zshrc に追加
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

#### Windows
```powershell
# システム環境変数として設定
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY','sk-...','User')
```

#### Python dotenvを使用（推奨）
```bash
pip install python-dotenv
```

`.env`ファイルを作成:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

```python
from dotenv import load_dotenv
load_dotenv()
```

## 🧪 動作確認

### テストスクリプト

```python
# test_api.py
import os
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

def test_openai():
    """OpenAI APIのテスト"""
    try:
        import openai
        from openai import OpenAI
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        print("✅ OpenAI API: 正常動作")
        return True
    except Exception as e:
        print(f"❌ OpenAI API: {e}")
        return False

def test_gemini():
    """Gemini APIのテスト"""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say hello")
        print("✅ Gemini API: 正常動作")
        return True
    except Exception as e:
        print(f"❌ Gemini API: {e}")
        return False

def test_claude():
    """Claude APIのテスト"""
    try:
        import anthropic
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        print("✅ Claude API: 正常動作")
        return True
    except Exception as e:
        print(f"❌ Claude API: {e}")
        return False

if __name__ == "__main__":
    print("APIテストを実行中...\n")
    test_openai()
    test_gemini()
    test_claude()
```

## 💰 コスト計算例

### 1日100テキスト生成の場合

| プロバイダー | 1テキスト（200語） | 1日（100テキスト） | 1ヶ月 |
|------------|------------------|-------------------|--------|
| GPT-4 | $0.06 | $6.00 | $180 |
| GPT-3.5 | $0.004 | $0.40 | $12 |
| Claude Opus | $0.075 | $7.50 | $225 |
| Claude Sonnet | $0.015 | $1.50 | $45 |
| Gemini | 無料 | 無料 | 無料* |

*Geminiは60リクエスト/分の制限内

## 🚨 トラブルシューティング

### よくある問題と解決策

**1. "API key not found"エラー**
```bash
# 環境変数が設定されているか確認
echo $OPENAI_API_KEY  # Mac/Linux
echo %OPENAI_API_KEY%  # Windows

# Pythonで確認
import os
print(os.getenv('OPENAI_API_KEY'))
```

**2. "Rate limit exceeded"エラー**
- 解決: time.sleep()で遅延を追加
- またはレート制限の高いプランにアップグレード

**3. "Model not found"エラー**
- GPT-4を使用するには課金履歴が必要
- 最初はgpt-3.5-turboを使用

**4. spaCyモデルエラー**
```bash
python -m spacy download en_core_web_sm
# または
python -m spacy download en_core_web_md  # より高精度
```

## 📝 実装チェックリスト

- [ ] Pythonバージョン3.7以上を確認
- [ ] 仮想環境を作成・有効化
- [ ] 必要なライブラリをインストール
- [ ] APIプロバイダーを選択
- [ ] APIキーを取得
- [ ] 環境変数に設定
- [ ] テストスクリプトで動作確認
- [ ] Excelデータファイルを配置
- [ ] ハイブリッドシステムを実行

## 🎓 次のステップ

1. **小規模テスト**: 5-10個のテキストで動作確認
2. **パラメータ調整**: プロンプトと生成パラメータの最適化
3. **大規模生成**: バッチ処理で大量のテキストを生成
4. **評価**: 生成されたテキストの品質評価
5. **デプロイ**: Webアプリケーション化

## 📧 サポート

問題が解決しない場合のリソース：
- OpenAI: https://help.openai.com/
- Anthropic: https://docs.anthropic.com/
- Google AI: https://ai.google.dev/tutorials

---

準備ができたら、`python hybrid_approach_setup.py`を実行してください！
