#!/bin/bash

# ========================================
# ハイブリッド型英文生成システム
# クイックセットアップスクリプト
# ========================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ハイブリッド型英文生成システム - セットアップ          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Python バージョンチェック
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "📌 Pythonバージョン: $python_version"

if [ $(echo "$python_version < 3.7" | bc) -eq 1 ]; then
    echo "❌ Python 3.7以上が必要です"
    exit 1
fi

# セットアップオプション選択
echo ""
echo "セットアップオプションを選択してください:"
echo "1) デモ版（APIキー不要）"
echo "2) OpenAI版（GPT-4/GPT-3.5）"
echo "3) Google Gemini版（無料枠あり）"
echo "4) Claude版（Anthropic）"
echo "5) フル機能版（全API対応）"
echo ""
read -p "選択 (1-5): " setup_choice

# 仮想環境の作成
echo ""
echo "🔧 仮想環境を作成中..."
python3 -m venv venv
source venv/bin/activate

# 基本ライブラリのインストール
echo "📦 基本ライブラリをインストール中..."
pip install --upgrade pip
pip install pandas numpy openpyxl scikit-learn

case $setup_choice in
    1)
        echo "📦 デモ版のセットアップ..."
        # 最小限のライブラリのみ
        ;;
    2)
        echo "📦 OpenAI版のセットアップ..."
        pip install openai textstat spacy
        python -m spacy download en_core_web_sm
        
        echo ""
        echo "🔑 OpenAI APIキーを入力してください:"
        echo "（https://platform.openai.com/api-keys で取得）"
        read -s openai_key
        echo ""
        
        # .envファイルの作成
        echo "OPENAI_API_KEY=$openai_key" > .env
        echo "✅ APIキーを.envファイルに保存しました"
        ;;
    3)
        echo "📦 Google Gemini版のセットアップ..."
        pip install google-generativeai textstat spacy
        python -m spacy download en_core_web_sm
        
        echo ""
        echo "🔑 Google AI APIキーを入力してください:"
        echo "（https://makersuite.google.com/app/apikey で取得）"
        read -s google_key
        echo ""
        
        echo "GOOGLE_API_KEY=$google_key" > .env
        echo "✅ APIキーを.envファイルに保存しました"
        ;;
    4)
        echo "📦 Claude版のセットアップ..."
        pip install anthropic textstat spacy
        python -m spacy download en_core_web_sm
        
        echo ""
        echo "🔑 Anthropic APIキーを入力してください:"
        echo "（https://console.anthropic.com/ で取得）"
        read -s anthropic_key
        echo ""
        
        echo "ANTHROPIC_API_KEY=$anthropic_key" > .env
        echo "✅ APIキーを.envファイルに保存しました"
        ;;
    5)
        echo "📦 フル機能版のセットアップ..."
        pip install openai anthropic google-generativeai textstat spacy python-dotenv
        python -m spacy download en_core_web_sm
        
        echo ""
        echo "🔑 APIキーを設定します（使用しないものは空欄でEnter）"
        
        echo "OpenAI APIキー:"
        read -s openai_key
        echo "Claude APIキー:"
        read -s anthropic_key
        echo "Google AI APIキー:"
        read -s google_key
        echo ""
        
        # .envファイルの作成
        {
            [ ! -z "$openai_key" ] && echo "OPENAI_API_KEY=$openai_key"
            [ ! -z "$anthropic_key" ] && echo "ANTHROPIC_API_KEY=$anthropic_key"
            [ ! -z "$google_key" ] && echo "GOOGLE_API_KEY=$google_key"
        } > .env
        
        echo "✅ APIキーを.envファイルに保存しました"
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac

# 実行スクリプトの作成
cat > run.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python3 -c "
import os
import sys

# .envファイルがあれば読み込む
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# 適切なスクリプトを実行
if any(key in os.environ for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
    print('🚀 フル機能版を起動中...')
    exec(open('hybrid_approach_setup.py').read())
else:
    print('🚀 デモ版を起動中...')
    exec(open('demo_system.py').read())
"
EOF

chmod +x run.sh

# 完了メッセージ
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅ セットアップ完了！                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "実行方法:"
echo "  ./run.sh          # システムを起動"
echo ""
echo "または:"
echo "  source venv/bin/activate"
echo "  python demo_system.py      # デモ版"
echo "  python hybrid_approach_setup.py  # フル機能版"
echo ""
echo "注意事項:"
echo "- Excelファイル（CVLA3_20250912133649_3373.xlsx）を同じディレクトリに配置してください"
echo "- APIキーは.envファイルに保存されています"
echo "- 詳細は api_setup_guide.md を参照してください"
