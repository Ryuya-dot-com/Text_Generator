@echo off
REM ========================================
REM ハイブリッド型英文生成システム
REM Windowsセットアップスクリプト
REM ========================================

echo ============================================================
echo    ハイブリッド型英文生成システム - Windowsセットアップ
echo ============================================================
echo.

REM Pythonバージョンチェック
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pythonがインストールされていません
    echo https://www.python.org/downloads/ からインストールしてください
    pause
    exit /b 1
)

echo Pythonバージョン:
python --version
echo.

REM セットアップオプション選択
echo セットアップオプションを選択してください:
echo 1) デモ版（APIキー不要）
echo 2) OpenAI版（GPT-4/GPT-3.5）
echo 3) Google Gemini版（無料枠あり）
echo 4) Claude版（Anthropic）
echo 5) フル機能版（全API対応）
echo.
set /p setup_choice="選択 (1-5): "

REM 仮想環境の作成
echo.
echo 仮想環境を作成中...
python -m venv venv
call venv\Scripts\activate.bat

REM 基本ライブラリのインストール
echo.
echo 基本ライブラリをインストール中...
pip install --upgrade pip
pip install pandas numpy openpyxl scikit-learn

if "%setup_choice%"=="1" (
    echo デモ版のセットアップ完了
    goto :create_run_script
)

if "%setup_choice%"=="2" (
    echo OpenAI版のセットアップ中...
    pip install openai textstat spacy python-dotenv
    python -m spacy download en_core_web_sm
    
    echo.
    set /p openai_key="OpenAI APIキーを入力: "
    echo OPENAI_API_KEY=%openai_key% > .env
    echo APIキーを.envファイルに保存しました
    goto :create_run_script
)

if "%setup_choice%"=="3" (
    echo Google Gemini版のセットアップ中...
    pip install google-generativeai textstat spacy python-dotenv
    python -m spacy download en_core_web_sm
    
    echo.
    set /p google_key="Google AI APIキーを入力: "
    echo GOOGLE_API_KEY=%google_key% > .env
    echo APIキーを.envファイルに保存しました
    goto :create_run_script
)

if "%setup_choice%"=="4" (
    echo Claude版のセットアップ中...
    pip install anthropic textstat spacy python-dotenv
    python -m spacy download en_core_web_sm
    
    echo.
    set /p anthropic_key="Anthropic APIキーを入力: "
    echo ANTHROPIC_API_KEY=%anthropic_key% > .env
    echo APIキーを.envファイルに保存しました
    goto :create_run_script
)

if "%setup_choice%"=="5" (
    echo フル機能版のセットアップ中...
    pip install openai anthropic google-generativeai textstat spacy python-dotenv
    python -m spacy download en_core_web_sm
    
    echo.
    echo APIキーを設定します（使用しないものは空欄でEnter）
    set /p openai_key="OpenAI APIキー: "
    set /p anthropic_key="Claude APIキー: "
    set /p google_key="Google AI APIキー: "
    
    echo. > .env
    if not "%openai_key%"=="" echo OPENAI_API_KEY=%openai_key% >> .env
    if not "%anthropic_key%"=="" echo ANTHROPIC_API_KEY=%anthropic_key% >> .env
    if not "%google_key%"=="" echo GOOGLE_API_KEY=%google_key% >> .env
    
    echo APIキーを.envファイルに保存しました
    goto :create_run_script
)

echo 無効な選択です
pause
exit /b 1

:create_run_script
REM 実行用バッチファイルの作成
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo python demo_system.py >> run.bat
echo pause >> run.bat

REM 完了メッセージ
echo.
echo ============================================================
echo    セットアップ完了！
echo ============================================================
echo.
echo 実行方法:
echo   run.bat をダブルクリック
echo.
echo または:
echo   venv\Scripts\activate.bat
echo   python demo_system.py
echo.
echo 注意事項:
echo - Excelファイルを同じディレクトリに配置してください
echo - APIキーは.envファイルに保存されています
echo.
pause
