"""
ハイブリッド型英文生成システム
Approach 3: ML Model + LLM Integration
"""

import os
import random
import re
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import pickle
from datetime import datetime

# ----------------------------------------
# 小さなユーティリティ
# ----------------------------------------
def _slug(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"[^a-z0-9\- _]+", "", t)
    t = t.replace(" ", "_")
    t = re.sub(r"_+", "_", t)
    return t or "text"

def _norm_user_path(p: str) -> str:
    # ユーザーがシェル風に空白を \ でエスケープした入力を正規化
    if not p:
        return p
    p = p.strip().strip('"').strip("'")
    p = p.replace('\\ ', ' ')
    return p

# ----------------------------------------
# 固定パス（ユーザー指定の絶対パスをコードに組み込み）
# ----------------------------------------
DEFAULT_EXCEL_PATH = \
    "/Users/ryuya/Library/CloudStorage/Dropbox/科研_CAT/Material/Text_Generation/CVLA3_20250912133649_3373.xlsx"
DEFAULT_WORDLIST_PATH = \
    "/Users/ryuya/Library/CloudStorage/Dropbox/科研_CAT/Material/Text_Generation/CEFR-J Wordlist Ver1.6.xlsx"
DEFAULT_READING_TEXT_DIR = \
    "/Users/ryuya/Library/CloudStorage/Dropbox/科研_CAT/Material/Reading_Text"

# モデルで用いる特徴量の列名（統一定義）
FEATURE_NAMES = ['ARI', 'AvrDiff', 'BperA', 'CVV1', 'AvrFreqRank', 'VperSent', 'POStypes', 'LenNP']

# ========================================
# STEP 1: API設定
# ========================================

class APIConfiguration:
    """OpenAI API設定の管理クラス（OpenAI専用）"""
    
    def __init__(self):
        self.config = {}
        self.setup_apis()
    
    def setup_apis(self):
        """OpenAIのみ設定"""
        self.config['openai'] = {
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'model': 'gpt-4',  # 予算に応じて変更可能
            'endpoint': 'https://api.openai.com/v1/chat/completions',
            'cost_per_1k_tokens': {
                'gpt-4': {'input': 0.03, 'output': 0.06},
                'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
            }
        }
    
    def get_api_key(self) -> str:
        """OpenAI APIキーを取得"""
        return self.config['openai'].get('api_key', '')
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> float:
        """OpenAIのコストを推定"""
        model = model or self.config['openai'].get('model', '')
        costs = self.config['openai'].get('cost_per_1k_tokens', {})
        if model in costs:
            input_cost = (input_tokens / 1000) * costs[model]['input']
            output_cost = (output_tokens / 1000) * costs[model]['output']
            return input_cost + output_cost
        return 0.0

# ========================================
# STEP 2: API別の実装
# ========================================

class LLMProvider:
    """OpenAI専用のLLMプロバイダー実装"""
    
    def __init__(self):
        self.api_config = APIConfiguration()
        self.setup_provider()
    
    def setup_provider(self):
        """OpenAIの初期設定"""
        api_key = self.api_config.get_api_key()
        self._setup_openai(api_key)
    
    def _setup_openai(self, api_key: str):
        """OpenAI APIの設定"""
        try:
            import openai
            from openai import OpenAI
            
            if not api_key:
                print("⚠️ OpenAI APIキーが設定されていません")
                print("設定方法:")
                print("1. https://platform.openai.com/api-keys でAPIキーを取得")
                print("2. 環境変数に設定: export OPENAI_API_KEY='your-key-here'")
                print("3. または直接指定: client = OpenAI(api_key='your-key-here')")
                return None
            
            self.client = OpenAI(api_key=api_key)
            print("✅ OpenAI API設定完了")
            
        except ImportError as e:
            print(f"OpenAIライブラリの読み込みに失敗しました: {e}")
            print("インストール/更新コマンド: pip install -U openai")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """テキスト生成の統一インターフェース"""
        return self._generate_openai(prompt, max_tokens, temperature)
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """OpenAIでの生成"""
        if not hasattr(self, 'client'):
            return "[OpenAI API未設定] デモテキスト"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in creating CEFR-leveled English texts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"エラー: {str(e)}"
    
    def classify_level(self, text: str, candidate_levels: Optional[List[str]] = None) -> Tuple[str, float]:
        """OpenAIにCEFRレベル判定を依頼（フォールバック用）"""
        if not hasattr(self, 'client'):
            return "unknown", 0.0
        levels = candidate_levels or ['A1.1','A1.2','A2.2','B1.1','B1.2','B2.1','B2.2','C1','C2']
        try:
            prompt = (
                "Classify the CEFR level of the following English text.\n"
                f"Choose one EXACTLY from: {', '.join(levels)}.\n"
                "Respond in JSON: {\"level\": <one of levels>, \"confidence\": <0-1 float>}\n\n"
                f"Text:\n{text}"
            )
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise CEFR evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.0
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            level = str(data.get("level", "unknown"))
            conf = float(data.get("confidence", 0.0))
            if level not in levels:
                return "unknown", conf
            return level, conf
        except Exception:
            return "unknown", 0.0
    

# ========================================
# STEP 3: MLモデルの実装
# ========================================

class ReadabilityPredictor:
    """読みやすさレベル予測モデル"""
    
    def __init__(self, excel_path: str):
        self.df = None
        if excel_path and os.path.exists(excel_path):
            try:
                self.df = pd.read_excel(excel_path)
            except Exception as e:
                print(f"⚠️ Excelの読み込みでエラーが発生しました: {e}")
                self.df = None
        else:
            if excel_path:
                print(f"⚠️ 指定されたExcelファイルが見つかりません: {excel_path}")
            else:
                print("ℹ️ Excelファイルが指定されていません。モデル学習はスキップされます。")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = list(FEATURE_NAMES)
    
    def train_model(self):
        """モデルを訓練"""
        if self.df is None:
            print("⏭️ 学習データがないため、モデル学習をスキップします。")
            return 0.0

        # 必須カラムの存在確認
        required_cols = set(self.feature_names + ['predict_level'])
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            print(f"⚠️ 学習データに必要なカラムが不足しています: {missing}")
            print("⏭️ モデル学習をスキップします。")
            return 0.0
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        
        print("📊 MLモデルを訓練中...")
        
        # データ準備
        X = self.df[self.feature_names].values
        y = self.df['predict_level'].values
        
        # ラベルエンコーディング
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # クラス分布を確認
        unique, counts = np.unique(y_encoded, return_counts=True)
        min_class_count = counts.min() if len(counts) else 0
        use_stratify = min_class_count >= 2
        if not use_stratify:
            print("⚠️ 一部クラスのサンプル数が1件のため、層化分割を無効化します。")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42,
            stratify=y_encoded if use_stratify else None
        )
        
        # 特徴量の正規化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 候補モデルを比較し、確率較正したモデルを採用
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        lr = LogisticRegression(
            max_iter=2000,
            multi_class='ovr',
            class_weight='balanced',
            solver='liblinear'
        )
        base_model = rf
        try:
            # cv分割数は各クラスの最小サンプル数に依存
            n_splits = min(5, int(min_class_count))
            if n_splits >= 2 and use_stratify:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores_rf = cross_val_score(rf, X_train_scaled, y_train, cv=cv)
                scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=cv)
                base_model = rf if scores_rf.mean() >= scores_lr.mean() else lr
                chosen = 'RF' if base_model is rf else 'LR'
                print(f"🔎 基本モデル選択: {chosen} (cv={max(scores_rf.mean(), scores_lr.mean()):.2%})")
            else:
                print("ℹ️ クラス数が少ないためCV評価をスキップし、RFを使用します。")
        except Exception as e:
            print(f"ℹ️ CV評価をスキップ（理由: {e}）。RFを使用します。")
            base_model = rf

        base_model.fit(X_train_scaled, y_train)
        # 確率較正（データが十分なとき）
        try:
            calib_splits = max(2, min(3, int(min_class_count))) if use_stratify else 2
            if calib_splits >= 2:
                self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=calib_splits)
                self.model.fit(X_train_scaled, y_train)
            else:
                self.model = base_model
                print("ℹ️ サンプル不足のため確率較正をスキップします。")
        except Exception:
            self.model = base_model
        
        # 精度評価
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ モデル訓練完了！精度: {accuracy:.2%}")
        
        # 特徴量の重要度
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📈 特徴量の重要度:")
        for _, row in importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")
        
        return accuracy
    
    def predict_level(self, features: Dict[str, float]) -> Tuple[str, float]:
        """レベルを予測"""
        if not self.model:
            return "B1.1", 0.5  # デフォルト値
        
        # 特徴量ベクトルを作成
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # 予測
        y_pred = self.model.predict(X_scaled)[0]
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        level = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = max(y_proba)
        
        return level, confidence
    
    def save_model(self, filepath: str = 'readability_model.pkl'):
        """モデルを保存"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
        print(f"💾 モデルを {filepath} に保存しました")
    
    def load_model(self, filepath: str = 'readability_model.pkl'):
        """モデルを読み込み"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_names = data['feature_names']
        print(f"📂 モデルを {filepath} から読み込みました")

# ========================================
# STEP 4: 特徴量抽出
# ========================================

class FeatureExtractor:
    """テキストから特徴量を抽出（CEFR-J Wordlist対応）"""
    
    def __init__(self, cefr_wordlist_path: Optional[str] = None):
        self.cefr_wordlist_path = cefr_wordlist_path or os.getenv('CEFR_WORDLIST_PATH', '')
        self.word_level_map: Dict[str, str] = {}
        self.setup_tools()
        # Wordlist 自動検出
        if not self.cefr_wordlist_path:
            for c in [
                'CEFR-J Wordlist Ver1.6.xlsx',
                'CEFR-J Wordlist.xlsx',
                'CEFR_J_Wordlist.xlsx'
            ]:
                if os.path.exists(c):
                    self.cefr_wordlist_path = c
                    break
        if self.cefr_wordlist_path and os.path.exists(self.cefr_wordlist_path):
            self._load_cefr_wordlist(self.cefr_wordlist_path)
        else:
            if self.cefr_wordlist_path:
                print(f"⚠️ CEFR-J Wordlistが見つかりません: {self.cefr_wordlist_path}")
            else:
                print("ℹ️ CEFR-J Wordlistのパスが未指定です（環境変数 CEFR_WORDLIST_PATH で指定可）")
    
    def setup_tools(self):
        """必要なツールをセットアップ"""
        try:
            import textstat
            import spacy
            
            # spaCyモデルのダウンロード
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("spaCyモデルをダウンロード中...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            self.textstat = textstat
            # 任意: 頻度推定用の wordfreq
            try:
                import wordfreq as _wf
                self.wordfreq = _wf
            except Exception:
                self.wordfreq = None
                print("ℹ️ wordfreq 未導入のため AvrFreqRank は近似値になります（pip install wordfreq で導入可）")
            print("✅ 特徴量抽出ツール準備完了")
            
        except ImportError as e:
            print(f"必要なライブラリをインストールしてください: {e}")
            print("pip install textstat spacy")
    
    def _load_cefr_wordlist(self, path: str):
        """CEFR-J Wordlist（Excel）を読み込み、見出し語→レベルの辞書を作成"""
        try:
            xls = pd.ExcelFile(path)
        except Exception as e:
            print(f"⚠️ CEFR-J Wordlistの読み込みに失敗: {e}")
            return
        # すべてのシートを走査して候補列を探索
        candidates = []
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet)
            except Exception:
                continue
            lower = {c.lower(): c for c in df.columns}
            def find_col(keys):
                for k in keys:
                    if k in lower:
                        return lower[k]
                return None
            word_col = find_col(['headword','lemma','lemmas','word','単語','見出し語'])
            level_col = find_col(['cefr-j','cefr','level','レベル','cefr level','cefr-j level','level (cefr)'])
            if word_col and level_col:
                candidates.append((sheet, df, word_col, level_col))
        if not candidates:
            print("⚠️ CEFR-J Wordlistの列名をいずれのシートでも検出できませんでした。シート構成をご確認ください。")
            print(f" 利用可能シート: {xls.sheet_names}")
            return
        # 最初に見つかった候補を使用
        sheet, df, word_col, level_col = candidates[0]
        def norm_word(w: str) -> str:
            return str(w).strip().lower()
        def norm_level(lbl: str) -> str:
            s = str(lbl).strip().upper().replace(' ', '')
            m = re.match(r'([ABC][12])', s)
            return m.group(1) if m else s
        mapping = {}
        for _, row in df[[word_col, level_col]].dropna().iterrows():
            w = norm_word(row[word_col])
            lv = norm_level(row[level_col])
            if w and lv:
                mapping[w] = lv
        self.word_level_map = mapping
        print(f"📚 CEFR-J Wordlistを読み込みました（シート: {sheet}）: {len(self.word_level_map)}語")
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """テキストから特徴量を抽出"""
        
        if not hasattr(self, 'textstat'):
            # ダミーデータを返す
            return {
                'ARI': 5.0,
                'AvrDiff': 1.5,
                'BperA': 0.3,
                'CVV1': 4.0,
                'AvrFreqRank': 1500,
                'VperSent': 2.5,
                'POStypes': 8.0,
                'LenNP': 3.5
            }
        
        doc = self.nlp(text)
        
        # 各特徴量を計算
        features = {}
        
        # ARI (Automated Readability Index)
        features['ARI'] = self.textstat.automated_readability_index(text)
        
        # AvrDiff（内容語のみ、CEFR-Jレベルの平均: A1=1..C2=6）
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        lv_nums = []
        for tok in doc:
            if tok.is_alpha and tok.pos_ in content_pos:
                lbl = self._lookup_level(tok.lemma_.lower())
                if lbl:
                    lv_nums.append(self._level_to_num(lbl))
        features['AvrDiff'] = float(np.mean(lv_nums)) if lv_nums else 0.0
        
        # BperA（Bレベル/ Aレベル の内容語比）
        a_cnt = 0
        b_cnt = 0
        for tok in doc:
            if tok.is_alpha and tok.pos_ in content_pos:
                lbl = self._lookup_level(tok.lemma_.lower())
                if not lbl:
                    continue
                if lbl.startswith('A'):
                    a_cnt += 1
                elif lbl.startswith('B'):
                    b_cnt += 1
        features['BperA'] = (b_cnt / a_cnt) if a_cnt > 0 else 0.0
        
        # CVV1 = unique verbs / sqrt(2 * total verbs)
        verbs = [t for t in doc if t.pos_ == 'VERB']
        v_total = len(verbs)
        v_types = len(set(t.lemma_.lower() for t in verbs))
        features['CVV1'] = (v_types / np.sqrt(2 * v_total)) if v_total > 0 else 0.0
        
        # AvrFreqRank（機能語含む、最も低頻度の3語を除外）
        if getattr(self, 'wordfreq', None):
            words = [t.text.lower() for t in doc if t.is_alpha]
            if words:
                zipfs = [(w, self.wordfreq.zipf_frequency(w, 'en')) for w in words]
                zipfs.sort(key=lambda x: x[1])  # 低→高
                trimmed = zipfs[3:] if len(zipfs) > 3 else zipfs
                ranks = [10.0 - z for _, z in trimmed]  # 小さいほど高頻度
                features['AvrFreqRank'] = float(np.mean(ranks)) if ranks else 0.0
            else:
                features['AvrFreqRank'] = 0.0
        else:
            features['AvrFreqRank'] = max(0.0, 10.0 - features['ARI'])
        
        # 文あたりの動詞数
        sentences = list(doc.sents)
        features['VperSent'] = (len(verbs) / len(sentences)) if sentences else 0.0
        
        # 品詞タイプ数
        pos_types = set([token.pos_ for token in doc])
        features['POStypes'] = float(len(pos_types))
        
        # LenNP（文あたりの名詞句長合計の平均）
        if sentences:
            per_sent_np_len = []
            for sent in sentences:
                nps = list(sent.noun_chunks)
                total_len = sum(len(np_.text.split()) for np_ in nps)
                per_sent_np_len.append(total_len)
            features['LenNP'] = float(np.mean(per_sent_np_len)) if per_sent_np_len else 0.0
        else:
            features['LenNP'] = 0.0
        
        return features

    # --- 辞書・レベル補助 ---
    def _lookup_level(self, lemma_lower: str) -> Optional[str]:
        if not self.word_level_map:
            return None
        return self.word_level_map.get(lemma_lower)
    
    @staticmethod
    def _level_to_num(lbl: str) -> int:
        s = lbl.upper().strip().replace(' ', '')
        m = re.match(r'([ABC][12])', s)
        key = m.group(1) if m else s
        table = {'A1':1, 'A2':2, 'B1':3, 'B2':4, 'C1':5, 'C2':6}
        return table.get(key, 0)

# ========================================
# STEP 5: ハイブリッドジェネレーター
# ========================================

class HybridTextGenerator:
    """MLモデルとLLMを組み合わせた生成システム"""
    
    def __init__(self, 
                 excel_path: str,
                 api_key: Optional[str] = None,
                 cefr_wordlist_path: Optional[str] = None):
        """
        Parameters:
        -----------
        excel_path: 読みやすさデータのExcelファイルパス
        api_key: OpenAI APIキー（環境変数で設定済みの場合は不要）
        """
        
        print("🚀 ハイブリッド生成システムを初期化中...")
        
        # まず特徴抽出器（CEFR語彙表込み）を初期化
        self.extractor = FeatureExtractor(cefr_wordlist_path)

        # 学習データの自動拡充（Reading_Text -> 特徴量 -> 既存Excelにマージ）
        merged_excel_path = self._maybe_build_and_merge_training_data(
            base_excel_path=excel_path,
            text_dir=DEFAULT_READING_TEXT_DIR
        )

        # 予測器の初期化（拡充済みがあればそちらを使用）
        self.predictor = ReadabilityPredictor(merged_excel_path or excel_path)
        
        # APIキーを環境変数に設定（指定された場合）
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        self.llm = LLMProvider()
        
        # MLモデルの訓練
        self.predictor.train_model()
        
        print("✅ システム初期化完了！")

    def _maybe_build_and_merge_training_data(self, base_excel_path: str, text_dir: str) -> Optional[str]:
        """Reading_Text ディレクトリから特徴量を作成し、既存Excelにマージして保存。
        戻り値: 生成されたExcelパス（失敗時はNone）
        """
        try:
            if not os.path.isdir(text_dir):
                return None
            if not (base_excel_path and os.path.exists(base_excel_path)):
                print("ℹ️ 既存の学習Excelが見つからないため、マージ処理をスキップします。")
                return None
            print("🧪 学習データを拡充中: Reading_Text -> 特徴量 -> 既存Excelにマージ")
            import glob
            txt_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
            if not txt_files:
                print("ℹ️ Reading_Text に .txt が見つからないためスキップします。")
                return None
            # 既存Excelの読み込み
            base_df = pd.read_excel(base_excel_path)
            # ラベルマッピングの推定
            label_map = self._build_label_map_from_excel(base_df)
            if not label_map:
                print("ℹ️ 既存Excelから 'Text_#' へのラベル対応を特定できませんでした。マージをスキップします。")
                return None
            # テキスト→特徴量
            rows = []
            try:
                from tqdm import tqdm
                iterator = tqdm(txt_files, desc="Featureizing", ncols=80)
            except Exception:
                iterator = txt_files
            for path in iterator:
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception:
                    continue
                feats = self.extractor.extract_features(text)
                fname = os.path.basename(path)
                text_id = self._normalize_text_id(fname)
                label = label_map.get(text_id)
                if not label:
                    continue  # ラベルなしは学習に使わない
                feats['predict_level'] = label
                feats['filename'] = fname
                rows.append(feats)
            if not rows:
                print("ℹ️ ラベル付テキストから特徴量を生成できませんでした。")
                return None
            new_df = pd.DataFrame(rows)
            # マージ（重複ファイルは既存優先）
            if 'filename' in base_df.columns and 'filename' in new_df.columns:
                merged = pd.concat([base_df, new_df], ignore_index=True)
                merged = merged.sort_values(by=['filename']).drop_duplicates(subset=['filename'], keep='first')
            else:
                # filename列が無い場合は単純結合（列整形）
                merged = pd.concat([base_df, new_df], ignore_index=True, sort=False)
            # 必要列の並び
            cols = ['filename'] + list(FEATURE_NAMES) + ['predict_level']
            for c in cols:
                if c not in merged.columns:
                    merged[c] = None
            merged = merged[cols]
            out_dir = os.path.join('outputs')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'training_dataset_merged.xlsx')
            merged.to_excel(out_path, index=False)
            print(f"✅ 学習データをマージして保存しました: {out_path} (行数: {len(merged)})")
            return out_path
        except Exception as e:
            print(f"⚠️ 学習データ拡充処理でエラー: {e}")
            return None

    @staticmethod
    def _normalize_text_id(name: str) -> str:
        s = os.path.splitext(name)[0]
        s = s.strip().lower()
        m = re.search(r'text[_\s-]?(\d+)', s)
        return f"text_{m.group(1)}" if m else s

    @staticmethod
    def _build_label_map_from_excel(df: pd.DataFrame) -> Dict[str, str]:
        # 候補の列名からファイル名/IDを探す
        lower = {c.lower(): c for c in df.columns}
        cand_keys = ['filename', 'file', 'text', 'text_name', 'name', 'id', 'text_id']
        name_col = None
        for k in cand_keys:
            if k in lower:
                name_col = lower[k]
                break
        if name_col is None:
            # 値をスキャンして 'Text_#' を含む列を推定
            for c in df.columns:
                if df[c].astype(str).str.contains(r'Text[_\s-]?\d+', case=False, regex=True).any():
                    name_col = c
                    break
        if name_col is None or 'predict_level' not in df.columns:
            return {}
        mapping = {}
        for _, row in df[[name_col, 'predict_level']].dropna().iterrows():
            nid = HybridTextGenerator._normalize_text_id(str(row[name_col]))
            mapping[nid] = row['predict_level']
        return mapping
    
    def generate_with_validation(self,
                                target_level: str,
                                topic: str,
                                word_count: int = 200,
                                max_attempts: int = 5,
                                word_tolerance: float = 0.1) -> Dict:
        """
        レベル検証付きでテキストを生成
        
        Returns:
        --------
        {
            'text': 生成されたテキスト,
            'actual_level': 予測されたレベル,
            'confidence': 信頼度,
            'attempts': 試行回数,
            'features': 特徴量,
            'success': 目標レベルに到達したか
        }
        """
        
        print(f"\n🎯 目標: {target_level}レベルの{word_count}語のテキスト（トピック: {topic}）")
        
        best_result = None
        best_distance = float('inf')
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n試行 {attempt}/{max_attempts}:")
            
            # プロンプトを生成
            prompt = self._create_prompt(target_level, topic, word_count, attempt)
            
            # LLMで生成
            temperature = 0.65 + 0.1 * random.random()
            generated_text = self.llm.generate(prompt, max_tokens=word_count * 2, temperature=temperature)
            
            # 特徴量を抽出
            features = self.extractor.extract_features(generated_text)
            
            # レベルを予測（モデルが無い場合はOpenAIで判定）
            if self.predictor.model is not None:
                predicted_level, confidence = self.predictor.predict_level(features)
            else:
                predicted_level, confidence = self.llm.classify_level(generated_text)
            
            print(f"  生成テキスト: {generated_text[:100]}...")
            print(f"  予測レベル: {predicted_level} (信頼度: {confidence:.2%})")
            
            # 結果を記録
            # 語数チェック
            wc = len(generated_text.split())
            lower = int(round(word_count * (1 - word_tolerance)))
            upper = int(round(word_count * (1 + word_tolerance)))
            within_wc = lower <= wc <= upper

            result = {
                'text': generated_text,
                'actual_level': predicted_level,
                'target_level': target_level,
                'topic': topic,
                'confidence': confidence,
                'attempts': attempt,
                'features': features,
                'success': (predicted_level == target_level) and within_wc,
                'word_count': wc,
                'word_range': {'lower': lower, 'upper': upper}
            }
            
            # 目標レベルかつ語数が範囲内なら終了
            if result['success']:
                print(f"  ✅ 目標レベルに到達！")
                if not within_wc:
                    print(f"  ⚠️ 語数が範囲外: {wc} (許容 {lower}-{upper})")
                return result
            
            # 最も近い結果を保存
            distance = self._calculate_level_distance(predicted_level, target_level)
            if distance < best_distance:
                best_distance = distance
                best_result = result
            
            # フィードバックを生成して次の試行に活かす
            if attempt < max_attempts:
                reason = []
                if predicted_level != target_level:
                    reason.append("level")
                if not within_wc:
                    reason.append("word_count")
                print(f"  ↻ 調整中... ({', '.join(reason)})")
        
        print(f"\n⚠️ {max_attempts}回の試行後、最も近い結果を返します")
        return best_result
    
    def _create_prompt(self, target_level: str, topic: str, word_count: int, attempt: int) -> str:
        """プロンプトを生成"""
        
        # レベル別の詳細な指示
        level_guidelines = {
            'A1.1': {
                'vocabulary': 'only the 500 most common English words',
                'grammar': 'present simple tense only, no complex structures',
                'sentence_length': '3-8 words per sentence',
                'example': 'I am happy. My family is big. We eat dinner together.'
            },
            'A1.2': {
                'vocabulary': 'the 750 most common English words',
                'grammar': 'present simple and continuous, basic past tense',
                'sentence_length': '5-10 words per sentence',
                'example': 'Yesterday I went to school. I am studying English now.'
            },
            'A2.2': {
                'vocabulary': 'the 1500 most common English words',
                'grammar': 'all basic tenses, simple connectors (and, but, because)',
                'sentence_length': '8-12 words per sentence',
                'example': 'I have been to Paris before because my sister lives there.'
            },
            'B1.1': {
                'vocabulary': 'common vocabulary with some less frequent words',
                'grammar': 'complex sentences, conditionals, passive voice',
                'sentence_length': '10-15 words per sentence',
                'example': 'If I had known about the meeting, I would have prepared a presentation.'
            },
            'B1.2': {
                'vocabulary': 'wider range including some abstract concepts',
                'grammar': 'all tenses, relative clauses, reported speech',
                'sentence_length': '12-18 words per sentence',
                'example': 'The manager said that the project, which had been delayed, would be completed soon.'
            },
            'B2.1': {
                'vocabulary': 'broad vocabulary including idiomatic expressions',
                'grammar': 'sophisticated structures, subjunctive mood',
                'sentence_length': '15-20 words per sentence',
                'example': 'Having considered all the options, we decided that it was essential that everyone be informed immediately.'
            },
            'B2.2': {
                'vocabulary': 'extensive vocabulary, technical terms',
                'grammar': 'all complex structures, nuanced expression',
                'sentence_length': '18-25 words per sentence',
                'example': 'The implications of this decision, while not immediately apparent, will undoubtedly have far-reaching consequences for the entire organization.'
            },
            'C1': {
                'vocabulary': 'sophisticated vocabulary, subtle distinctions',
                'grammar': 'native-like complexity and flexibility',
                'sentence_length': '20-30 words per sentence',
                'example': 'The notion that technological progress invariably leads to improved quality of life is, at best, a gross oversimplification of the complex relationship between innovation and human welfare.'
            },
            'C2': {
                'vocabulary': 'complete mastery including rare and literary forms',
                'grammar': 'full native speaker competence',
                'sentence_length': 'varied, up to 35 words',
                'example': 'One might venture to suggest that the zeitgeist of our epoch, characterized as it is by an almost pathological obsession with immediacy, militates against the cultivation of those contemplative virtues upon which genuine wisdom has traditionally been thought to depend.'
            }
        }
        
        guidelines = level_guidelines.get(target_level, level_guidelines['B1.1'])
        
        # 試行回数に応じて指示を強化
        emphasis = "STRICTLY" if attempt > 2 else ""
        lower = int(round(word_count * 0.9))
        upper = int(round(word_count * 1.1))
        
        prompt = f"""
Generate a {word_count}-word English text about "{topic}" at CEFR level {target_level}.

{emphasis} Requirements for {target_level}:
- Vocabulary: Use {guidelines['vocabulary']}
- Grammar: {guidelines['grammar']}
- Sentence length: {guidelines['sentence_length']}

Style example for {target_level}:
"{guidelines['example']}"

Important instructions:
1. Match the complexity level EXACTLY
2. Make the text natural and engaging
3. Stay strictly within the {target_level} constraints
4. Write between {lower} and {upper} words (not fewer or more)
5. Output only the text (no headings, no explanations)

Generate the text now:
"""
        
        return prompt

    def generate_n_texts(self,
                         target_level: str,
                         topic: str,
                         word_count: int,
                         n_texts: int,
                         max_attempts_each: int = 3,
                         word_tolerance: float = 0.1,
                         save_txt: bool = True,
                         save_dir: str = 'outputs/texts') -> List[Dict]:
        """指定条件でN本のテキストを生成し、条件を満たすものだけを収集"""
        outputs = []
        attempts_total = 0
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        topic_slug = _slug(topic)
        level_slug = _slug(target_level)
        if save_txt:
            os.makedirs(save_dir, exist_ok=True)
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_texts, desc="Collecting", ncols=80)
        except Exception:
            pbar = None
        while len(outputs) < n_texts and attempts_total < n_texts * max_attempts_each * 2:
            res = self.generate_with_validation(
                target_level=target_level,
                topic=topic,
                word_count=word_count,
                max_attempts=max_attempts_each,
                word_tolerance=word_tolerance
            )
            attempts_total += res.get('attempts', 1)
            if res.get('success'):
                outputs.append(res)
                if save_txt:
                    idx = len(outputs)
                    fname = f"{level_slug}_{topic_slug}_{word_count}w_{ts}_{idx:02d}.txt"
                    path = os.path.join(save_dir, fname)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(res['text'])
                if pbar:
                    pbar.update(1)
            else:
                # 収集対象外だが、近い結果としてログを残す
                pass
        if pbar:
            pbar.close()
        return outputs
    
    def _calculate_level_distance(self, level1: str, level2: str) -> float:
        """レベル間の距離を計算"""
        level_order = ['A1.1', 'A1.2', 'A2.2', 'B1.1', 'B1.2', 'B2.1', 'B2.2', 'C1', 'C2']
        
        try:
            idx1 = level_order.index(level1)
            idx2 = level_order.index(level2)
            return abs(idx1 - idx2)
        except ValueError:
            return float('inf')
    
    def batch_generate(self, 
                       levels: List[str],
                       topics: List[str],
                       word_count: int = 200,
                       save_txt: bool = True,
                       save_dir: str = 'outputs/texts') -> List[Dict]:
        """複数のレベル・トピックで一括生成。必要に応じて.txt保存。"""
        
        results = []
        total = len(levels) * len(topics)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if save_txt:
            os.makedirs(save_dir, exist_ok=True)
        print(f"\n📚 {total}個のテキストを生成します...")
        
        try:
            from tqdm import tqdm
            iterator = tqdm(itertools.product(levels, topics), total=total, desc="Generating", ncols=80)
        except Exception:
            iterator = itertools.product(levels, topics)
        
        count = 0
        for level, topic in iterator:
            count += 1
            result = self.generate_with_validation(
                target_level=level,
                topic=topic,
                word_count=word_count,
                max_attempts=3
            )
            results.append(result)
            if save_txt and result.get('success'):
                fname = f"{_slug(level)}_{_slug(topic)}_{word_count}w_{ts}_{count:02d}.txt"
                path = os.path.join(save_dir, fname)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
        
        # 結果をDataFrameにまとめる
        df_results = pd.DataFrame(results)
        success_rate = df_results['success'].mean()
        
        print(f"\n📊 生成完了！")
        print(f"  成功率: {success_rate:.1%}")
        print(f"  平均信頼度: {df_results['confidence'].mean():.1%}")
        print(f"  平均試行回数: {df_results['attempts'].mean():.1f}")
        
        return results

# ========================================
# STEP 6: 実行例とテスト
# ========================================

def main():
    """メイン実行関数"""
    
    print("=" * 70)
    print("🎓 ハイブリッド型英文生成システム - 実行デモ")
    print("=" * 70)
    
    # ExcelファイルとCEFR-J Wordlistのパス（絶対パスをコードに組み込み）
    excel_path = DEFAULT_EXCEL_PATH
    
    # CEFR-J Wordlistのパス
    default_wordlist = os.getenv('CEFR_WORDLIST_PATH', '').strip()
    cefr_wordlist_path = DEFAULT_WORDLIST_PATH
    if not os.path.exists(excel_path):
        print(f"⚠️ 学習Excelが見つかりません: {excel_path}")
    if not os.path.exists(cefr_wordlist_path):
        print(f"⚠️ CEFR-J Wordlistが見つかりません: {cefr_wordlist_path}")

    # OpenAI APIキーの入力
    print("\n🔑 OpenAI APIキーを入力してください")
    print("（環境変数に設定済みの場合はEnterキーを押してスキップ）")
    api_key = input("APIキー: ").strip() or None
    
    # システムの初期化
    print("\n🔧 システムを初期化中...")
    generator = HybridTextGenerator(
        excel_path=excel_path,
        api_key=api_key,
        cefr_wordlist_path=cefr_wordlist_path
    )
    
    # テスト生成
    print("\n" + "=" * 70)
    print("📝 テスト生成を実行")
    print("=" * 70)
    print("対応CEFRレベル: A1.1, A1.2, A2.2, B1.1, B1.2, B2.1, B2.2, C1, C2")
    print("例トピック: Environmental Protection, Technology, Education, Health, Culture, Travel, Science, Sports, Food, Society")
    
    # 単一生成の例
    result = generator.generate_with_validation(
        target_level='B1.1',
        topic='Environmental Protection',
        word_count=150,
        max_attempts=3
    )
    
    print("\n" + "=" * 50)
    print("📄 生成結果:")
    print("=" * 50)
    print(f"目標レベル: {result['target_level']}")
    print(f"実際のレベル: {result['actual_level']}")
    print(f"信頼度: {result['confidence']:.1%}")
    print(f"単語数: {result['word_count']}")
    print(f"成功: {'✅' if result['success'] else '❌'}")
    print(f"\n生成テキスト:\n{result['text']}")
    # 単体出力をテキストとして保存
    ts1 = datetime.now().strftime('%Y%m%d_%H%M%S')
    first_txt_dir = os.path.join('outputs', 'texts')
    os.makedirs(first_txt_dir, exist_ok=True)
    first_fname = f"{_slug(result['target_level'])}_{_slug(result['topic'])}_{result['word_count']}w_{ts1}.txt"
    with open(os.path.join(first_txt_dir, first_fname), 'w', encoding='utf-8') as f:
        f.write(result['text'])
    print(f"📝 テキストを outputs/texts/{first_fname} に保存しました")
    
    # 任意: 指定本数のカスタム生成
    do_custom = input("\n指定本数のカスタム生成を実行しますか？ (y/N): ").strip().lower() == 'y'
    if do_custom:
        level_in = input("CEFRレベルを入力（例: A1.1/B1.1/C1）: ").strip() or 'B1.1'
        topic_in = input("トピック（例: Technology）: ").strip() or 'General Topic'
        try:
            wc_in = int(input("語数（例: 120）: ").strip() or '120')
        except ValueError:
            wc_in = 120
        try:
            n_in = int(input("生成本数（例: 5）: ").strip() or '5')
        except ValueError:
            n_in = 5
        print("\n🚀 指定条件で生成を開始します...")
        items = generator.generate_n_texts(
            target_level=level_in,
            topic=topic_in,
            word_count=wc_in,
            n_texts=n_in,
            max_attempts_each=3,
            word_tolerance=0.1,
            save_txt=True
        )
        df_custom = pd.DataFrame(items)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_custom = os.path.join('outputs', f'custom_texts_{ts}.csv')
        os.makedirs(os.path.dirname(out_custom), exist_ok=True)
        df_custom.to_csv(out_custom, index=False)
        print(f"✅ カスタム生成の結果を {out_custom} に保存しました")

    # バッチ生成の例
    print("\n" + "=" * 70)
    print("📚 バッチ生成のデモ")
    print("=" * 70)
    
    batch_results = generator.batch_generate(
        levels=['A1.1','A1.2','A2.2','B1.1','B1.2','B2.1','B2.2','C1','C2'],
        topics=['My Daily Routine','Technology','Education','Health','Travel'],
        word_count=100,
        save_txt=True,
        save_dir=os.path.join('outputs', 'texts')
    )
    
    # 結果の保存
    print("\n💾 結果を保存中...")
    
    # CSVファイルとして保存
    df_results = pd.DataFrame(batch_results)
    output_path = os.path.join('outputs', 'generation_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"✅ 結果を {output_path} に保存しました")
    
    # モデルの保存
    model_path = os.path.join('outputs', 'readability_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    generator.predictor.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("✨ 完了！")
    print("=" * 70)
    print("\n次のステップ:")
    print("1. より多くのテキストデータで訓練")
    print("2. 特徴量抽出の精度向上")
    print("3. プロンプトの最適化")
    print("4. Webアプリケーション化")

if __name__ == "__main__":
    # 必要なライブラリのインストール確認
    print("""
    ============================================
    必要なライブラリのインストール
    ============================================
    
    # 基本ライブラリ
    pip install pandas numpy scikit-learn openpyxl
    
    # テキスト分析
    pip install textstat spacy wordfreq
    python -m spacy download en_core_web_sm
    
    # LLM API
    pip install openai        # OpenAI のみ使用
    
    ============================================
    """)
    
    # メイン実行
    main()
