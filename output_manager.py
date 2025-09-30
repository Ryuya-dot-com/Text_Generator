"""
出力管理システム - CSV、個別テキストファイル、ZIP対応
Enhanced Output Manager for Hybrid Text Generation System
"""

import os
import re
import pandas as pd
import json
import zipfile
from datetime import datetime
from typing import Dict, List, Optional
import shutil

class OutputManager:
    """生成結果を様々な形式で保存・管理するクラス"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Parameters:
        -----------
        output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.timestamp}")
        
        # ディレクトリ構造を作成
        self._create_directories()
        
    def _create_directories(self):
        """必要なディレクトリを作成"""
        # メインディレクトリ
        os.makedirs(self.session_dir, exist_ok=True)
        
        # サブディレクトリ
        self.dirs = {
            'texts': os.path.join(self.session_dir, 'texts'),
            'reports': os.path.join(self.session_dir, 'reports')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def save_single_result(self, result: Dict, format_options: Dict = None) -> Dict[str, str]:
        """
        単一の生成結果を保存
        
        Parameters:
        -----------
        result: 生成結果の辞書
        format_options: 保存形式のオプション
        
        Returns:
        --------
        保存されたファイルパスの辞書
        """
        if format_options is None:
            format_options = {
                'save_txt': True,
                'save_json': True,
                'include_metadata': True
            }
        
        saved_files = {}
        
        # ファイル名の生成
        band = str(result.get('band', result.get('target_level', 'unknown'))).replace('.', '_')
        text_type = result.get('text_type', 'text')
        topic_raw = result.get('topic', 'untitled').lower()
        topic_slug = re.sub(r'[^a-z0-9]+', '_', topic_raw).strip('_')[:40] or 'text'
        set_index = result.get('set_index', 1)
        filename_base = f"set{set_index}_{band}_{text_type}_{topic_slug}_{self.timestamp}"

        # 1. テキストファイルとして保存
        if format_options.get('save_txt', True):
            txt_path = os.path.join(self.dirs['texts'], f"{filename_base}.txt")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                # メタデータをヘッダーに含める
                if format_options.get('include_metadata', True):
                    f.write(f"# Band: {band}\n")
                    f.write(f"# CEFR Anchor: {result.get('target_level', 'N/A')}\n")
                    f.write(f"# Type: {text_type}\n")
                    f.write(f"# Topic: {result.get('topic', 'N/A')}\n")
                    f.write(f"# Set Index: {set_index}\n")
                    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Word Count: {result.get('word_count', 0)}\n")
                    f.write("-" * 50 + "\n\n")
                
                # テキスト本文
                f.write(result.get('text', '').strip())
                f.write("\n\n[質問]\n")
                questions = result.get('questions', [])
                if questions:
                    f.write(f"Q1: {questions[0]}\n")
                if len(questions) > 1:
                    f.write(f"Q2: {questions[1]}\n")
            
            saved_files['txt'] = txt_path

        # 2. JSONファイルとして保存（完全なデータ）
        if format_options.get('save_json', True):
            json_path = os.path.join(self.dirs['texts'], f"{filename_base}.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            saved_files['json'] = json_path
        
        return saved_files
    
    def save_batch_results(self, results: List[Dict], output_name: str = None) -> Dict[str, str]:
        """
        バッチ生成結果を保存
        
        Returns:
        --------
        {
            'csv': CSVファイルのパス,
            'excel': Excelファイルのパス,
            'zip': ZIPファイルのパス,
            'summary': サマリーファイルのパス
        }
        """
        if output_name is None:
            output_name = f"batch_{self.timestamp}"
        
        saved_paths = {}

        for result in results:
            self.save_single_result(result)

        # テキスト本体のCSV
        text_records = []
        for result in results:
            questions = result.get('questions', [])
            while len(questions) < 2:
                questions.append('本文について50語程度で回答してください。')
            text_records.append({
                'set': result.get('set_index', 1),
                'level': result.get('band', result.get('level', '')),
                'type': result.get('text_type', ''),
                'text': result.get('text', ''),
                'question1': questions[0],
                'question2': questions[1]
            })
        text_df = pd.DataFrame(text_records)
        text_csv_path = os.path.join(self.dirs['reports'], f"{output_name}_texts.csv")
        text_df.to_csv(text_csv_path, index=False, encoding='utf-8-sig')
        saved_paths['texts_csv'] = text_csv_path
        print(f"✅ テキストCSV保存: {text_csv_path}")

        # メトリクスのCSV
        metric_records = []
        for result in results:
            evaluation = result.get('evaluation', {})
            metrics = evaluation.get('metrics', {})
            delta = evaluation.get('delta_from_target', {})
            lexical = result.get('lexical_profile', {})
            record = {
                'set': result.get('set_index', 1),
                'level': result.get('band', result.get('level', '')),
                'type': result.get('text_type', ''),
                'anchor_level': result.get('target_level', ''),
                'word_count': evaluation.get('word_count', 0),
                'sentence_count': evaluation.get('sentence_count', 0),
                'distance_score': evaluation.get('distance_score', float('nan')),
                'max_rank': lexical.get('max_rank'),
                'out_of_band_count': lexical.get('out_of_band_count', 0),
                'out_of_band_tokens': '; '.join(lexical.get('out_of_band_tokens', []))
            }
            for key, value in metrics.items():
                record[f'metric_{key}'] = value
                record[f'delta_{key}'] = delta.get(key)
            metric_records.append(record)
        metrics_df = pd.DataFrame(metric_records)
        metrics_csv_path = os.path.join(self.dirs['reports'], f"{output_name}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
        saved_paths['metrics_csv'] = metrics_csv_path
        print(f"✅ 指標CSV保存: {metrics_csv_path}")

        # ZIPアーカイブ
        zip_path = os.path.join(self.output_dir, f"{output_name}.zip")
        self._create_zip_archive(zip_path)
        saved_paths['zip'] = zip_path
        print(f"✅ ZIP保存: {zip_path}")

        return saved_paths
    
    def _create_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """結果からDataFrameを作成"""
        data = []
        for result in results:
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'target_level': result.get('target_level', ''),
                'topic': result.get('topic', ''),
                'predicted_level': result.get('predicted_level', ''),
                'confidence': result.get('confidence', 0),
                'success': result.get('success', False),
                'word_count': result.get('word_count', 0),
                'attempts': result.get('attempts', 1),
                'text_preview': result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', ''),
                'full_text': result.get('text', '')
            }
            
            # 特徴量を追加
            features = result.get('features', {})
            for feature_name in ['ARI', 'AvrDiff', 'VperSent', 'LenNP', 'BperA', 'CVV1', 'AvrFreqRank', 'POStypes']:
                row[f'feature_{feature_name}'] = features.get(feature_name, 0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_excel_report(self, results: List[Dict], filepath: str):
        """詳細なExcelレポートを作成"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. メインデータシート
            df_main = self._create_dataframe(results)
            df_main.to_excel(writer, sheet_name='All_Results', index=False)
            
            # 2. レベル別シート
            for level in df_main['target_level'].unique():
                df_level = df_main[df_main['target_level'] == level]
                sheet_name = f"Level_{level.replace('.', '_')}"[:31]  # Excel制限
                df_level.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 3. 統計サマリーシート
            summary_data = {
                'Metric': ['Total Texts', 'Success Rate', 'Average Confidence', 
                          'Average Word Count', 'Average Attempts'],
                'Value': [
                    len(results),
                    f"{df_main['success'].mean():.1%}",
                    f"{df_main['confidence'].mean():.1%}",
                    f"{df_main['word_count'].mean():.0f}",
                    f"{df_main['attempts'].mean():.1f}"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # 4. 成功率分析シート
            success_by_level = df_main.groupby('target_level').agg({
                'success': 'mean',
                'confidence': 'mean',
                'attempts': 'mean',
                'word_count': 'mean'
            }).round(3)
            success_by_level.to_excel(writer, sheet_name='Success_Analysis')
    
    def _create_summary_report(self, results: List[Dict], filepath: str):
        """テキストサマリーレポートを作成"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("英文生成システム - バッチ処理サマリー\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総生成数: {len(results)}\n\n")
            
            # 成功率の計算
            success_count = sum(1 for r in results if r.get('success', False))
            f.write(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)\n\n")
            
            # レベル別統計
            f.write("レベル別結果:\n")
            f.write("-" * 40 + "\n")
            
            level_stats = {}
            for result in results:
                level = result.get('target_level', 'unknown')
                if level not in level_stats:
                    level_stats[level] = {'total': 0, 'success': 0, 'confidence_sum': 0}
                
                level_stats[level]['total'] += 1
                if result.get('success', False):
                    level_stats[level]['success'] += 1
                level_stats[level]['confidence_sum'] += result.get('confidence', 0)
            
            for level in sorted(level_stats.keys()):
                stats = level_stats[level]
                success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                avg_confidence = stats['confidence_sum'] / stats['total'] if stats['total'] > 0 else 0
                
                f.write(f"{level:8} | 生成: {stats['total']:3} | 成功: {stats['success']:3} | ")
                f.write(f"成功率: {success_rate:5.1f}% | 平均信頼度: {avg_confidence:.1%}\n")
            
            # トピック別統計
            f.write("\n\nトピック別結果:\n")
            f.write("-" * 40 + "\n")
            
            topic_stats = {}
            for result in results:
                topic = result.get('topic', 'unknown')[:30]
                if topic not in topic_stats:
                    topic_stats[topic] = {'count': 0, 'success': 0}
                topic_stats[topic]['count'] += 1
                if result.get('success', False):
                    topic_stats[topic]['success'] += 1
            
            for topic, stats in topic_stats.items():
                success_rate = stats['success'] / stats['count'] * 100 if stats['count'] > 0 else 0
                f.write(f"{topic:30} | 生成: {stats['count']:3} | 成功率: {success_rate:5.1f}%\n")
    
    def _create_zip_archive(self, zip_path: str):
        """全ファイルをZIPアーカイブに圧縮"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # セッションディレクトリ内のすべてのファイルを追加
            for root, dirs, files in os.walk(self.session_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.output_dir)
                    zipf.write(file_path, arcname)
    
    def create_custom_package(self, results: List[Dict], package_options: Dict) -> str:
        """
        カスタムパッケージを作成
        
        Parameters:
        -----------
        package_options: {
            'format': 'zip' or 'folder',
            'include_csv': True,
            'include_excel': True,
            'include_texts': True,
            'organize_by': 'level' or 'topic' or 'both',
            'include_failed': False
        }
        """
        # フィルタリング（失敗を除外する場合）
        if not package_options.get('include_failed', True):
            results = [r for r in results if r.get('success', False)]
        
        # 保存処理
        saved_paths = self.save_batch_results(results)
        
        print("\n" + "=" * 60)
        print("📦 出力パッケージ作成完了！")
        print("=" * 60)
        print(f"📁 出力ディレクトリ: {self.session_dir}")
        print(f"📊 CSVファイル: {saved_paths['csv']}")
        print(f"📑 Excelファイル: {saved_paths['excel']}")
        print(f"📄 サマリー: {saved_paths['summary']}")
        print(f"🗜️ ZIPファイル: {saved_paths['zip']}")
        print("\n含まれるファイル:")
        print(f"  - 個別テキストファイル: {len(results)}個")
        print(f"  - レベル別フォルダ: 9個")
        print(f"  - レポート: 3個")
        
        return saved_paths['zip']

# ========================================
# 統合された生成・出力システム
# ========================================

class EnhancedHybridGenerator:
    """出力管理機能を統合したハイブリッドジェネレーター"""
    
    def __init__(self, excel_path: str, llm_provider: str = 'openai', api_key: str = None):
        # 既存のハイブリッドジェネレーターを初期化
        from hybrid_approach_setup import HybridTextGenerator
        self.generator = HybridTextGenerator(excel_path, llm_provider, api_key)
        
        # 出力マネージャーを初期化
        self.output_manager = OutputManager()
    
    def generate_and_save(self, 
                         target_level: str,
                         topic: str,
                         word_count: int = 200,
                         save_options: Dict = None) -> Dict:
        """
        テキストを生成して保存
        
        Returns:
        --------
        {
            'result': 生成結果,
            'saved_files': 保存されたファイルパス
        }
        """
        # テキスト生成
        result = self.generator.generate_with_validation(
            target_level=target_level,
            topic=topic,
            word_count=word_count
        )
        
        # トピック情報を追加
        result['topic'] = topic
        
        # ファイルに保存
        saved_files = self.output_manager.save_single_result(result, save_options)
        
        return {
            'result': result,
            'saved_files': saved_files
        }
    
    def batch_generate_and_save(self,
                               levels: List[str],
                               topics: List[str],
                               word_count: int = 200,
                               output_format: str = 'all') -> str:
        """
        バッチ生成して保存
        
        Parameters:
        -----------
        output_format: 'csv', 'txt', 'zip', 'all'
        
        Returns:
        --------
        ZIPファイルのパス
        """
        print(f"\n📚 バッチ生成開始: {len(levels)}レベル × {len(topics)}トピック = {len(levels)*len(topics)}テキスト")
        
        results = []
        total = len(levels) * len(topics)
        
        for i, level in enumerate(levels):
            for j, topic in enumerate(topics):
                current = i * len(topics) + j + 1
                print(f"\n[{current}/{total}] 生成中: {level} - {topic}")
                
                result = self.generator.generate_with_validation(
                    target_level=level,
                    topic=topic,
                    word_count=word_count,
                    max_attempts=3
                )
                result['topic'] = topic  # トピック情報を追加
                results.append(result)
        
        # 結果を保存
        if output_format == 'csv':
            csv_path = os.path.join(self.output_manager.dirs['reports'], 'results.csv')
            df = self.output_manager._create_dataframe(results)
            df.to_csv(csv_path, index=False)
            return csv_path
        
        elif output_format == 'txt':
            for result in results:
                self.output_manager.save_single_result(result)
            return self.output_manager.session_dir
        
        else:  # 'zip' or 'all'
            saved_paths = self.output_manager.save_batch_results(results)
            return saved_paths['zip']

# ========================================
# 実行例
# ========================================

def main():
    """使用例"""
    
    print("=" * 70)
    print("📦 拡張出力システム - デモ")
    print("=" * 70)
    
    # システム初期化
    generator = EnhancedHybridGenerator(
        excel_path='CVLA3_20250912133649_3373.xlsx',
        llm_provider='demo',  # デモモード
        api_key=None
    )
    
    # 1. 単一生成と保存
    print("\n1️⃣ 単一テキスト生成")
    result = generator.generate_and_save(
        target_level='B1.1',
        topic='Climate Change',
        word_count=150
    )
    print(f"✅ 保存完了: {result['saved_files']}")
    
    # 2. バッチ生成と保存
    print("\n2️⃣ バッチ生成（複数形式で出力）")
    zip_path = generator.batch_generate_and_save(
        levels=['A1.1', 'B1.1', 'C1'],
        topics=['Technology', 'Environment'],
        word_count=100,
        output_format='all'  # CSV + TXT + ZIP
    )
    print(f"✅ ZIPファイル作成: {zip_path}")
    
    # 3. カスタムパッケージ
    print("\n3️⃣ カスタムパッケージ作成")
    
    # テスト用の結果を生成
    test_results = [
        {
            'text': f"Sample text for level {level}",
            'target_level': level,
            'predicted_level': level,
            'confidence': 0.85,
            'success': True,
            'word_count': 100,
            'topic': 'Test Topic',
            'features': {}
        }
        for level in ['A1.1', 'B1.1', 'C1']
    ]
    
    # カスタムオプションで保存
    package_path = generator.output_manager.create_custom_package(
        results=test_results,
        package_options={
            'format': 'zip',
            'include_csv': True,
            'include_excel': True,
            'include_texts': True,
            'organize_by': 'both',
            'include_failed': False
        }
    )
    
    print(f"\n✅ カスタムパッケージ: {package_path}")
    
    print("\n" + "=" * 70)
    print("📊 出力形式の説明")
    print("=" * 70)
    print("""
    生成されたファイル構造:
    
    output/
    └── session_YYYYMMDD_HHMMSS/
        ├── texts/                  # 全テキストファイル
        │   ├── A1_1_topic_success.txt
        │   ├── A1_1_topic_success.json
        │   └── ...
        ├── by_level/               # レベル別整理
        │   ├── A1_1/
        │   ├── B1_1/
        │   └── C1/
        ├── reports/                # レポート
        │   ├── batch_TIMESTAMP.csv
        │   ├── batch_TIMESTAMP.xlsx
        │   └── batch_TIMESTAMP_summary.txt
        └── batch_TIMESTAMP.zip     # 全ファイルの圧縮版
    
    CSV形式: 分析・フィルタリングに最適
    TXT形式: 個別の教材として利用可能
    Excel形式: 詳細な分析・レポート作成用
    ZIP形式: 配布・アーカイブ用
    """)

if __name__ == "__main__":
    main()
