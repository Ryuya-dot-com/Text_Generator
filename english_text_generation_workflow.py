"""
英文読みやすさレベル指定テキスト生成システム
Multiple approaches for generating English texts with specified readability levels
"""

import os
import json
import ast
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import openai
from dataclasses import dataclass, asdict
import math

import spacy
from wordfreq import zipf_frequency
import textstat
from dotenv import load_dotenv
from output_manager import OutputManager


LEVEL_SCORES = {
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C1': 5,
    'C2': 6
}

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
FUNCTION_POS = {
    'DET', 'PRON', 'ADP', 'AUX', 'CCONJ', 'SCONJ', 'PART', 'INTJ',
    'PUNCT', 'SPACE', 'SYM'
}

# Default ceiling for frequency rank when no estimate is available
FALLBACK_FREQUENCY_RANK = 100_000

LEVEL_GROUPS = [
    ('A1', ['A1.1', 'A1.2'], 'A1 (A1.1–A1.2)'),
    ('A2', ['A2.2'], 'A2 (A2.2)'),
    ('B1.1', ['B1.1'], 'B1.1'),
    ('B1.2', ['B1.2'], 'B1.2'),
    ('B2.1', ['B2.1'], 'B2.1'),
    ('B2.2', ['B2.2'], 'B2.2'),
    ('C1+', ['C1', 'C2'], 'C1+ (C1–C2)')
]

VALID_TEXT_TYPES = {'narrative', 'expository'}

METRIC_KEYS = ['AvrDiff', 'BperA', 'CVV1', 'AvrFreqRank', 'ARI', 'VperSent', 'POStypes', 'LenNP']

DIFFICULTY_BANDS = {
    '2K': {
        'anchor_level': 'A1',
        'max_rank': 2000,
        'word_count': 200,
        'topics': {
            'narrative': 'A new friend at the community park',
            'expository': 'Morning routines for healthy students'
        }
    },
    '3K': {
        'anchor_level': 'A1',
        'max_rank': 3000,
        'word_count': 200,
        'topics': {
            'narrative': 'Helping at the school festival',
            'expository': 'How school clubs welcome new members'
        }
    },
    '4K': {
        'anchor_level': 'A2',
        'max_rank': 4000,
        'word_count': 200,
        'topics': {
            'narrative': 'A family trip to the science museum',
            'expository': 'Why communities start recycling projects'
        }
    },
    '5K': {
        'anchor_level': 'B1.2',
        'max_rank': 5000,
        'word_count': 200,
        'topics': {
            'narrative': 'A teen journalist covering a local mystery',
            'expository': 'Strategies for students to manage stress'
        }
    },
    '6K': {
        'anchor_level': 'B2.2',
        'max_rank': 6000,
        'word_count': 200,
        'topics': {
            'narrative': 'Preparing for an international debate competition',
            'expository': 'How coastal towns adapt to climate change'
        }
    },
    '7K': {
        'anchor_level': 'C1+',
        'max_rank': 7000,
        'word_count': 200,
        'topics': {
            'narrative': 'A researcher confronting ethical choices in AI labs',
            'expository': 'Long-term impacts of global data governance policies'
        }
    }
}

# ========================================
# 1. データ構造とレベル定義
# ========================================

@dataclass
class ReadabilityMetrics:
    """読みやすさ指標のデータクラス"""
    AvrDiff: float        # 平均難易度
    BperA: float          # 文章の複雑さ
    CVV1: float           # 語彙の多様性
    AvrFreqRank: float    # 平均頻度ランク
    ARI: float            # Automated Readability Index
    VperSent: float       # 文あたりの動詞数
    POStypes: float       # 品詞タイプ数
    LenNP: float          # 名詞句の長さ
    
@dataclass
class CEFRLevel:
    """CEFRレベルとその特徴"""
    level: str
    metrics_range: Dict[str, Tuple[float, float]]
    description: str
    example_features: List[str]

# ========================================
# 2. 読みやすさ分析器
# ========================================

class ReadabilityAnalyzer:
    """テキストの読みやすさを分析するクラス"""

    def __init__(self,
                 training_data_path: str,
                 cefr_wordlist_path: Optional[str] = None):
        """読みやすさ分析に必要なリソースを初期化"""
        self.df = pd.read_excel(training_data_path)
        self.level_profiles = self._create_level_profiles()

        # spaCyモデルをロード（固有表現認識は不要なので無効化）
        self.nlp = spacy.load('en_core_web_sm', disable=['ner'])

        # CEFR-J語彙リストからレベル辞書を作成
        self.cefr_by_pos, self.cefr_fallback = self._load_cefr_wordlist(cefr_wordlist_path) if cefr_wordlist_path else ({}, {})

    def _create_level_profiles(self) -> Dict[str, Dict]:
        """各レベルの統計的プロファイルを作成"""
        profiles = {}
        self.level_display_names: Dict[str, str] = {}
        self.level_aliases: Dict[str, str] = {}

        if 'predict_level' not in self.df.columns:
            return profiles

        metrics = ['AvrDiff', 'BperA', 'CVV1', 'AvrFreqRank',
                   'ARI', 'VperSent', 'POStypes', 'LenNP']

        for group_name, members, display_name in LEVEL_GROUPS:
            level_data = self.df[self.df['predict_level'].isin(members)]
            if level_data.empty:
                continue

            available_metrics = [m for m in metrics if m in level_data.columns]
            if not available_metrics:
                continue

            profiles[group_name] = {
                'mean': level_data[available_metrics].mean().to_dict(),
                'std': level_data[available_metrics].std().to_dict(),
                'min': level_data[available_metrics].min().to_dict(),
                'max': level_data[available_metrics].max().to_dict(),
                'sample_count': len(level_data)
            }

            self.level_display_names[group_name] = display_name
            self.level_aliases[group_name] = group_name
            for member in members:
                self.level_aliases[member] = group_name
        return profiles

    def resolve_level(self, requested_level: str) -> str:
        """利用者の指定を標準化されたレベル名に解決"""
        if not requested_level:
            raise ValueError("Level must be specified")

        if requested_level in self.level_profiles:
            return requested_level

        resolved = self.level_aliases.get(requested_level)
        if not resolved or resolved not in self.level_profiles:
            available = ', '.join(self.level_profiles.keys())
            raise ValueError(f"Level {requested_level} not recognized. Available levels: {available}")
        return resolved

    def get_display_name(self, level: str) -> str:
        return self.level_display_names.get(level, level)

    def _load_cefr_wordlist(self, path: str) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
        """CEFR-J語彙リストから(語, 品詞)→レベルの辞書を構築"""
        if not path:
            return {}, {}

        df = pd.read_excel(path, sheet_name='ALL')
        lookup: Dict[Tuple[str, str], str] = {}
        fallback: Dict[str, str] = {}

        for _, row in df.iterrows():
            headword = str(row.get('headword', '')).strip().lower()
            pos = str(row.get('pos', '')).strip().lower()
            level = str(row.get('CEFR', '')).strip().upper()

            if not headword or level not in LEVEL_SCORES:
                continue

            coarse_pos = self._normalize_wordlist_pos(pos)
            if coarse_pos:
                lookup[(headword, coarse_pos)] = level

            # 最初に出てきたレベルをフォールバックとして保持
            fallback.setdefault(headword, level)

        return lookup, fallback

    @staticmethod
    def _normalize_wordlist_pos(pos: str) -> Optional[str]:
        """CEFR-J語彙リストの品詞表記をspaCyのPOSに正規化"""
        pos = pos.lower()
        mapping = {
            'noun': 'NOUN',
            'verb': 'VERB',
            'adjective': 'ADJ',
            'adj': 'ADJ',
            'adverb': 'ADV',
            'adv': 'ADV',
            'proper noun': 'PROPN',
            'pv': 'VERB',
            'modal verb': 'VERB'
        }
        return mapping.get(pos)

    def _lookup_word_level(self, token) -> str:
        """語のCEFRレベルを辞書および頻度情報から推定"""
        lemma = token.lemma_.lower()
        surface = token.text.lower()
        token_pos = token.pos_

        level = None
        if self.cefr_by_pos:
            level = self.cefr_by_pos.get((lemma, token_pos))
            if not level:
                level = self.cefr_by_pos.get((surface, token_pos))
        if not level and self.cefr_fallback:
            level = self.cefr_fallback.get(lemma) or self.cefr_fallback.get(surface)

        if not level:
            if token_pos != 'PROPN':
                level = 'C1'
            else:
                level = self._infer_level_from_frequency(surface, lemma)

        return level

    @staticmethod
    def _infer_level_from_frequency(surface: str, lemma: str) -> str:
        """語の頻度情報からCEFRレベルを近似"""
        zipf = zipf_frequency(surface, 'en')
        if zipf == 0:
            zipf = zipf_frequency(lemma, 'en')

        if zipf >= 4.5:
            return 'A1'
        if zipf >= 4.0:
            return 'A2'
        if zipf >= 3.5:
            return 'B1'
        if zipf >= 3.0:
            return 'B2'
        if zipf >= 2.5:
            return 'C1'
        return 'C2'

    @staticmethod
    def _is_content_word(token) -> bool:
        return token.pos_ in CONTENT_POS and token.is_alpha

    @staticmethod
    def _is_function_word(token) -> bool:
        return token.pos_ in FUNCTION_POS

    def _estimate_frequency_rank(self, word: str) -> int:
        """語の頻度ランクをZipf頻度から近似"""
        zipf = zipf_frequency(word, 'en')
        if zipf <= 0:
            return FALLBACK_FREQUENCY_RANK

        # Zipf頻度(0-7)を順位に変換（Zipfが高いほど順位を小さく）
        scaled = max(0.0, min(7.0, 7.0 - zipf))
        rank = int(round(10 ** scaled))
        return max(1, min(rank, FALLBACK_FREQUENCY_RANK))

    def calculate_metrics(self, text: str) -> ReadabilityMetrics:
        """テキストから読みやすさ指標を計算"""
        if not text or not text.strip():
            return ReadabilityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_space and not token.is_punct]

        if not tokens:
            return ReadabilityMetrics(0, 0, 0, 0, 0, 0, 0, 0)

        content_tokens = [token for token in tokens if self._is_content_word(token)]
        content_levels = []
        a_level = b_level = 0

        for token in content_tokens:
            level = self._lookup_word_level(token)
            score = LEVEL_SCORES.get(level, LEVEL_SCORES['B2'])
            content_levels.append(score)

            if score <= LEVEL_SCORES['A2']:
                a_level += 1
            elif score >= LEVEL_SCORES['B1']:
                b_level += 1

        avr_diff = float(np.mean(content_levels)) if content_levels else 0.0
        bper_a = float(b_level) / max(1, a_level)

        verbs = [token for token in doc if token.pos_ == 'VERB']
        verb_lemmas = {token.lemma_.lower() for token in verbs}
        total_verbs = len(verbs)
        cvv1 = ((len(verb_lemmas) / math.sqrt(total_verbs)) * 2) if total_verbs else 0.0

        ranks = []
        for token in tokens:
            if token.is_alpha:
                ranks.append(self._estimate_frequency_rank(token.text.lower()))
        if len(ranks) > 3:
            ranks = sorted(ranks)[:-3]  # 末尾3件（最も頻度が低い語）を除外
        avr_freq_rank = float(np.mean(ranks)) if ranks else 0.0

        try:
            ari = float(textstat.automated_readability_index(text))
        except Exception:
            ari = 0.0

        sentences = [sent for sent in doc.sents if any(not t.is_space for t in sent)]
        sentence_count = len(sentences) if sentences else 1
        vper_sent = total_verbs / sentence_count

        alpha_tokens = [token for token in tokens if token.is_alpha]
        unique_pos = {token.pos_ for token in alpha_tokens}
        postypes = float(len(unique_pos))

        total_np_tokens = 0
        noun_phrase_count = 0
        try:
            for chunk in doc.noun_chunks:
                length = sum(1 for token in chunk if token.is_alpha)
                if length:
                    total_np_tokens += length
                    noun_phrase_count += 1
        except ValueError:
            # noun_chunksは解析に失敗することがある
            total_np_tokens = 0
            noun_phrase_count = 0
        lennp = (total_np_tokens / noun_phrase_count) if noun_phrase_count else 0.0

        return ReadabilityMetrics(
            AvrDiff=avr_diff,
            BperA=bper_a,
            CVV1=cvv1,
            AvrFreqRank=avr_freq_rank,
            ARI=ari,
            VperSent=vper_sent,
            POStypes=postypes,
            LenNP=lennp
        )

    def analyze_lexical_profile(self, text: str, max_rank: int) -> Dict:
        """語彙制約に対する逸脱をチェック"""
        if not text or max_rank <= 0:
            return {
                'max_rank': max_rank,
                'total_tokens': 0,
                'content_tokens': 0,
                'out_of_band_count': 0,
                'out_of_band_tokens': []
            }

        doc = self.nlp(text)
        content_tokens = [
            token for token in doc
            if token.is_alpha and not token.is_stop and token.pos_ != 'PROPN'
        ]

        flagged: Dict[str, int] = {}
        for token in content_tokens:
            token_text = token.text.lower()
            rank = self._estimate_frequency_rank(token_text)
            level = self._lookup_word_level(token)
            if level in {'C1', 'C2'} and max_rank < 7000:
                flagged[token.text] = rank
                continue
            if rank > max_rank:
                flagged[token.text] = rank

        return {
            'max_rank': max_rank,
            'total_tokens': len([token for token in doc if token.is_alpha]),
            'content_tokens': len(content_tokens),
            'out_of_band_count': len(flagged),
            'out_of_band_tokens': [f"{word} (rank≈{rank})" for word, rank in flagged.items()]
        }

# ========================================
# 3. アプローチ1: プロンプトエンジニアリングベース
# ========================================

class PromptBasedGenerator:
    """プロンプトエンジニアリングによる生成"""
    
    def __init__(self, api_key: str, analyzer: ReadabilityAnalyzer):
        self.api_key = api_key
        self.analyzer = analyzer
        openai.api_key = api_key
        
    def generate_with_level_constraints(self, 
                                       target_level: str,
                                       topic: str,
                                       band_label: str,
                                       max_rank: int,
                                       word_count: int,
                                       min_words: int,
                                       max_words: int,
                                       text_type: str = 'narrative') -> Dict[str, Union[str, List[str]]]:
        """レベル制約付きでテキストを生成し、質問も取得"""
        if target_level not in self.analyzer.level_profiles:
            raise ValueError(f"No readability profile available for level {target_level}")

        if text_type not in VALID_TEXT_TYPES:
            raise ValueError(f"Text type must be one of {VALID_TEXT_TYPES}")

        profile = self.analyzer.level_profiles[target_level]
        metric_requirements = self._format_metric_requirements(profile)
        display_name = self.analyzer.get_display_name(target_level)
        type_guideline = self._get_type_guideline(text_type)
        lexical_brief = (
            f"Difficulty band: {band_label}. Use vocabulary primarily within the most frequent {max_rank} word families. "
            "If an unavoidable higher-level word appears, immediately support it with a simple paraphrase or explanation. "
            "Avoid strings of low-frequency phrasal verbs or idioms that would exceed the band expectations."
        )
        
        lower_bound = min_words
        upper_bound = max_words

        question_rules = (
            "Question design requirements:\n"
            "- question1: Ask learners to reconstruct a key sequence, cause, or factual explanation from the passage. Reference concrete details from the text and finish with『50語程度で答えてください。』\n"
            "- question2: Ask learners to interpret motives, consequences, or implications using explicit evidence from the passage. Mention the topic explicitly and finish with『50語程度で答えてください。』"
        )

        prompt = f"""
        You are an expert English writer producing a CEFR {display_name} {text_type} reading passage about "{topic}".

        Constraints:
        - Target length: between {lower_bound} and {upper_bound} English words (aim for {word_count}). Revise internally until you satisfy this window.
        - {lexical_brief}
        - Maintain syntax and cohesion typical for CEFR {display_name} learners.
        - Text type requirement: {type_guideline}

        Readability Targets (match quantitative metrics):
        {metric_requirements}

        Style Guidelines for {target_level}:
        {self._get_style_guidelines(target_level)}

        {question_rules}

        Workflow:
        1. Plan the {text_type} structure internally (do not output the outline).
        2. Draft the passage respecting all constraints above.
        3. Compose two questions in Japanese that satisfy the question design requirements.
        4. Respond ONLY with a JSON object (no prose, no code fences):
           {{
             "text": "... English passage ...",
             "question1": "... Japanese comprehension question ... 50語程度で答えてください。",
             "question2": "... Japanese comprehension question ... 50語程度で答えてください。"
           }}
        """

        system_prompt = (
            "You must respond with a strict JSON object containing keys text, question1, question2. "
            "Do not include explanations, bullet lists, code fences, or any extra commentary."
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        raw = response.choices[0].message.content.strip()
        return self._parse_generation_output(raw)

    @staticmethod
    def _parse_generation_output(raw: str) -> Dict[str, Union[str, List[str]]]:
        """LLM出力からJSONを解析"""
        candidate = raw.strip()
        if candidate.startswith('```'):
            blocks = candidate.split('```')
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                if block.lower().startswith('json'):
                    candidate = block[4:].strip()
                    break
                candidate = block

        data = None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass

        if not isinstance(data, dict):
            return {
                'text': PromptBasedGenerator._sanitize_passage(raw),
                'questions': []
            }

        text = str(data.get('text', '')).strip()
        questions: List[str] = []

        if isinstance(data.get('questions'), list):
            questions = [str(q).strip() for q in data['questions'] if q]
        else:
            for key in ('question1', 'question2'):
                if key in data and data[key]:
                    questions.append(str(data[key]).strip())

        return {
            'text': PromptBasedGenerator._sanitize_passage(text),
            'questions': questions[:2]
        }

    @staticmethod
    def _sanitize_passage(text: str) -> str:
        if not text:
            return ''
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                lines.append('')
                continue
            if re.match(r'^\d+\.\s*(outline|text|passage|questions?)', lowered):
                continue
            if lowered.startswith(('outline:', 'passage:', 'text:', 'json object:', 'questions:', 'japanese questions')):
                continue
            lines.append(stripped)
        sanitized = '\n'.join(line for line in lines if line is not None).strip()
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        return sanitized

    @staticmethod
    def _normalize_questions(questions: List[str], topic: str, text_type: str) -> List[str]:
        normalized = []
        for question in questions[:2]:
            q = question.strip()
            if '50語程度' not in q:
                if q.endswith('。'):
                    q = q[:-1]
                q += '。50語程度で答えてください。'
            normalized.append(q)

        while len(normalized) < 2:
            normalized.append(PromptBasedGenerator._default_question(topic, text_type, len(normalized) + 1))

        return normalized[:2]

    @staticmethod
    def _default_question(topic: str, text_type: str, index: int) -> str:
        topic_phrase = topic if topic else '本文'
        if text_type == 'narrative':
            if index == 1:
                return f"{topic_phrase} に関して本文で起きた出来事や登場人物の行動を具体的に整理し、根拠となる描写とともに50語程度で答えてください。"
            return f"{topic_phrase} の結果として登場人物が抱いた感情や学んだことを、本文の記述を引用しながら50語程度で説明してください。"
        # expository
        if index == 1:
            return f"{topic_phrase} について本文で示された主要なポイントや事実を要約し、重要な根拠を添えて50語程度で答えてください。"
        return f"{topic_phrase} に関する本文の説明から、理由や影響を踏まえて自分の言葉で50語程度で考察を述べてください。"
    
    def _get_style_guidelines(self, level: str) -> str:
        """レベル別のスタイルガイドライン"""
        guidelines = {
            'A1': "Use present simple and very common phrases. Keep sentences under 10 words with concrete vocabulary.",
            'A2': "Use present and past simple with occasional future forms. Maintain short clear sentences on familiar topics.",
            'B1.1': "Various tenses. Compound sentences. Concrete topics with some abstract ideas.",
            'B1.2': "Complex sentences allowed. Mix of concrete and abstract topics.",
            'B2.1': "Sophisticated grammar. Abstract concepts. Some idiomatic expressions.",
            'B2.2': "Advanced structures. Nuanced vocabulary. Complex argumentation.",
            'C1+': "Near-native complexity. Implicit meanings. Wide range of expressions including nuanced vocabulary."
        }
        return guidelines.get(level, "Standard English appropriate for the level.")

    @staticmethod
    def _get_type_guideline(text_type: str) -> str:
        descriptions = {
            'narrative': "Narrative prose with characters, events, and a clear beginning–middle–end. Use temporal connectors and a personal or character-driven voice.",
            'expository': "Expository/informational prose that explains facts or concepts clearly and logically. Use neutral tone, topic sentences, and supporting details."
        }
        return descriptions[text_type]

    def _format_metric_requirements(self, profile: Dict[str, Dict]) -> str:
        """定量指標をプロンプトに組み込む文字列を生成"""
        mean = profile.get('mean', {})
        std = profile.get('std', {})

        def tolerance(metric: str) -> str:
            if metric not in mean:
                return ""
            base = f"≈ {mean[metric]:.2f}"
            spread = std.get(metric)
            if spread and not np.isnan(spread):
                base += f" (±{spread:.2f})"
            return base

        hints = {
            'AvrDiff': "Average CEFR difficulty of content words (1=A1 … 6=C2). Keep vocabulary near the target difficulty.",
            'BperA': "Ratio of B-level to A-level content words. Lower values mean simpler vocabulary.",
            'CVV1': "Verb diversity. Use a controlled set of verbs when the target is low.",
            'AvrFreqRank': "Average frequency rank (lower is more common). Choose common words when target is low.",
            'ARI': "Automated Readability Index. Shorter words/sentences reduce this value.",
            'VperSent': "Average verbs per sentence. Simpler constructions use fewer verbs.",
            'POStypes': "POS diversity. Limit variety to simplify when necessary.",
            'LenNP': "Average noun phrase length per sentence. Use compact noun phrases if the target is low."
        }

        lines = []
        for metric, hint in hints.items():
            target = tolerance(metric)
            if target:
                lines.append(f"- {metric}: {target}. {hint}")

        return "\n        ".join(lines) if lines else "- Follow standard CEFR-J readability expectations."  # インデントは上位f文字列で調整

# ========================================
# 4. アプローチ2: 反復改善ベース
# ========================================

class IterativeRefinementGenerator:
    """生成と分析を繰り返して目標レベルに近づける"""
    
    def __init__(self, api_key: str, analyzer: ReadabilityAnalyzer):
        self.api_key = api_key
        self.analyzer = analyzer
        openai.api_key = api_key
        
    def generate_with_refinement(self,
                                target_level: str,
                                topic: str,
                                max_iterations: int = 5,
                                text_type: str = 'narrative') -> Dict:
        """
        反復的に改善しながら生成
        """
        if text_type not in VALID_TEXT_TYPES:
            raise ValueError(f"Text type must be one of {VALID_TEXT_TYPES}")

        current_text = self._initial_generation(target_level, topic, text_type)
        history = []
        
        for iteration in range(max_iterations):
            # 現在のテキストの指標を計算（仮想的）
            metrics = self._estimate_metrics(current_text)
            distance = self._calculate_distance(metrics, target_level)
            
            history.append({
                'iteration': iteration,
                'text': current_text,
                'metrics': metrics,
                'distance': distance
            })
            
            if distance < 0.1:  # 十分に近い
                break
                
            # 改善指示を生成
            refinement_prompt = self._create_refinement_prompt(
                current_text, metrics, target_level, text_type
            )
            
            # テキストを改善
            current_text = self._refine_text(refinement_prompt)
            
        return {
            'final_text': current_text,
            'iterations': len(history),
            'history': history,
            'text_type': text_type
        }
    
    def _initial_generation(self, level: str, topic: str, text_type: str) -> str:
        """初期テキスト生成"""
        type_instruction = PromptBasedGenerator._get_type_guideline(text_type)
        prompt = (
            f"Write a CEFR {level} {text_type} English text about {topic}. About 200 words. "
            f"Ensure the text follows this requirement: {type_instruction}"
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _estimate_metrics(self, text: str) -> Dict:
        """spaCyや語彙リストを用いて全指標を算出"""
        metrics = self.analyzer.calculate_metrics(text)
        return asdict(metrics)
    
    def _calculate_distance(self, metrics: Dict, target_level: str) -> float:
        """目標レベルとの距離を計算"""
        if target_level not in self.analyzer.level_profiles:
            raise ValueError(f"No readability profile available for level {target_level}")

        target_profile = self.analyzer.level_profiles[target_level]['mean']
        shared_keys = [key for key in metrics.keys() if key in target_profile]
        if not shared_keys:
            return float('inf')

        total = 0.0
        for key in shared_keys:
            measured = metrics[key]
            target_value = target_profile[key]
            scale = abs(target_value) if abs(target_value) > 1e-6 else 1.0
            total += abs(measured - target_value) / scale

        return total / len(shared_keys)
    
    def _create_refinement_prompt(self, text: str, metrics: Dict, target_level: str, text_type: str) -> str:
        """改善プロンプトを生成"""
        if target_level not in self.analyzer.level_profiles:
            raise ValueError(f"No readability profile available for level {target_level}")

        target_profile = self.analyzer.level_profiles[target_level]['mean']

        adjustments = []

        def add_adjustment(metric: str,
                           high_text: str,
                           low_text: str,
                           threshold: float) -> None:
            if metric not in metrics or metric not in target_profile:
                return
            diff = metrics[metric] - target_profile[metric]
            if abs(diff) < threshold:
                return
            if diff > 0:
                adjustments.append(high_text.format(diff=diff, target=target_profile[metric]))
            else:
                adjustments.append(low_text.format(diff=diff, target=target_profile[metric]))

        add_adjustment(
            'AvrDiff',
            "Replace higher-level vocabulary with CEFR-J A-level synonyms until average difficulty drops toward {target:.2f} (currently +{diff:.2f}).",
            "Introduce a few B-level content words so average difficulty rises toward {target:.2f} (currently {diff:.2f}).",
            threshold=0.15
        )
        add_adjustment(
            'BperA',
            "Reduce the proportion of B-level content words by swapping them for A-level alternatives.",
            "Add a handful of B-level words to avoid oversimplification.",
            threshold=0.1
        )
        add_adjustment(
            'CVV1',
            "Reuse core verbs or repeat key verb lemmas to lower verb diversity.",
            "Introduce additional distinct verbs to increase diversity.",
            threshold=0.05
        )
        add_adjustment(
            'AvrFreqRank',
            "Choose more common vocabulary items from the CEFR-J list to lower the average frequency rank.",
            "Incorporate a few topic-appropriate mid-frequency words to avoid being overly basic.",
            threshold=50
        )
        add_adjustment(
            'ARI',
            "Break long sentences or shorten words to reduce ARI toward {target:.2f}.",
            "Combine short sentences or add clauses to raise ARI toward {target:.2f}.",
            threshold=0.5
        )
        add_adjustment(
            'VperSent',
            "Use simpler sentence structures with fewer verbs per sentence.",
            "Add supporting clauses with verbs to reach the target density.",
            threshold=0.1
        )
        add_adjustment(
            'POStypes',
            "Limit parts-of-speech variety; reuse established patterns to reduce diversity.",
            "Include more adjectives/adverbs or alternate sentence patterns to raise POS diversity.",
            threshold=0.02
        )
        add_adjustment(
            'LenNP',
            "Split long noun phrases into shorter units or add determiners/pronouns to simplify.",
            "Extend noun phrases with modifiers to increase their average length.",
            threshold=0.5
        )

        if not adjustments:
            adjustments.append("Metrics already align; focus on fluency and naturalness within the level.")

        metrics_snapshot = "\n".join(
            f"- {key}: current {metrics[key]:.2f}, target {target_profile.get(key, float('nan')):.2f}"
            for key in metrics if key in target_profile
        )

        prompt = f"""
        Adjust this text to better match {target_level} level:
        
        Current text: {text}
        
        Text type requirement: {PromptBasedGenerator._get_type_guideline(text_type)}

        Metric comparison:
        {metrics_snapshot}

        Required adjustments:
        {chr(10).join(f"- {adj}" for adj in adjustments)}
        
        Revised text:
        """
        return prompt
    
    def _refine_text(self, prompt: str) -> str:
        """テキストを改善"""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# ========================================
# 5. アプローチ3: テンプレートベース生成
# ========================================

class TemplateBasedGenerator:
    """レベル別テンプレートを使用した生成"""
    
    def __init__(self, analyzer: ReadabilityAnalyzer):
        self.analyzer = analyzer
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """レベル別のテンプレートを定義"""
        return {
            'A1': {
                'narrative': [
                    "This is {character}. {character} lives in {place}. Every day {character} {daily_activity}.",
                    "One day, {character} meets {friend}. They {shared_activity} and feel {feeling}.",
                    "At night, {character} thinks about {simple_dream} and feels {emotion}."
                ],
                'expository': [
                    "{topic} is about {simple_fact}. People use it to {simple_use}.",
                    "It is usually found in {simple_location}. Many people think it is {adjective}.",
                    "When you see {topic}, you can {simple_action} and feel {emotion}."
                ]
            }
        }

    def generate_from_template(self, 
                               level: str, 
                               text_type: str,
                               content_dict: Dict[str, str]) -> str:
        """テンプレートを使用してテキストを生成"""
        template_bundle = self.templates.get(level, {})
        type_templates = template_bundle.get(text_type, [])
        if not type_templates:
            raise ValueError(f"No templates defined for level {level} and text type {text_type}")

        generated_sentences = []
        for template in type_templates:
            sentence = template.format(**content_dict)
            generated_sentences.append(sentence)

        return " ".join(generated_sentences)

# ========================================
# 6. アプローチ4: 機械学習モデルベース
# ========================================

class MLBasedGenerator:
    """機械学習モデルを使用した生成"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = self._load_or_train_model(model_path)
        
    def _load_or_train_model(self, model_path: Optional[str]):
        """モデルをロードまたは訓練"""
        if model_path:
            # 既存モデルをロード
            import joblib
            return joblib.load(model_path)
        else:
            # 新規モデルを訓練
            from sklearn.ensemble import RandomForestRegressor
            # 訓練コード...
            return RandomForestRegressor()
    
    def train_level_predictor(self, texts: List[str], levels: List[str]):
        """テキストからレベルを予測するモデルを訓練"""
        # 特徴量抽出
        features = [self._extract_features(text) for text in texts]
        
        # モデル訓練
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded_levels = le.fit_transform(levels)
        
        self.model.fit(features, encoded_levels)
        self.label_encoder = le
        
    def _extract_features(self, text: str) -> np.ndarray:
        """テキストから特徴量を抽出"""
        import textstat
        
        features = [
            textstat.flesch_reading_ease(text),
            textstat.automated_readability_index(text),
            textstat.coleman_liau_index(text),
            len(text.split()),
            len(text.split('.'))
        ]
        return np.array(features)

# ========================================
# 7. 統合システム
# ========================================

class IntegratedTextGenerator:
    """複数のアプローチを統合したシステム"""
    
    def __init__(self,
                 excel_path: str,
                 api_key: Optional[str] = None,
                 cefr_wordlist_path: Optional[str] = None):
        self.analyzer = ReadabilityAnalyzer(
            training_data_path=excel_path,
            cefr_wordlist_path=cefr_wordlist_path
        )
        self.generators = {
            'prompt': PromptBasedGenerator(api_key, self.analyzer) if api_key else None,
            'iterative': IterativeRefinementGenerator(api_key, self.analyzer) if api_key else None,
            'template': TemplateBasedGenerator(self.analyzer),
            'ml': MLBasedGenerator()
        }
        
    def generate(self, 
                target_level: str,
                topic: str,
                method: str = 'auto',
                text_type: str = 'narrative',
                **kwargs) -> Dict:
        """
        指定されたレベルのテキストを生成
        
        Parameters:
        -----------
        target_level: 目標CEFRレベル (A1.1, A1.2, ..., C2)
        topic: トピック
        method: 生成方法 ('prompt', 'iterative', 'template', 'ml', 'auto')
        
        Returns:
        --------
        生成結果の辞書
        """
        
        level_key = self.analyzer.resolve_level(target_level)

        if text_type not in VALID_TEXT_TYPES:
            raise ValueError(f"text_type must be one of {VALID_TEXT_TYPES}")

        if method == 'auto':
            # レベルに応じて最適な方法を選択
            if level_key == 'A1':
                method = 'template'  # 簡単なレベルはテンプレート
            elif level_key in ['A2', 'B1.1', 'B1.2', 'B2.1', 'B2.2']:
                method = 'iterative'  # 中級は反復改善
            else:
                method = 'prompt'  # 上級はプロンプト

        generator = self.generators.get(method)
        if not generator:
            raise ValueError(f"Method {method} not available")
            
        # 生成実行
        evaluation = None

        if method == 'prompt':
            max_attempts = kwargs.get('generation_attempts', 3)
            band_label = kwargs.get('band_label', 'N/A')
            max_rank = kwargs.get('max_rank', 7000)
            target_words = kwargs.get('word_count', 200)

            for attempt in range(max_attempts):
                payload = generator.generate_with_level_constraints(
                    level_key,
                    topic,
                    band_label,
                    max_rank,
                    target_words,
                    kwargs.get('min_words', 180),
                    kwargs.get('max_words', 220),
                    text_type=text_type
                )

                questions = PromptBasedGenerator._normalize_questions(payload.get('questions', []), topic, text_type)
                result = {
                    'text': payload.get('text', ''),
                    'questions': questions,
                    'method': method
                }

                evaluation = self._evaluate_generation(result['text'], level_key)
                word_count_actual = evaluation.get('word_count', 0)
                lower = kwargs.get('min_words', 180)
                upper = kwargs.get('max_words', 220)
                if lower <= word_count_actual <= upper or attempt == max_attempts - 1:
                    break
            # evaluation reused later
        elif method == 'iterative':
            result = generator.generate_with_refinement(
                level_key, topic, kwargs.get('max_iterations', 5), text_type=text_type
            )
            result['method'] = method

        elif method == 'template':
            content = kwargs.get('content_dict', self._default_content(topic, text_type))
            text = generator.generate_from_template(level_key, text_type, content)
            result = {
                'text': text,
                'questions': PromptBasedGenerator._normalize_questions(
                    kwargs.get('questions', []), topic, text_type
                ),
                'method': method
            }

        else:
            raise ValueError(f"Unknown method: {method}")

        if method != 'prompt':
            evaluation = self._evaluate_generation(
                result.get('text', result.get('final_text', '')),
                level_key
            )

        # 基本情報を付与
        result['evaluation'] = evaluation
        result['level'] = level_key
        result['text_type'] = text_type
        result['label'] = self._format_label(level_key, text_type)
        result['topic'] = topic
        result['target_level'] = level_key
        result['requested_level'] = target_level
        result['text'] = result.get('text', result.get('final_text', ''))
        result['word_count'] = evaluation.get('word_count', 0)
        result['distance_score'] = evaluation.get('distance_score', float('inf'))
        result['delta_from_target'] = evaluation.get('delta_from_target', {})
        result['features'] = evaluation.get('metrics', {})
        result['attempts'] = result.get('iterations', 1)
        result['band'] = kwargs.get('band_label')

        if 'questions' not in result or not result['questions']:
            result['questions'] = PromptBasedGenerator._normalize_questions([], topic, text_type)

        max_rank = kwargs.get('max_rank')
        if max_rank:
            result['lexical_profile'] = self.analyzer.analyze_lexical_profile(result['text'], max_rank)
        else:
            result['lexical_profile'] = {
                'max_rank': None,
                'total_tokens': 0,
                'content_tokens': 0,
                'out_of_band_count': 0,
                'out_of_band_tokens': []
            }

        return result
    
    def _default_content(self, topic: str, text_type: str) -> Dict:
        """デフォルトのコンテンツ辞書を生成"""
        base = {
            'topic': topic,
            'character': 'Sora',
            'place': 'a small town',
            'daily_activity': 'walks to school with a smile',
            'friend': 'Mika',
            'shared_activity': 'play in the park',
            'feeling': 'happy',
            'simple_dream': 'sharing stories with friends',
            'emotion': 'calm',
            'simple_fact': f'{topic} helps people every day',
            'simple_use': 'learn new things',
            'simple_location': 'many towns',
            'simple_action': 'think about new ideas',
            'adjective': 'helpful'
        }
        if text_type == 'narrative':
            return base
        if text_type == 'expository':
            return base
        return base
    
    def _evaluate_generation(self, text: str, target_level: str) -> Dict:
        """生成されたテキストを評価"""
        metrics = self.analyzer.calculate_metrics(text)
        metrics_dict = asdict(metrics)

        target_profile = self.analyzer.level_profiles.get(target_level, {}).get('mean', {})
        delta = {
            key: metrics_dict[key] - target_profile[key]
            for key in metrics_dict
            if key in target_profile
        }

        doc = self.analyzer.nlp(text)
        sentences = [sent for sent in doc.sents if any(not token.is_space for token in sent)]
        word_count = len([token for token in doc if token.is_alpha])

        shared_keys = delta.keys()
        if shared_keys:
            total = 0.0
            for key in shared_keys:
                target_val = target_profile[key]
                scale = abs(target_val) if abs(target_val) > 1e-6 else 1.0
                total += abs(delta[key]) / scale
            distance = total / len(list(shared_keys))
        else:
            distance = float('inf')

        return {
            'target_level': target_level,
            'word_count': word_count,
            'sentence_count': len(sentences),
            'metrics': metrics_dict,
            'delta_from_target': delta,
            'distance_score': distance
        }

    def generate_dual_texts(self,
                            target_level: str,
                            topics: Union[Dict[str, str], str],
                            method: str = 'auto',
                            word_count: int = 200,
                            **kwargs) -> Dict:
        """同一レベルで物語文と説明文の両方を生成"""
        level_key = self.analyzer.resolve_level(target_level)
        topic_map = self._normalize_topics(topics)
        outputs = {}

        for text_type, topic in topic_map.items():
            result = self.generate(
                level_key,
                topic,
                method=method,
                text_type=text_type,
                word_count=word_count,
                **kwargs
            )
            outputs[text_type] = result

        return {
            'level': level_key,
            'display_name': self.analyzer.get_display_name(level_key),
            'texts': outputs
        }

    @staticmethod
    def _normalize_topics(topics: Union[Dict[str, str], str]) -> Dict[str, str]:
        if isinstance(topics, str):
            return {'narrative': topics, 'expository': topics}

        normalized = {
            text_type: topic
            for text_type, topic in topics.items()
            if text_type in VALID_TEXT_TYPES
        }

        if not normalized:
            normalized = {'narrative': 'Daily life', 'expository': 'Daily life'}
        else:
            first_topic = next(iter(normalized.values()))
            for required in VALID_TEXT_TYPES:
                normalized.setdefault(required, first_topic)

        return normalized

    def _format_label(self, level_key: str, text_type: str) -> str:
        display = self.analyzer.get_display_name(level_key)
        type_label = text_type.capitalize()
        return f"CEFR-J Level: {display} ({type_label})"

def print_generation_report(result: Dict) -> None:
    """生成結果の評価指標を整理して表示"""
    evaluation = result.get('evaluation', {})
    metrics = evaluation.get('metrics', {})
    delta = evaluation.get('delta_from_target', {})

    def _fmt(value: Optional[float]) -> str:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return 'N/A'
            return f"{value:.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    header = (
        f"Band: {result.get('band', '-')}, {result.get('label', 'Unknown')}"
        f" | Method: {result.get('method', 'unknown')}"
        f" | Type: {result.get('text_type', '-') }"
    )
    print(header)
    print(
        f"Topic: {result.get('topic', '-')}"
        f" | Distance: {_fmt(evaluation.get('distance_score'))}"
        f" | Word count: {evaluation.get('word_count', 'N/A')}"
        f" | Sentences: {evaluation.get('sentence_count', 'N/A')}"
    )

    if not metrics:
        return

    print("Metric      Current  Target   Delta")
    for key in METRIC_KEYS:
        if key not in metrics:
            continue
        current = metrics.get(key)
        diff = delta.get(key, 0.0)
        target = current - diff if current is not None else None
        print(f"{key:<10}  {_fmt(current):>7}  {_fmt(target):>7}  {_fmt(diff):>7}")

    lexical_profile = result.get('lexical_profile', {})
    if lexical_profile:
        print(
            f"Out-of-band content words (> rank {lexical_profile.get('max_rank', 'N/A')}): "
            f"{lexical_profile.get('out_of_band_count', 0)}"
        )
        if lexical_profile.get('out_of_band_tokens'):
            examples = ', '.join(lexical_profile['out_of_band_tokens'][:10])
            print(f"Examples: {examples}")
        print()

# ========================================
# 8. 使用例
# ========================================

def main():
    """使用例"""

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is not set. Add it to your .env file before running this script.')

    generator = IntegratedTextGenerator(
        excel_path='/Users/tohokusla/Dropbox/科研_CAT/Material/Text_Generation/CVLA3_20250912133649_3373.xlsx',
        api_key=api_key,
        cefr_wordlist_path='/Users/tohokusla/Dropbox/科研_CAT/Material/Text_Generation/CEFR-J Wordlist Ver1.6.xlsx'
    )
    output_manager = OutputManager(output_dir='outputs')
    collected_results: List[Dict] = []

    try:
        sets_input = input('Number of text sets to generate (default=1): ').strip()
    except EOFError:
        sets_input = ''

    try:
        sets_count = int(sets_input) if sets_input else 1
    except ValueError:
        sets_count = 1

    sets_count = max(1, sets_count)

    band_order = ['2K', '3K', '4K', '5K', '6K', '7K']

    for set_index in range(1, sets_count + 1):
        for band_label in band_order:
            band_conf = DIFFICULTY_BANDS[band_label]
            for text_type in ('narrative', 'expository'):
                topic = band_conf['topics'][text_type]
                result = generator.generate(
                    target_level=band_conf['anchor_level'],
                    topic=topic,
                    method='prompt',
                    text_type=text_type,
                    band_label=band_label,
                    max_rank=band_conf['max_rank'],
                    word_count=band_conf['word_count'],
                    min_words=180,
                    max_words=220
                )
                result['set_index'] = set_index
                collected_results.append(result)

                print("\n" + "=" * 80)
                print(f"Set {set_index} | Band {band_label} | {text_type.capitalize()} | Topic: {topic}")
                print("-" * 80)
                print(result['text'])
                print("\n[質問]")
                print(f"Q1: {result['questions'][0]}")
                print(f"Q2: {result['questions'][1]}")
                print_generation_report(result)

    saved_paths = output_manager.save_batch_results(collected_results, output_name='reading_band_set')
    print("\nSaved files:")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")

if __name__ == "__main__":
    # 必要なライブラリのインストールコマンド
    print("""
    Required libraries installation:
    pip install pandas numpy openai textstat spacy scikit-learn joblib openpyxl
    python -m spacy download en_core_web_sm
    """)
    
    # メイン実行
    main()
