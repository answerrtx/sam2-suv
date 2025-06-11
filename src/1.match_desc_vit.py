import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

class MedicalConfig:
    WEIGHTS = {
        'semantic': 0.6,
        'name': 0.2,
        'color': 0.05,
        'shape': 0.05,
        'transducer': 0.05,
        'anatomy': 0.05
    }
    MODEL_CHOICE = 'all-MiniLM-L6-v2'
    THRESHOLD = 0.2

class DescriptionParser:
    @staticmethod
    def parse_description(text):
        features = {
            'name': [],
            'color': [],
            'shape': [],
            'transducer': [],
            'anatomy': [],
            'bio_features': []
        }

        name_matches = re.findall(r'(\b[A-Z]{3,}\b)\s*\((.*?)\)', text)
        for abbrev, full in name_matches:
            features['name'].extend([abbrev, full])

        return features  # 不再定义具体 pattern 提取其他类别

class MedicalTermMapper:
    MEDICAL_LEXICON = {
        'mc': {'standard': 'metacarpal', 'features': ['thick', 'curved']},
        'prxpx': {'standard': 'proximal phalanx', 'features': ['short', 'curved']},
        'ucl': {'standard': 'ulnar collateral ligament', 'features': ['band-like']},
        'small and white curve': {'mapping': ['prxpx', 'mc']},
        'sp': {'standard': 'vertebral','mapping': ['sp', 'vertebral arch']},
        'bone structure': {'mapping': ['prxpx', 'mc', 'vertebra']}
    }

    @classmethod
    def expand_terms(cls, text):
        for term in cls.MEDICAL_LEXICON:
            if term in text.lower():
                mappings = cls.MEDICAL_LEXICON[term].get('mapping', [])
                text += ' ' + ' '.join(mappings)
        return text

class FeatureEncoder:
    def __init__(self, config):
        self.model = SentenceTransformer(config.MODEL_CHOICE)

    def encode_text(self, text):
        return self.model.encode(text, convert_to_numpy=True).reshape(1, -1)

class StructureMatcher:
    def __init__(self, desc_dir):
        self.config = MedicalConfig()
        self.encoder = FeatureEncoder(self.config)
        self.term_mapper = MedicalTermMapper()
        self.db = self._build_database(desc_dir)

    def _build_database(self, desc_dir):
        database = []
        for file in os.listdir(desc_dir):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(desc_dir, file)) as f:
                text = f.read()
                features = DescriptionParser.parse_description(text)
                database.append({
                    'id': file.split('.')[0],
                    'features': features,
                    'raw_text': text
                })
        return database

    def _calculate_feature_similarity(self, query_feats, db_feats):
        query_text = query_feats['raw']
        db_text = db_feats['raw_text']
        semantic_sim = cosine_similarity(
            self.encoder.encode_text(query_text),
            self.encoder.encode_text(db_text)
        )[0][0]

        name_score = max([
            fuzz.token_set_ratio(query_feats['clean'], name) / 100
            for name in db_feats['features']['name']
        ], default=0)

        dummy_score = 0  # 没有定义 color/shape 等则设为 0

        total_score = (
            self.config.WEIGHTS['semantic'] * semantic_sim +
            self.config.WEIGHTS['name'] * name_score +
            self.config.WEIGHTS['color'] * dummy_score +
            self.config.WEIGHTS['shape'] * dummy_score +
            self.config.WEIGHTS['transducer'] * dummy_score +
            self.config.WEIGHTS['anatomy'] * dummy_score
        )

        return total_score

    def match_instruction(self, instruction):
        query_feats = self._parse_instruction(instruction)
        results = []
        for item in self.db:
            score = self._calculate_feature_similarity(query_feats, item)
            results.append((item['id'], item['raw_text'], score))

        results = [r for r in results if r[2] >= self.config.THRESHOLD]
        results.sort(key=lambda x: x[2], reverse=True)

        # 提取top1结构ID及其描述文本
        top_result = results[0] if results else (None, "", 0)
        top_struct_id, top_text, top_score = top_result

        # 在top文本中查找是否包含某些标准术语,这里需要改，只要找到top 1 结构ID就可以对着id找类别
        detected_terms = []
        for key, entry in self.term_mapper.MEDICAL_LEXICON.items():
            if isinstance(entry, dict) and 'standard' in entry:
                std_term = entry['standard']
                if std_term.lower() in top_text.lower() or key in top_text.lower():
                    detected_terms.append(key)

        return results, detected_terms, top_struct_id

    def _parse_instruction(self, instruction):
        parsed = DescriptionParser.parse_description(instruction)
        return {
            'raw': instruction,
            'clean': self.term_mapper.expand_terms(instruction),
            'color': [],
            'shape': [],
            'transducer': [],
            'anatomy': []
        }

# 示例用法
import argparse
def load_test_cases(path):

    if path.endswith(".txt"):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file format for test_cases. Use .txt or .json.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test LLM instruction matching to descriptions")
    parser.add_argument("--desc_dir", type=str, required=True, help="Path to the folder containing description files")
    parser.add_argument("--instruction", type=str, required=True, help="Path to text or JSON file containing test instructions")
    parser.add_argument("--api_key", type=str, required=False, help="API key for the LLM")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for the LLM (e.g., gpt-4, gemini-pro)")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output results")
    args = parser.parse_args()

    test_cases = load_test_cases(args.instruction)
    matcher = StructureMatcher(desc_dir=args.desc_dir)

    """
    "please segment the small and white curve from the video",
    "segment the structures with hyperechoic features"
    """
    for instruction in test_cases:
        print(f"\nInstruction: '{instruction}'")
        results, categories, top_id = matcher.match_instruction(instruction)
        for i, (struct_id, _, score) in enumerate(results[:3]):
            print(f"{i+1}. {struct_id} (confidence: {score:.2f})")
        print(f"→ Matched categories: {categories}")

        # 保存top1结构ID和类别
        if top_id:
            with open(args.output_path.replace('cat','id'), "w") as f:
                f.write(f"{top_id}\n")
            with open(args.output_path, "w") as f:
                f.write(", ".join(categories) + "\n")
