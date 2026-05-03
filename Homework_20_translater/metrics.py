import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from pandas import DataFrame, Series
from tqdm.auto import tqdm

class Metrics:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def levenshtein_distance(self, pred, target):
        return Levenshtein.distance(pred, target)
    
    def levenshtein_similarity(self, pred, target):
        return Levenshtein.ratio(pred, target)
    
    def jaccard_similarity(self, pred, target):
        set1 = set(pred.split())
        set2 = set(target.split())
        union = len(set1 | set2)
        return len(set1 & set2) / union if union else 1.0

    def tfidf_similarity(self, pred, target):
        p, t = (pred or "").strip(), (target or "").strip()
        if not p and not t:
            return 1.0
        if not p or not t:
            return 0.0
        try:
            X = self.vectorizer.fit_transform([p, t])
        except ValueError:
            return 1.0 if p == t else 0.0
        return float(cosine_similarity(X[0:1], X[1:2])[0, 0])
    
    def sequence_matcher(self, pred, target):
        return SequenceMatcher(None, pred, target).ratio()
    
    def bleu_score(self, pred, target):
        return sentence_bleu([target.split()], pred.split())
    
    def sentence_transformer(self, pred, target):
        emb1 = self.model.encode(pred, convert_to_tensor=True)
        emb2 = self.model.encode(target, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2)
        return similarity.item()

    def calculate_metrics(self, preds, targets):
        metrics = {
            "levenshtein_distance": 0,
            "levenshtein_similarity": 0,
            "jaccard_similarity": 0,
            "tfidf_similarity": 0,
            "sequence_matcher": 0,
            "bleu_score": 0,
            "sentence_transformer": 0
        }
        for pred, target in tqdm(zip(preds, targets), total=len(preds), desc="Calculating metrics"):
            metrics["levenshtein_distance"] += self.levenshtein_distance(pred, target)
            metrics["levenshtein_similarity"] += self.levenshtein_similarity(pred, target)
            metrics["jaccard_similarity"] += self.jaccard_similarity(pred, target)
            metrics["tfidf_similarity"] += self.tfidf_similarity(pred, target)
            metrics["sequence_matcher"] += self.sequence_matcher(pred, target)
            metrics["bleu_score"] += self.bleu_score(pred, target)
            metrics["sentence_transformer"] += self.sentence_transformer(pred, target)
        n = len(preds)
        return DataFrame({"mean": Series(metrics, dtype="float64") / n})