from typing import Callable, List
from evaluate import load as eval_load
import os

os.environ['HF_EVALUATE_OFFLINE'] = '1'

class BaseEvaluator:
    def __init__(self, metric_name: str, metric_fn: Callable):
        self.metric_name = metric_name
        self.metric_fn = metric_fn

    def evaluate(self, predictions, references):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_metric_name(self):
        return self.metric_name

class RougeRawEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("rouge_raw", eval_load("CZLC/rouge_raw"))

    def evaluate(self, predictions, references):
        return self.metric_fn.compute(predictions=predictions, references=references)

class RougeEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("rouge", eval_load("rouge"))

    def evaluate(self, predictions, references):
        return self.metric_fn.compute(predictions=predictions, references=references)

class BertScoreEvaluator(BaseEvaluator):
    def __init__(self, lang="cz"):
        super().__init__("bertscore", eval_load("bertscore"))
        self.lang = lang

    def evaluate(self, predictions, references):
        scores = self.metric_fn.compute(predictions=predictions, references=references, lang=self.lang)
        # average scores for precision, recall, f1
        scores["precision"] = sum(scores["precision"]) / len(scores["precision"])
        scores["recall"] = sum(scores["recall"]) / len(scores["recall"])
        scores["f1"] = sum(scores["f1"]) / len(scores["f1"])
        return scores

class BleuEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("bleu", eval_load("bleu"))

    def evaluate(self, predictions, references):
        return self.metric_fn.compute(predictions=predictions, references=references)

    
class MeteorEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("meteor", eval_load("meteor"))

    def evaluate(self, predictions, references):
        return self.metric_fn.compute(predictions=predictions, references=references)
    

def get_all_evaluators() -> List[BaseEvaluator]:
    evaluators = [
        RougeRawEvaluator(),
        RougeEvaluator(),
        BertScoreEvaluator(),
        BleuEvaluator(),
        MeteorEvaluator()
    ]
    return evaluators