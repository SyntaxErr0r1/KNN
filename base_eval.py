import json
from typing import List, Dict
import glob
from torch.utils.data import Dataset, DataLoader
from evaluate_metrics import BaseEvaluator, get_all_evaluators

class SummaryDatasetSumeCzech(Dataset):
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to the data file
        """
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   

def benchmark_model(evaluators: List[BaseEvaluator], references: List[str], predictions: List[str]):

    scores = {}
    for evaluator in evaluators:
        metric_name = evaluator.get_metric_name()
        scores[metric_name] = evaluator.evaluate(predictions, references)

    return scores

if __name__ == "__main__":
    
    sumeczech_0shot_abstract_to_headline = json.load(open("data/sumeczech/eval_150samples_test_mistral7b_v0.2/sumeczech_0shot_abstract-to-headline.json"))
    sumeczech_3shot_abstract_to_headline = json.load(open("data/sumeczech/eval_150samples_test_mistral7b_v0.2/sumeczech_3shot_abstract-to-headline.json"))
    sumeczech_3shot_text_to_abstract = json.load(open("data/sumeczech/eval_150samples_test_mistral7b_v0.2/sumeczech_3shot_text-to-abstract.json"))

    evaluators = get_all_evaluators()

    abstract_to_headline_scores_0shot = benchmark_model(evaluators, sumeczech_0shot_abstract_to_headline["references"], sumeczech_0shot_abstract_to_headline["predictions"])
    abstract_to_headline_scores_3shot = benchmark_model(evaluators, sumeczech_3shot_abstract_to_headline["references"], sumeczech_3shot_abstract_to_headline["predictions"])
    text_to_abstract_scores_3shot = benchmark_model(evaluators, sumeczech_3shot_text_to_abstract["references"], sumeczech_3shot_text_to_abstract["predictions"])

    for metric_name, score in abstract_to_headline_scores_0shot.items():
        print(f"{metric_name}: {json.dumps(score, indent=4)}")

    for metric_name, score in abstract_to_headline_scores_3shot.items():
        print(f"{metric_name}: {json.dumps(score, indent=4)}")

    for metric_name, score in text_to_abstract_scores_3shot.items():
        print(f"{metric_name}: {json.dumps(score, indent=4)}")