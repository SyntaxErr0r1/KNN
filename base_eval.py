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

def benchmark_model_sumeczech(model, data_dir: str, evaluators: List[BaseEvaluator], task: str, batch_size: int = 1):
    test_data_paths = glob.glob(f"{data_dir}/*-test.jsonl")
    dataset = SummaryDatasetSumeCzech(test_data_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    scores = {}
    for evaluator in evaluators:
        metric_name = evaluator.get_metric_name()
        if task == "text_to_abstract":
            references = [item["abstract"] for item in dataset.data[:1]] # TODO: using one example
        elif task == "text_to_headline":
            references = [item["headline"] for item in dataset.data]
        else:
            raise ValueError("Invalid task specified. Choose either 'text_to_abstract' or 'text_to_headline'")

        predictions = []
        for batch in data_loader:
            batch_predictions = ["Výběr podlahy do interiéru je rozhodující, neboť ovlivňuje celkový vzhled místnosti a je klíčovým prvkem pro harmonii designu. Možnosti jsou rozmanité, od umělých materiálů jako je PVC a laminát po přírodní jako je dřevo, kámen nebo linoleum. Současný trend směřuje k přírodním materiálům, přičemž linoleum, známé též pod obchodním názvem marmoleum, nabízí nejen estetickou hodnotu, ale i praktickou odolnost a snadnou údržbu."]  
             # model.generate_summaries(batch, task)
            predictions.extend(batch_predictions)
            break # TODO: Remove this break statement, used for demonstration purposes
        print(predictions, references)
        
        scores[metric_name] = evaluator.evaluate(predictions, references)

    return scores

if __name__ == "__main__":
    model = ... # TODO: 
    evaluators = get_all_evaluators()

    # Benchmark for text to abstract
    text_to_abstract_scores = benchmark_model_sumeczech(model, "./data/sumeczech/", evaluators, "text_to_abstract")
    print("Text to Abstract Scores:")
    for metric_name, score in text_to_abstract_scores.items():
        print(f"{metric_name}: {json.dumps(score, indent=2)}")