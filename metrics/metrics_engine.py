import os
import logging
import warnings

# Suppress transformers/RoBERTa loading warnings (e.g. "Some weights... not initialized", "You should probably TRAIN")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
for _name in ("transformers", "transformers.modeling_utils"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path

import evaluate
from bert_score import score as bertscore_score


# Load HF metrics once
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
semf1_metric = evaluate.load("nbansal/semf1")


# -------------------------------------------------
# helpers
# -------------------------------------------------

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


# -------------------------------------------------
# BERTScore
# -------------------------------------------------

def compute_bertscore(preds, refs, device):
    P, R, F1 = bertscore_score(
        preds,
        refs,
        lang="en",
        device=device,
        batch_size=64,
        verbose=False
    )
    return F1.tolist()


# -------------------------------------------------
# ROUGE / METEOR / ChrF
# -------------------------------------------------

# def compute_overlap_metrics(preds, refs):
#     rouge_scores = rouge.compute(predictions=preds, references=refs)
#     meteor_scores = meteor.compute(predictions=preds, references=refs)
#     chrf_scores = chrf.compute(predictions=preds, references=refs)

#     return {
#         "rouge1": [rouge_scores["rouge1"]] * len(preds),
#         "rougeL": [rouge_scores["rougeL"]] * len(preds),
#         "meteor": [meteor_scores["meteor"]] * len(preds),
#         "chrf": [chrf_scores["score"]/100] * len(preds),
#     }


def compute_overlap_metrics(preds, refs):
    rouge1, rougeL, meteor_list, chrf_list = [], [], [], []

    for p, r in zip(preds, refs):
        r_score = rouge.compute(predictions=[p], references=[r])
        m_score = meteor.compute(predictions=[p], references=[r])
        c_score = chrf.compute(predictions=[p], references=[r])

        rouge1.append(r_score["rouge1"])
        rougeL.append(r_score["rougeL"])
        meteor_list.append(m_score["meteor"])
        chrf_list.append(c_score["score"] / 100)

    return {
        "rouge1": rouge1,
        "rougeL": rougeL,
        "meteor": meteor_list,
        "chrf": chrf_list,
    }



# -------------------------------------------------
# REAL SEM-F1
# -------------------------------------------------

def compute_semf1(preds, refs, device, batch_size):
    """
    Returns per-sample precision, recall, f1
    """

    results = semf1_metric.compute(
        predictions=preds,
        references=refs,
        model_type="pv1",      # paraphrase-distilroberta-base-v1
        gpu=True if device=="cuda" else False,
        batch_size=batch_size,
        aggregate=False
    )

    prec = [s.precision for s in results]
    rec  = [s.recall[0] for s in results]
    f1   = [(2*p*r)/(p+r+1e-8) for p,r in zip(prec,rec)]

    return prec, rec, f1


# -------------------------------------------------
# MASTER EVALUATOR
# -------------------------------------------------

class Evaluator:

    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
        predictions = [safe_text(x) for x in predictions]
        references = [safe_text(x) for x in references]

        results = {
            "bert_f1": [],
            "rouge1": [],
            "rougeL": [],
            "meteor": [],
            "chrf": [],
            "sem_precision": [],
            "sem_recall": [],
            "sem_f1": []
        }

        for pred_batch, ref_batch in tqdm(
            zip(chunk_list(predictions, self.batch_size),
                chunk_list(references, self.batch_size)),
            total=len(predictions)//self.batch_size + 1,
            desc="Evaluating batches"
        ):

            # 1️⃣ BERTScore
            bert = compute_bertscore(pred_batch, ref_batch, self.device)
            results["bert_f1"].extend(bert)

            # 2️⃣ SEM-F1 (official)
            sem_p, sem_r, sem_f = compute_semf1(
                pred_batch, ref_batch, self.device, self.batch_size
            )
            results["sem_precision"].extend(sem_p)
            results["sem_recall"].extend(sem_r)
            results["sem_f1"].extend(sem_f)

            # 3️⃣ ROUGE / METEOR / ChrF
            overlap = compute_overlap_metrics(pred_batch, ref_batch)
            for k in overlap:
                results[k].extend(overlap[k])

        return results
