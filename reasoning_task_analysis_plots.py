import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Any
plt.rcParams['svg.fonttype'] = 'none'

# Generator lists
generator_list = [
    "o3",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash-001",
    "gemini-2.0-flash-001",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "exaone3.5:32b",
    "exaone3.5:7.8b",
    "EXAONE-3.0-7.8B-Instruct",
]

# Evaluator lists
evaluator_list = [
    "human_group1", "human_group2",
    # "o3",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "exaone3.5:7.8b",
    "exaone3.5:2.4b",
    # "EXAONE-3.0-7.8B-Instruct",
]

mini_generator_list = [ # selected models for deep analysis
    "gpt-4.1",
    "gemini-2.5-flash-001",
    "gemma-3-12b-it",
    "exaone3.5:7.8b",
]

cmap_custom_blue = LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", "#caedec", "#8ed7d7", "#00afbd", "#00383c"], N=256)

def cohens_kappa(ann1: List[Any], ann2: List[Any], labels: List[Any]) -> float:
    """
    Compute Cohen's Kappa between two annotators.
    
    Parameters:
        ann1   : list of annotations from annotator 1
        ann2   : list of annotations from annotator 2
        labels : list of possible labels
        
    Returns:
        kappa  : float, Cohen's kappa score
    """
    if len(ann1) != len(ann2):
        raise ValueError("Annotation lists must have the same length.")
    
    n = len(ann1)
    if n == 0:
        raise ValueError("Annotation lists must not be empty.")
    
    # Build confusion matrix
    matrix = {l1: {l2: 0 for l2 in labels} for l1 in labels}
    for a1, a2 in zip(ann1, ann2):
        matrix[a1][a2] += 1
    
    # Observed agreement
    po = sum(matrix[l][l] for l in labels) / n
    
    # Expected agreement
    ann1_counts = Counter(ann1)
    ann2_counts = Counter(ann2)
    pe = sum((ann1_counts[l] / n) * (ann2_counts[l] / n) for l in labels)
    
    # Cohen's kappa
    if pe == 1.0:
        return 1.0  # avoid division by zero
    kappa = (po - pe) / (1 - pe)
    
    return kappa


# Dict storing:
# evaluator -> generator -> docid -> List[evaluations] (evaluations: Dict[str, Dict[str, bool]])
evaluation_dict = {
    evaluator: {
        generator: {}
        for generator in generator_list
    }
    for evaluator in evaluator_list
}
# Difficulty dict
difficulty_dict = {}
with open("data/reasoning_tasks_test.jsonl", "r") as f:
    for line in f:
        datum = json.loads(line.strip())
        doc_id = datum["doc_id"]
        difficulty = datum["difficulty"]
        difficulty_dict[doc_id] = difficulty

# Load all files
for evaluator in evaluator_list:
    if "human" in evaluator:
        continue
    for generator in generator_list:
        # Load the JSON file for each evaluator and generator
        file_path = f"results/reasoning_tasks_{generator}_evaluator_{evaluator}.jsonl"
        try:
            with open(file_path, "r") as f:
                for line in f:
                    datum = json.loads(line.strip())
                    doc_id = datum["doc_id"]
                    evaluation_dict[evaluator][generator][doc_id] = datum
        except FileNotFoundError:
            # print(f"File not found: {file_path}")
            pass
for evaluator in ["human_group1", "human_group2"]:
    with open(f"results/{evaluator}.json", "r") as f:
        data = json.load(f)
        for model, datum in data.items():
            if model not in evaluation_dict[evaluator]:
                evaluation_dict[evaluator][model] = {}
            for doc_id, issues in datum.items():
                evaluation_dict[evaluator][model][doc_id] = issues

# Compute N*N label agreement / Spearman's correlation matrix
total = np.zeros((len(evaluator_list), len(evaluator_list)))
agreed = np.zeros((len(evaluator_list), len(evaluator_list))) # Cohen's kappa
total_root = np.zeros((len(evaluator_list), len(evaluator_list)))
agreed_root = np.zeros((len(evaluator_list), len(evaluator_list))) # Cohen's kappa
agreed_root_strict = np.zeros((len(evaluator_list), len(evaluator_list))) # Cohen's kappa
pearson = np.zeros((len(evaluator_list), len(evaluator_list)))
pearson_without_hierarchy = np.zeros((len(evaluator_list), len(evaluator_list)))
score_individual_scatter_plots = [("human_group1", "human_group2"), ("human_group1", "gemini-2.0-flash-001"), ("human_group1", "gemini-2.5-flash")]

# Top eschelon form
for i, evaluator_i in enumerate(evaluator_list):
    for j, evaluator_j in enumerate(evaluator_list[i:], start=i):
        # print(i, j)
        scores_i = [] # score (0-10)
        scores_j = []
        scores_without_hierarchy_i = []
        scores_without_hierarchy_j = []
        all_results_i = [] # categorical - Bool * Bool
        all_results_j = []
        root_results_i = [] # categorical (only root) - Bool
        root_results_j = []
        root_results_strict_i = []
        root_results_strict_j = []
        for k, generator in enumerate(generator_list):
            evaluator_i_docids = set(evaluation_dict[evaluator_i][generator].keys())
            evaluator_j_docids = set(evaluation_dict[evaluator_j][generator].keys())
            docids = evaluator_i_docids.intersection(evaluator_j_docids)
            for doc_id in docids:
                doc_results_i, doc_results_j = [], []
                scores_i.append(evaluation_dict[evaluator_i][generator][doc_id]["score"])
                scores_j.append(evaluation_dict[evaluator_j][generator][doc_id]["score"])

                for issue_id, result_i in evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"].items():
                    result_j = evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"].get(issue_id)
                    if result_j is None:
                        continue
                    # Remove non-standard labels for Kappa calculation
                    result_i["contains_issue"] = result_i["contains_issue"] == True
                    result_j["contains_issue"] = result_j["contains_issue"] == True
                    result_i["correct_conclusion"] = result_i["correct_conclusion"] == True
                    result_j["correct_conclusion"] = result_j["correct_conclusion"] == True
                    # print(doc_id, issue_id, result_i is None, result_j is None)
                    total[i, j] += 1
                    doc_results_i.append((result_i["contains_issue"], result_i["correct_conclusion"]))
                    doc_results_j.append((result_j["contains_issue"], result_j["correct_conclusion"]))
                    
                # scores without hierarchy
                scores_without_hierarchy_i_doc = 0
                scores_without_hierarchy_j_doc = 0
                for issue_id, result_i in evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"].items():
                    result_j = evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"].get(issue_id)
                    if result_j is None:
                        continue
                    if issue_id == "":
                        total_root[i, j] += 1
                        root_results_i.append(result_i["correct_conclusion"])
                        root_results_j.append(result_j["correct_conclusion"])

                        # Strict case (no (True, False) pairs)
                        all_subissues_correct_i = all([x != (True, False) for x in doc_results_i])
                        all_subissues_correct_j = all([x != (True, False) for x in doc_results_j])
                        root_results_strict_i.append(result_i["correct_conclusion"] and all_subissues_correct_i)
                        root_results_strict_j.append(result_j["correct_conclusion"] and all_subissues_correct_j)

                        scores_without_hierarchy_i_doc += 5 if result_i["correct_conclusion"] else 0
                        scores_without_hierarchy_j_doc += 5 if result_j["correct_conclusion"] else 0
                    else:
                        if len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) <= 1:
                            pass
                        elif result_i["contains_issue"] and result_i["correct_conclusion"]:
                            scores_without_hierarchy_i_doc += 5 / (len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) - 1)
                        elif result_i["contains_issue"]:
                            scores_without_hierarchy_i_doc += 2 / (len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) - 1)
                        if len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) <= 1:
                            pass
                        elif result_j["contains_issue"] and result_j["correct_conclusion"]:
                            scores_without_hierarchy_j_doc += 5 / (len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) - 1)
                        elif result_j["contains_issue"]:
                            scores_without_hierarchy_j_doc += 2 / (len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) - 1)
                all_results_i.extend(doc_results_i)
                all_results_j.extend(doc_results_j)
                scores_without_hierarchy_i.append(scores_without_hierarchy_i_doc)
                scores_without_hierarchy_j.append(scores_without_hierarchy_j_doc)

        agreed[i, j] = cohens_kappa(all_results_i, all_results_j, [(False, False), (False, True), (True, False), (True, True)])
        agreed_root[i, j] = cohens_kappa(root_results_i, root_results_j, [False, True])
        agreed_root_strict[i, j] = cohens_kappa(root_results_strict_i, root_results_strict_j, [False, True])
        pearson[i, j] = np.corrcoef(scores_i, scores_j)[0, 1] if scores_i and scores_j else np.nan
        pearson_without_hierarchy[i, j] = np.corrcoef(scores_without_hierarchy_i, scores_without_hierarchy_j)[0, 1] if scores_without_hierarchy_i and scores_without_hierarchy_j else np.nan

        if (evaluator_list[i], evaluator_list[j]) in score_individual_scatter_plots:
            plt.figure(figsize=(8, 6))
            plt.scatter(scores_i, scores_j)
            plt.xlabel(evaluator_list[i])
            plt.ylabel(evaluator_list[j])
            plt.title(f"Scatter plot: {evaluator_list[i]} vs {evaluator_list[j]}")
            plt.savefig(f"plots/legit_score_scatter_{evaluator_list[i]}_{evaluator_list[j]}.svg")
            plt.close()

# draw a colored grid based on agreed / total. Treat NaN as gray
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(agreed), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for j in range(len(evaluator_list)):
        if total[i, j] > 0:
            plt.text(j, i, f"{agreed[i, j]:.2f}", ha="center", va="center", color="white")
plt.title("Cohen's Kappa between Evaluators")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/legit_issue_label_agreement.svg")
plt.close()

# root nodes agreement
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(agreed_root), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for k in range(len(evaluator_list)):
        if total_root[i, k] > 0:
            plt.text(k, i, f"{agreed_root[i, k]:.2f}", ha="center", va="center", color="white")
plt.title("Cohen's Kappa between Evaluators (Final order only)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/final_order_agreement.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(agreed_root_strict), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for k in range(len(evaluator_list)):
        if total_root[i, k] > 0:
            plt.text(k, i, f"{agreed_root_strict[i, k]:.2f}", ha="center", va="center", color="white")
plt.title("Pearson's r on LEGIT score against human evaluators")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/final_order_agreement_strict.svg")
plt.close()


# Plot bar graph for final agreement
plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
# take avereage accuracy each mode <>human1 and <>human2
human_avg = (agreed_root[0, 2:] + agreed_root[1, 2:]) / 2
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#8ed7d7")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Pearson's r")
# dotted line at human-human agreement
human_human_agreement = agreed_root[0, 1]
plt.axhline(y=human_human_agreement, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_agreement + 0.02, f"Human-Human: {human_human_agreement:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Cohen's Kappa on final order against human evaluators")
plt.savefig("plots/PAPER_final_order_acc_cohenkappa.svg")
plt.close()


# Plot bar graph.
plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
# Fisher z transformation between each mode <>human1 and <>human2
human1_z = np.arctanh(pearson[0, 2:])
human2_z = np.arctanh(pearson[1, 2:])
human_avg_z = (human1_z + human2_z) / 2
human_avg = np.tanh(human_avg_z)
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#00afbd")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Pearson's r")
# dotted line at human-human agreement
human_human_pearson = pearson[0, 1]
plt.axhline(y=human_human_pearson, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_pearson + 0.02, f"Human-Human: {human_human_pearson:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Average Pearson's r against Human Evaluators")
plt.savefig("plots/PAPER_legit_score_pearson.svg")
plt.close()


# Pearson
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(pearson), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for j in range(len(evaluator_list)):
        if not np.isnan(pearson[i, j]):
            plt.text(j, i, f"{pearson[i, j]:.2f}", ha="center", va="center", color="white")
plt.title("Pearson Correlation between Evaluators (LEGIT score)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/legit_pearson.svg")
plt.close()

# Pearson without hierarchy
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(pearson_without_hierarchy), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for j in range(len(evaluator_list)):
        if not np.isnan(pearson_without_hierarchy[i, j]):
            plt.text(j, i, f"{pearson_without_hierarchy[i, j]:.2f}", ha="center", va="center", color="white")
plt.title("Pearson Correlation between Evaluators (LEGIT score)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/legit_pearson_without_hierarchy.svg")
plt.close()

################################################################################


datapoints = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_score_total = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_score_easy = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_score_medium = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_score_hard = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_root_correct_total = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_root_correct_easy = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_root_correct_medium = np.zeros((len(generator_list), len(evaluator_list)-2))
gen_eval_root_correct_hard = np.zeros((len(generator_list), len(evaluator_list)-2))
for i, evaluator in enumerate(evaluator_list[2:]):
    for j, generator in enumerate(generator_list):
        scores_i = {"easy": [], "medium": [], "hard": []}
        root_correct = {"easy": 0, "medium": 0, "hard": 0}
        for doc_id in evaluation_dict[evaluator][generator].keys():
            scores_i[difficulty_dict[doc_id]].append(evaluation_dict[evaluator][generator][doc_id]["score"])
            if "" in evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]:
                root_correct[difficulty_dict[doc_id]] += 1 if evaluation_dict[evaluator][generator][doc_id]["evaluation_result"][""]["correct_conclusion"] else 0
        easy_score = sum(scores_i["easy"]) / len(scores_i["easy"]) if scores_i["easy"] else 0
        medium_score = sum(scores_i["medium"]) / len(scores_i["medium"]) if scores_i["medium"] else 0
        hard_score = sum(scores_i["hard"]) / len(scores_i["hard"]) if scores_i["hard"] else 0
        gen_eval_score_total[j, i] = (easy_score + medium_score + hard_score) / 3
        gen_eval_score_easy[j, i] = easy_score
        gen_eval_score_medium[j, i] = medium_score
        gen_eval_score_hard[j, i] = hard_score
        datapoints[j, i] = len(scores_i["easy"]) + len(scores_i["medium"]) + len(scores_i["hard"])

        gen_eval_root_correct_easy[j, i] = root_correct["easy"] / len(scores_i["easy"]) if len(scores_i["easy"]) > 0 else 0
        gen_eval_root_correct_medium[j, i] = root_correct["medium"] / len(scores_i["medium"]) if len(scores_i["medium"]) > 0 else 0
        gen_eval_root_correct_hard[j, i] = root_correct["hard"] / len(scores_i["hard"]) if len(scores_i["hard"]) > 0 else 0
        gen_eval_root_correct_total[j, i] = (gen_eval_root_correct_easy[j, i] + gen_eval_root_correct_medium[j, i] + gen_eval_root_correct_hard[j, i]) / 3


# Plot datapoints (for monitoring purposes)
plt.figure(figsize=(8, 6))
plt.imshow(datapoints, cmap="Blues", vmin=0, vmax=300)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
for i in range(len(generator_list)):
    for j in range(len(evaluator_list[2:])):
        if datapoints[i, j] > 0:
            plt.text(j, i, f"{datapoints[i, j]:.0f}", ha="center", va="center", color="white")
plt.title("Number of Data Points per (Generator, Evaluator) Pair")
plt.savefig("plots/legit_evaluation_count.svg")
plt.close()

# Gen-Eval plot
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_score_total), cmap="Blues", vmin=0, vmax=10)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_score_total[k, i]):
            plt.text(i, k, f"{gen_eval_score_total[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Score (total)")
plt.savefig("plots/legit_score_total.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_score_easy), cmap="Blues", vmin=0, vmax=10)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_score_easy[k, i]):
            plt.text(i, k, f"{gen_eval_score_easy[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Score (easy)")
plt.savefig("plots/legit_score_easy.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_score_medium), cmap="Blues", vmin=0, vmax=10)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_score_medium[k, i]):
            plt.text(i, k, f"{gen_eval_score_medium[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Score (medium)")
plt.savefig("plots/legit_score_medium.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_score_hard), cmap="Blues", vmin=0, vmax=10)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_score_hard[k, i]):
            plt.text(i, k, f"{gen_eval_score_hard[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Score (hard)")
plt.savefig("plots/legit_score_hard.svg")
plt.close()

# Root correct figure
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_root_correct_total), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_root_correct_total[k, i]):
            plt.text(i, k, f"{gen_eval_root_correct_total[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Score (hard)")
plt.savefig("plots/root_correct_total.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_root_correct_easy), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_root_correct_easy[k, i]):
            plt.text(i, k, f"{gen_eval_root_correct_easy[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Root accuracy (easy)")
plt.savefig("plots/root_correct_easy.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_root_correct_medium), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_root_correct_medium[k, i]):
            plt.text(i, k, f"{gen_eval_root_correct_medium[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Root accuracy (medium)")
plt.savefig("plots/root_correct_medium.svg")
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(gen_eval_root_correct_hard), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)-2):
    for k in range(len(generator_list)):
        if not np.isnan(gen_eval_root_correct_hard[k, i]):
            plt.text(i, k, f"{gen_eval_root_correct_hard[k, i]:.2f}", ha="center", va="center", color="white")
plt.title("Root accuracy (hard)")
plt.savefig("plots/root_correct_hard.svg")
plt.close()


##################################################################################

# Plot Issue Depth vs. Metric chart
# Use Gemini-2.0-Flash evaluation results
generator_issue_total = {} # root, 1, 2, 3+
generator_issue_coverage_per_depth = {}
generator_issue_coverage_and_correctness_per_depth = {}
generator_issue_leaf_nonleaf_total = {}
generator_issue_leaf_nonleaf_coverage = {}
generator_issue_leaf_nonleaf_coverage_and_correctness = {}
for j, generator in enumerate(generator_list):
    generator_issue_total[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    generator_issue_coverage_per_depth[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    generator_issue_coverage_and_correctness_per_depth[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    generator_issue_leaf_nonleaf_total[generator] = {"leaf": 0, "nonleaf": 0}
    generator_issue_leaf_nonleaf_coverage[generator] = {"leaf": 0, "nonleaf": 0}
    generator_issue_leaf_nonleaf_coverage_and_correctness[generator] = {"leaf": 0, "nonleaf": 0}
    for doc_id in evaluation_dict["gemini-2.0-flash-001"][generator].keys():
        issue_id_set = list(evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"].keys())
        for issue_id, result in evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"].items():
            # Check leaf
            leaf = "leaf"
            for issue_id2 in issue_id_set:
                if issue_id2.startswith(issue_id) and issue_id2 != issue_id:
                    leaf = "nonleaf"
                    break

            if issue_id == "":
                depth = 0
            else:
                depth = issue_id.count(".") + 1
            if depth >= 3:
                depth = 3
            generator_issue_total[generator][depth] += 1
            generator_issue_leaf_nonleaf_total[generator][leaf] += 1
            if result["contains_issue"]:
                generator_issue_coverage_per_depth[generator][depth] += 1
                generator_issue_leaf_nonleaf_coverage[generator][leaf] += 1
            if result["contains_issue"] and result["correct_conclusion"]:
                generator_issue_coverage_and_correctness_per_depth[generator][depth] += 1
                generator_issue_leaf_nonleaf_coverage_and_correctness[generator][leaf] += 1

# Draw overlapping line chart
depth_labels = ["0 (Final order)", "1", "2", "3+"] #, "4+"]
plt.figure(figsize=(8, 6))
for generator in mini_generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_per_depth[generator][i] / generator_issue_total[generator][i] if generator_issue_total[generator][i] > 0 else 0 for i in range(4)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Coverage")
plt.title("Issue Coverage by Depth")
plt.legend()
plt.savefig("plots/issues_coverage_by_depth.svg")

plt.figure(figsize=(8, 6))
for generator in mini_generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_and_correctness_per_depth[generator][i] / generator_issue_total[generator][i] if generator_issue_total[generator][i] > 0 else 0 for i in range(4)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Coverage and Correctness")
plt.title("Issue Coverage and Correctness by Depth")
plt.legend()
plt.savefig("plots/issues_coverage_and_correctness_by_depth.svg")

plt.figure(figsize=(8, 6))
for generator in mini_generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_and_correctness_per_depth[generator][i] / generator_issue_coverage_per_depth[generator][i] if generator_issue_coverage_per_depth[generator][i] > 0 else 0 for i in range(4)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Correctness, only when issue is covered")
plt.title("Conditional Issue Correctness by Depth")
plt.legend()
plt.savefig("plots/issues_coverage_and_correctness_by_depth_conditional.svg")

# Merge two plots above, with coverage and correctness in one plot
plt.figure(figsize=(8, 5))
plt.title("Issue Completeness and Correctness by Depth")
plt.xlabel("Depth")
plt.ylabel("Issue completeness")
style_map = {g: (line, marker) for g, line, marker in zip(mini_generator_list, ["-", "--", "-.", ":"], ["o", "s", "D", "^"])}
for generator in mini_generator_list:
    line, marker = style_map[generator]
    plt.plot(depth_labels, [generator_issue_coverage_per_depth[generator][i] / generator_issue_total[generator][i] if generator_issue_total[generator][i] > 0 else 0 for i in range(4)], marker=marker, linestyle=line, label=f"{generator} Coverage", color="#ff8ca1")
plt.ylim(0.2, 1)
plt.twinx()
plt.ylabel("Correctness (for covered issues)")
for generator in mini_generator_list:
    line, marker = style_map[generator]
    plt.plot(depth_labels, [generator_issue_coverage_and_correctness_per_depth[generator][i] / generator_issue_coverage_per_depth[generator][i] if generator_issue_coverage_per_depth[generator][i] > 0 else 0 for i in range(4)], marker=marker, linestyle=line, label=f"{generator} Correctness", color="#feb253")
plt.ylim(0.2, 1)
plt.legend(loc='lower left')
plt.savefig("plots/PAPER_issues_completeness_and_correctness_by_depth_combined.svg", bbox_inches='tight')

# leaf vs. nonleaf coverage
plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(len(mini_generator_list))
leaf_coverage = [generator_issue_leaf_nonleaf_coverage[gen]["leaf"] / generator_issue_leaf_nonleaf_total[gen]["leaf"] if generator_issue_leaf_nonleaf_total[gen]["leaf"] > 0 else 0 for gen in mini_generator_list]
nonleaf_coverage = [generator_issue_leaf_nonleaf_coverage[gen]["nonleaf"] / generator_issue_leaf_nonleaf_total[gen]["nonleaf"] if generator_issue_leaf_nonleaf_total[gen]["nonleaf"] > 0 else 0 for gen in mini_generator_list]
plt.bar(x - bar_width/2, nonleaf_coverage, width=bar_width, label="Non-Leaf Issues", color="#ffc5d0")
plt.bar(x + bar_width/2, leaf_coverage, width=bar_width, label="Leaf Issues", color="#ff8ca1")
plt.xticks(ticks=x, labels=mini_generator_list, rotation=45)
plt.ylim(0, 1)
for i in range(len(mini_generator_list)):
    plt.text(i - bar_width/2, nonleaf_coverage[i] + 0.02, f"{nonleaf_coverage[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i + bar_width/2, leaf_coverage[i] + 0.02, f"{leaf_coverage[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Coverage")
plt.title("Leaf vs Non-Leaf Issue Coverage")
plt.legend(loc="lower left")
plt.savefig("plots/issues_coverage_leaf_vs_nonleaf_coverage.svg")

# leaf vs. nonleaf conditional correctness
plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(len(mini_generator_list))
leaf_correctness = [generator_issue_leaf_nonleaf_coverage_and_correctness[gen]["leaf"] / generator_issue_leaf_nonleaf_coverage[gen]["leaf"] if generator_issue_leaf_nonleaf_coverage[gen]["leaf"] > 0 else 0 for gen in mini_generator_list]
nonleaf_correctness = [generator_issue_leaf_nonleaf_coverage_and_correctness[gen]["nonleaf"] / generator_issue_leaf_nonleaf_coverage[gen]["nonleaf"] if generator_issue_leaf_nonleaf_coverage[gen]["nonleaf"] > 0 else 0 for gen in mini_generator_list]
plt.bar(x - bar_width/2, nonleaf_correctness, width=bar_width, label="Non-Leaf Issues", color="#ffe5c5")
plt.bar(x + bar_width/2, leaf_correctness, width=bar_width, label="Leaf Issues", color="#ffcd8e")
plt.xticks(ticks=x, labels=mini_generator_list, rotation=45)
plt.ylim(0, 1)
for i in range(len(mini_generator_list)):
    plt.text(i - bar_width/2, nonleaf_correctness[i] + 0.02, f"{nonleaf_correctness[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i + bar_width/2, leaf_correctness[i] + 0.02, f"{leaf_correctness[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Correctness (only when issue is covered)")
plt.title("Leaf vs Non-Leaf Issue Correctness")
# place legend left bottom
plt.legend(loc="lower left")
plt.savefig("plots/issues_coverage_leaf_vs_nonleaf_conditional_correctness.svg")

##################################################################################

# Plot root correctness, issue completeness, issue correctness.

plt.figure(figsize=(8, 5))
plt.ylim(0,10)
generator_root_scores = []
generator_issue_completeness = []
generator_issue_correctness = []
for generator in generator_list:
    evaluator = "gemini-2.0-flash-001"
    root_scores = []
    issue_completeness = []
    issue_correctness = []
    for doc_id in evaluation_dict[evaluator][generator].keys():
        num_subissues = len(evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]) - 1
        completeness = 0
        correctness = 0
        if num_subissues <= 0:
            continue
        for issue_id, issue in evaluation_dict[evaluator][generator][doc_id]["evaluation_result"].items():
            if issue_id == "":
                root_scores.append(5 if issue["correct_conclusion"] else 0)
            else:
                if issue["contains_issue"]:
                    completeness += 2/num_subissues
                if issue["contains_issue"] and issue["correct_conclusion"]:
                    correctness += 3/num_subissues
        issue_completeness.append(completeness)
        issue_correctness.append(correctness)

    root_scores = sum(root_scores) / len(root_scores) if root_scores else 0
    issue_completeness = sum(issue_completeness) / len(issue_completeness) if issue_completeness else 0
    issue_correctness = sum(issue_correctness) / len(issue_correctness) if issue_correctness else 0
    generator_root_scores.append(root_scores)
    generator_issue_completeness.append(issue_completeness)
    generator_issue_correctness.append(issue_correctness)
# Stacked bar chart: add reverse to match legend order
plt.bar(generator_list, generator_issue_correctness, bottom=[x+y for x, y in zip(generator_root_scores, generator_issue_completeness)], label="Issue Correctness", color="#ff8ca1")
plt.bar(generator_list, generator_issue_completeness, bottom=generator_root_scores, label="Issue Completeness", color="#feb253")
plt.bar(generator_list, generator_root_scores, bottom=0, label="Final order Correctness", color="#98c126")
for i in range(len(generator_list)):
    print(generator_list[i], generator_root_scores[i], generator_issue_completeness[i], generator_issue_correctness[i], generator_root_scores[i] + generator_issue_completeness[i] + generator_issue_correctness[i])
    total_height = generator_root_scores[i] + generator_issue_completeness[i] + generator_issue_correctness[i]
    plt.text(i, total_height + 0.2, f"{total_height:.2f}", ha="center", va="bottom", color="black")
plt.legend(loc="upper right")
plt.savefig("plots/PAPER_LEGIT_scores.svg")
plt.close()

################################################################################

# Dataset statistics: number of issues distribution
issue_counts = []
with open("data/reasoning_tasks_train.jsonl", "r") as f:
    for line in f:
        datum = json.loads(line.strip())
        length = len(datum["issues"])
        issue_counts.append(length if length < 26 else 26)  # cap at 25
with open("data/reasoning_tasks_test.jsonl", "r") as f:
    for line in f:
        datum = json.loads(line.strip())
        length = len(datum["issues"])
        issue_counts.append(length if length < 26 else 26)  # cap at 25
plt.figure(figsize=(8, 5))
# relative frequency histogram
plt.hist(issue_counts, bins=range(0, 27), density=True, color="#feb253", edgecolor="#7f7f7f")
# Change colors: for issue_count <= 4, 4< issue_count <=8, issue_count > 8
for i in range(len(plt.gca().patches)):
    if plt.gca().patches[i].get_x() < 4:
        plt.gca().patches[i].set_color("#ffe5c5")
    elif plt.gca().patches[i].get_x() < 8:
        plt.gca().patches[i].set_color("#ffcd8e")
    else:
        plt.gca().patches[i].set_color("#feb253")
# Change last xtick to "25+"
plt.xticks(ticks=[0,5,10,15,20,25,26])
plt.gca().set_xticklabels([str(i) if i <= 25 else "25+" for i in [0,5,10,15,20,25,26]])
plt.xlabel("Number of Issues")
plt.ylabel("Relative Frequency")
plt.title("Distribution of Number of Issues per Instance")
plt.savefig("plots/PAPER_issue_count_distribution.svg")
plt.close()

# Dataset statistics: response length distribution of selected generators for difficulty
# response_lengths_by_difficulty = {
#     "easy": [],
#     "medium": [],
#     "hard": []
# }
# for generator in ["o3"]:
#     with open(f"results/reasoning_tasks_{generator}.jsonl", "r") as f:
#         for line in f:
#             datum = json.loads(line.strip())
#             response_length = len(datum["response"].split())
#             difficulty = difficulty_dict[datum["doc_id"]]

#             if response_length > 10:
#                 response_lengths_by_difficulty[difficulty].append(response_length)
# # Boxplot
# plt.figure(figsize=(8, 5))
# plt.boxplot([response_lengths_by_difficulty["easy"], response_lengths_by_difficulty["medium"], response_lengths_by_difficulty["hard"]], labels=["Easy", "Medium", "Hard"], patch_artist=True, boxprops=dict(facecolor="#ffcd8e", color="#7f7f7f7f"), medianprops=dict(color="black"))
# plt.ylabel("Response Length (in words)")
# plt.title("Response Length Distribution by Difficulty")
# plt.savefig("plots/PAPER_response_length_by_difficulty.svg")
# plt.close()

# Effects of children node results
no_children_identified = {x: [] for x in generator_list}
some_children_identified_are_wrong = {x: [] for x in generator_list}
all_children_identified_are_correct = {x: [] for x in generator_list}
for generator in generator_list:
    evaluator = "gemini-2.0-flash-001"
    for doc_id in evaluation_dict[evaluator][generator].keys():
        issue_dict = evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]
        for issue_id, issue in issue_dict.items():
            children_results = []
            for issue_id2, issue2 in issue_dict.items():
                if issue_id2.startswith(issue_id) and issue_id2 != issue_id:
                    children_results.append((issue2["contains_issue"], issue2["correct_conclusion"]))
            if len(children_results) == 0:
                # No children -> pass
                continue
            if issue["contains_issue"]:
                if all([not res[0] for res in children_results]):
                    no_children_identified[generator].append(int(issue["correct_conclusion"]))
                elif any([res[0] and not res[1] for res in children_results]):
                    some_children_identified_are_wrong[generator].append(int(issue["correct_conclusion"]))
                elif all([res[0] and res[1] for res in children_results]):
                    all_children_identified_are_correct[generator].append(int(issue["correct_conclusion"]))
                
# Bar chart for all three categories, grouped by generator in mini_generator_list
fig = plt.figure(figsize=(8, 5))
bar_width = 0.25
x = np.arange(len(mini_generator_list))
no_children_identified_avg = [sum(no_children_identified[gen]) / len(no_children_identified[gen]) if len(no_children_identified[gen]) > 0 else 0 for gen in mini_generator_list]
some_children_identified_are_wrong_avg = [sum(some_children_identified_are_wrong[gen]) / len(some_children_identified_are_wrong[gen]) if len(some_children_identified_are_wrong[gen]) > 0 else 0 for gen in mini_generator_list]
all_children_identified_are_correct_avg = [sum(all_children_identified_are_correct[gen]) / len(all_children_identified_are_correct[gen]) if len(all_children_identified_are_correct[gen]) > 0 else 0 for gen in mini_generator_list]
plt.bar(x - bar_width, no_children_identified_avg, width=bar_width, label="No children identified", color="#feb253")
plt.bar(x, some_children_identified_are_wrong_avg, width=bar_width, label="Some children identified are wrong", color="#ffc5d0")
plt.bar(x + bar_width, all_children_identified_are_correct_avg, width=bar_width, label="All children identified are correct", color="#ff8ca1")
plt.xticks(ticks=x, labels=mini_generator_list)
plt.ylim(0, 1)
for i in range(len(mini_generator_list)):
    plt.text(i - bar_width, no_children_identified_avg[i] + 0.02, f"{no_children_identified_avg[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i, some_children_identified_are_wrong_avg[i] + 0.02, f"{some_children_identified_are_wrong_avg[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i + bar_width, all_children_identified_are_correct_avg[i] + 0.02, f"{all_children_identified_are_correct_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Correctness of identified parent issue")
plt.title("Effects of Child Issue Results on Parent Issue Correctness")
plt.legend(loc="lower left")
plt.savefig("plots/PAPER_effects_of_children_node_results.svg")
plt.close()

# Whenever a non-leaf issue has (True, False), classify them as no child identified / some child identified are wrong / all children identified are correct
no_children_identified = {x: 0 for x in generator_list}
some_children_identified_are_wrong = {x: 0 for x in generator_list}
all_children_identified_are_correct = {x: 0 for x in generator_list}
for generator in generator_list:
    evaluator = "gemini-2.0-flash-001"
    for doc_id in evaluation_dict[evaluator][generator].keys():
        issue_dict = evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]
        for issue_id, issue in issue_dict.items():
            children_results = []
            for issue_id2, issue2 in issue_dict.items():
                if issue_id2.startswith(issue_id) and issue_id2 != issue_id:
                    children_results.append((issue2["contains_issue"], issue2["correct_conclusion"]))
            if len(children_results) == 0:
                # No children -> pass
                continue
            if issue["contains_issue"] and not issue["correct_conclusion"]:
                if all([not res[0] for res in children_results]):
                    no_children_identified[generator] += 1
                elif any([res[0] and not res[1] for res in children_results]):
                    some_children_identified_are_wrong[generator] += 1
                elif all([res[0] == res[1] for res in children_results]):
                    all_children_identified_are_correct[generator] += 1
# Draw stacked bar chart for all three categories, grouped by generator in mini_generator_list
fig = plt.figure(figsize=(8, 5))
bar_width = 0.5
x = np.arange(len(mini_generator_list))
no_children_identified_avg = [no_children_identified[gen] for gen in mini_generator_list]
some_children_identified_are_wrong_avg = [some_children_identified_are_wrong[gen] for gen in mini_generator_list]
all_children_identified_are_correct_avg = [all_children_identified_are_correct[gen] for gen in mini_generator_list]
plt.bar(x, no_children_identified_avg, width=bar_width, label="No children identified", color="#ffcd8e")
plt.bar(x, some_children_identified_are_wrong_avg, bottom=no_children_identified_avg, width=bar_width, label="Some children identified are wrong", color="#feb253")
plt.bar(x, all_children_identified_are_correct_avg, bottom=[x+y for x, y in zip(no_children_identified_avg, some_children_identified_are_wrong_avg)], width=bar_width, label="All children identified are correct", color="#ff8ca1")
plt.xticks(ticks=x, labels=mini_generator_list, rotation=45)
# Add count labels to all three sections
for i in range(len(mini_generator_list)):
    plt.text(i, no_children_identified_avg[i] / 2, f"{no_children_identified_avg[i]}", ha="center", va="center", color="black")
    plt.text(i, no_children_identified_avg[i] + some_children_identified_are_wrong_avg[i] / 2, f"{some_children_identified_are_wrong_avg[i]}", ha="center", va="center", color="black")
    plt.text(i, no_children_identified_avg[i] + some_children_identified_are_wrong_avg[i] + all_children_identified_are_correct_avg[i] / 2, f"{all_children_identified_are_correct_avg[i]}", ha="center", va="center", color="black")
plt.ylabel("Number of Non-Leaf Issues Identified but Incorrect")
plt.title("Effects of Subissue Coverage/Correctness on Parent Issue Incorrectness")
plt.legend(loc="upper right")
plt.savefig("plots/effects_of_children_node_results_count.svg")
plt.close()


# Correlation between leaf issue coverage and final order correctness
leaf_issue_coverage = {generator: [] for generator in generator_list}
leaf_issue_correctness = {generator: [] for generator in generator_list}
final_order_correctness = {generator: [] for generator in generator_list}
for generator in generator_list:
    evaluator = "gemini-2.0-flash-001"
    for doc_id in evaluation_dict[evaluator][generator].keys():
        issue_dict = evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]
        num_leaf_issues = 0
        num_leaf_issues_covered = 0
        num_leaf_issues_correct = 0
        final_order_correct = False
        for issue_id, issue in issue_dict.items():
            # Check if leaf
            leaf = True
            for issue_id2 in issue_dict.keys():
                if issue_id2.startswith(issue_id) and issue_id2 != issue_id:
                    leaf = False
                    break
            if leaf:
                num_leaf_issues += 1
                if issue["contains_issue"]:
                    num_leaf_issues_covered += 1
                    if issue["correct_conclusion"]:
                        num_leaf_issues_correct += 1
            if issue_id == "":
                final_order_correct = issue["correct_conclusion"]
        if num_leaf_issues > 0:
            leaf_issue_coverage[generator].append(num_leaf_issues_covered / num_leaf_issues)
            leaf_issue_correctness[generator].append(num_leaf_issues_correct / num_leaf_issues_covered if num_leaf_issues_covered > 0 else 0)
            final_order_correctness[generator].append(int(final_order_correct))

# Binned line chart for mini_generator_list
plt.figure(figsize=(8, 5))
for generator in mini_generator_list:
    coverage_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.00001]
    bin_centers = [(coverage_bins[i] + coverage_bins[i+1]) / 2 for i in range(len(coverage_bins)-1)]
    binned_final_order_correctness = []
    for i in range(len(coverage_bins)-1):
        bin_values = [final_order_correctness[generator][j] for j in range(len(leaf_issue_coverage[generator])) if coverage_bins[i] <= leaf_issue_coverage[generator][j] < coverage_bins[i+1]]
        if len(bin_values) > 0:
            binned_final_order_correctness.append(sum(bin_values) / len(bin_values))
        else:
            binned_final_order_correctness.append(np.nan)
    plt.plot(bin_centers, binned_final_order_correctness, marker="o", label=generator)
plt.xlabel("Leaf Issue Coverage")
plt.ylabel("Final Order Correctness")
plt.title("Final Order Correctness vs Leaf Issue Coverage")
plt.ylim(0, 1)
plt.legend(loc="upper left")
plt.savefig("plots/final_order_correctness_vs_leaf_issue_coverage.svg")
plt.close()

plt.figure(figsize=(8, 5))
for generator in mini_generator_list:
    coverage_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.00001]
    bin_centers = [(coverage_bins[i] + coverage_bins[i+1]) / 2 for i in range(len(coverage_bins)-1)]
    binned_final_order_correctness = []
    for i in range(len(coverage_bins)-1):
        bin_values = [final_order_correctness[generator][j] for j in range(len(leaf_issue_correctness[generator])) if coverage_bins[i] <= leaf_issue_correctness[generator][j] < coverage_bins[i+1]]
        if len(bin_values) > 0:
            binned_final_order_correctness.append(sum(bin_values) / len(bin_values))
        else:
            binned_final_order_correctness.append(np.nan)
    plt.plot(bin_centers, binned_final_order_correctness, marker="o", label=generator)
plt.xlabel("Leaf Issue Correctness (only when covered)")
plt.ylabel("Final Order Correctness")
plt.title("Final Order Correctness vs Leaf Issue Correctness")
plt.ylim(0, 1)
plt.legend(loc="upper left")
plt.savefig("plots/final_order_correctness_vs_leaf_issue_correctness.svg")
plt.close()

##################################################################################
# Likert scale analysis

# Dict storing:
# evaluator -> generator -> docid -> List[evaluations] (evaluations: Dict[str, Dict[str, bool]])
evaluation_dict = {
    evaluator: {
        generator: {}
        for generator in generator_list
    }
    for evaluator in evaluator_list
}
# Difficulty dict
difficulty_dict = {}
with open("data/reasoning_tasks_test.jsonl", "r") as f:
    for line in f:
        datum = json.loads(line.strip())
        doc_id = datum["doc_id"]
        difficulty = datum["difficulty"]
        difficulty_dict[doc_id] = difficulty

# Load all files
for evaluator in evaluator_list:
    if "human" in evaluator:
        continue
    for generator in generator_list:
        # Load the JSON file for each evaluator and generator
        file_path = f"results/reasoning_tasks_{generator}_likertevaluator_{evaluator}.jsonl"
        try:
            with open(file_path, "r") as f:
                for line in f:
                    datum = json.loads(line.strip())
                    doc_id = datum["doc_id"]
                    evaluation_dict[evaluator][generator][doc_id] = datum
        except FileNotFoundError:
            # print(f"File not found: {file_path}")
            pass
for evaluator in ["human_group1", "human_group2"]:
    with open(f"results/{evaluator}.json", "r") as f:
        data = json.load(f)
        for model, datum in data.items():
            if model not in evaluation_dict[evaluator]:
                evaluation_dict[evaluator][model] = {}
            for doc_id, issues in datum.items():
                evaluation_dict[evaluator][model][doc_id] = issues

# Compute N*N label agreement / Spearman's correlation matrix
pearson_likert = np.zeros((len(evaluator_list), len(evaluator_list)))
score_individual_scatter_plots = [("human_group1", "human_group2"), ("human_group1", "gemini-2.0-flash-001"), ("human_group1", "gemini-2.5-flash")]

# Top eschelon form
for i, evaluator_i in enumerate(evaluator_list):
    for j, evaluator_j in enumerate(evaluator_list[i:], start=i):
        # print(i, j)
        scores_i = [] # score (0-10)
        scores_j = []
        for k, generator in enumerate(generator_list):
            evaluator_i_docids = set(evaluation_dict[evaluator_i][generator].keys())
            evaluator_j_docids = set(evaluation_dict[evaluator_j][generator].keys())
            docids = evaluator_i_docids.intersection(evaluator_j_docids)
            for doc_id in docids:
                doc_results_i, doc_results_j = [], []
                scores_i.append(evaluation_dict[evaluator_i][generator][doc_id]["score"])
                scores_j.append(evaluation_dict[evaluator_j][generator][doc_id]["score"])
        pearson_likert[i, j] = np.corrcoef(scores_i, scores_j)[0, 1] if scores_i and scores_j else np.nan

        if (evaluator_list[i], evaluator_list[j]) in score_individual_scatter_plots:
            plt.figure(figsize=(8, 6))
            plt.scatter(scores_i, scores_j)
            plt.xlabel(evaluator_list[i])
            plt.ylabel(evaluator_list[j])
            plt.title(f"Scatter plot: {evaluator_list[i]} vs {evaluator_list[j]}")
            plt.savefig(f"plots/likert_score_scatter_{evaluator_list[i]}_{evaluator_list[j]}.svg")
            plt.close()

# Pearson
plt.figure(figsize=(8, 6))
plt.imshow(np.nan_to_num(pearson_likert), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for j in range(len(evaluator_list)):
        if not np.isnan(pearson_likert[i, j]):
            plt.text(j, i, f"{pearson_likert[i, j]:.2f}", ha="center", va="center", color="white")
plt.title("Pearson Correlation between Evaluators (Likert scale)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/likert_pearson.svg")

# Place LLM-LLM agreement of LEGIT score and likert score side by side for comparison
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.nan_to_num(pearson[2:, 2:]), cmap=cmap_custom_blue, vmin=0, vmax=1)
plt.xticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
plt.yticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
# add data labels inside boxes
for i in range(len(evaluator_models)):
    for j in range( i, len(evaluator_models)):
        if not np.isnan(pearson[i+2, j+2]):
            plt.text(j, i, f"{pearson[i+2, j+2]:.2f}", ha="center", va="center", color="white" if pearson[i+2, j+2] > 0.11 else "black")
plt.title("Pearson Correlation between Evaluators (LEGIT score)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.nan_to_num(pearson_likert[2:, 2:]), cmap=cmap_custom_blue, vmin=0, vmax=1)
plt.xticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
plt.yticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
# add data labels inside boxes
for i in range(len(evaluator_models)):
    for j in range(i, len(evaluator_models)):
        if not np.isnan(pearson_likert[i+2, j+2]):
            plt.text(j, i, f"{pearson_likert[i+2, j+2]:.2f}", ha="center", va="center", color="white" if pearson_likert[i+2, j+2] > 0.11 else "black")
plt.title("Pearson Correlation between Evaluators (Likert scale)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.colorbar()
plt.savefig("plots/PAPER_likert_vs_legit_pearson_comparison.svg")