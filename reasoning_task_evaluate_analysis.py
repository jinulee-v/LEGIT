import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Any

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
    "EXAONE-3.0-7.8B-Instruct",
]

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
score_individual_scatter_plots = [("human_group1", "human_group2"), ("human_group1", "gemini-2.0-flash-001"), ("human_group1", "gemini-2.5-flash")]

# Top eschelon form
for i, evaluator_i in enumerate(evaluator_list):
    for j, evaluator_j in enumerate(evaluator_list[i:], start=i):
        # print(i, j)
        scores_i = [] # score (0-10)
        scores_j = []
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
                        break
                all_results_i.extend(doc_results_i)
                all_results_j.extend(doc_results_j)

        agreed[i, j] = cohens_kappa(all_results_i, all_results_j, [(False, False), (False, True), (True, False), (True, True)])
        agreed_root[i, j] = cohens_kappa(root_results_i, root_results_j, [False, True])
        agreed_root_strict[i, j] = cohens_kappa(root_results_strict_i, root_results_strict_j, [False, True])
        pearson[i, j] = np.corrcoef(scores_i, scores_j)[0, 1] if scores_i and scores_j else np.nan

        if (evaluator_list[i], evaluator_list[j]) in score_individual_scatter_plots:
            plt.figure(figsize=(8, 6))
            plt.scatter(scores_i, scores_j)
            plt.xlabel(evaluator_list[i])
            plt.ylabel(evaluator_list[j])
            plt.title(f"Scatter plot: {evaluator_list[i]} vs {evaluator_list[j]}")
            plt.savefig(f"plots/{evaluator_list[i]}_{evaluator_list[j]}_score_scatter.svg")
            plt.close()

# draw a colored grid based on agreed / total. Treat NaN as gray
plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_evaluators_label_agreement.svg")

# root nodes agreement
plt.figure(figsize=(10, 8))
plt.imshow(np.nan_to_num(agreed_root), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for k in range(len(evaluator_list)):
        if total_root[i, k] > 0:
            plt.text(k, i, f"{agreed_root[i, k]:.2f}", ha="center", va="center", color="white")
plt.title("Cohen's Kappa between Evaluators (Root only)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/all_evaluators_root_label_agreement.svg")

plt.figure(figsize=(10, 8))
plt.imshow(np.nan_to_num(agreed_root_strict), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for k in range(len(evaluator_list)):
        if total_root[i, k] > 0:
            plt.text(k, i, f"{agreed_root_strict[i, k]:.2f}", ha="center", va="center", color="white")
plt.title("Cohen's Kappa between Evaluators (Root only - strict)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/all_evaluators_root_label_agreement_strict.svg")


# Pearson
plt.figure(figsize=(10, 8))
plt.imshow(np.nan_to_num(pearson), cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list, rotation=45)
plt.yticks(ticks=np.arange(len(evaluator_list)), labels=evaluator_list)
# add data labels inside boxes
for i in range(len(evaluator_list)):
    for j in range(len(evaluator_list)):
        if not np.isnan(pearson[i, j]):
            plt.text(j, i, f"{pearson[i, j]:.2f}", ha="center", va="center", color="white")
plt.title("Pearson Correlation between Evaluators")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.savefig("plots/all_evaluators_pearson.svg")

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
plt.figure(figsize=(10, 8))
plt.imshow(datapoints, cmap="Blues", vmin=0, vmax=300)
plt.colorbar()
plt.xticks(ticks=np.arange(len(evaluator_list[2:])), labels=evaluator_list[2:], rotation=45)
plt.yticks(ticks=np.arange(len(generator_list)), labels=generator_list)
for i in range(len(generator_list)):
    for j in range(len(evaluator_list[2:])):
        if datapoints[i, j] > 0:
            plt.text(j, i, f"{datapoints[i, j]:.0f}", ha="center", va="center", color="white")
plt.title("Number of Data Points per (Generator, Evaluator) Pair")
plt.savefig("plots/all_generators_evaluators_count.svg")

# Gen-Eval plot
plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_score_total.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_score_easy.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_score_medium.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_score_hard.svg")

# Root correct figure
plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_root_correct_total.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_root_correct_easy.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_root_correct_medium.svg")

plt.figure(figsize=(10, 8))
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
plt.savefig("plots/all_generators_evaluators_root_correct_hard.svg")


##################################################################################

# Use Gemini-2.0-Flash evaluation results
generator_issue_total = {} # root, 1, 2, 3, 4+
generator_issue_coverage_per_depth = {}
generator_issue_coverage_and_correctness_per_depth = {}
for j, generator in enumerate(generator_list):
    generator_issue_total[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    generator_issue_coverage_per_depth[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    generator_issue_coverage_and_correctness_per_depth[generator] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for doc_id in evaluation_dict["gemini-2.5-flash"][generator].keys():
        for issue_id, result in evaluation_dict["gemini-2.5-flash"][generator][doc_id]["evaluation_result"].items():
            if issue_id == "":
                depth = 0
            else:
                depth = issue_id.count(".") + 1
            if depth >= 4:
                depth = 4
            generator_issue_total[generator][depth] += 1
            if result["contains_issue"]:
                generator_issue_coverage_per_depth[generator][depth] += 1
            if result["contains_issue"] and result["correct_conclusion"]:
                generator_issue_coverage_and_correctness_per_depth[generator][depth] += 1

for generator in generator_list:
    print(generator, generator_issue_total[generator])

# Draw overlapping line chart
depth_labels = ["Root", "1", "2", "3", "4+"]
plt.figure(figsize=(10, 8))
for generator in generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_per_depth[generator][i] / generator_issue_total[generator][i] if generator_issue_total[generator][i] > 0 else 0 for i in range(5)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Coverage")
plt.title("Issue Coverage by Depth")
plt.legend()
plt.savefig("plots/all_generators_issues_coverage_by_depth.svg")

plt.figure(figsize=(10, 8))
for generator in generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_and_correctness_per_depth[generator][i] / generator_issue_total[generator][i] if generator_issue_total[generator][i] > 0 else 0 for i in range(5)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Coverage and Correctness")
plt.title("Issue Coverage and Correctness by Depth")
plt.legend()
plt.savefig("plots/all_generators_issues_coverage_and_correctness_by_depth.svg")

plt.figure(figsize=(10, 8))
for generator in generator_list:
    plt.plot(depth_labels, [generator_issue_coverage_and_correctness_per_depth[generator][i] / generator_issue_coverage_per_depth[generator][i] if generator_issue_coverage_per_depth[generator][i] > 0 else 0 for i in range(5)], marker="o", label=generator)
plt.xlabel("Depth")
plt.ylabel("Correctness, only when issue is covered")
plt.title("Conditional Issue Correctness by Depth")
plt.legend()
plt.savefig("plots/all_generators_issues_coverage_and_correctness_by_depth_conditional.svg")
