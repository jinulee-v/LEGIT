import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Any
from matplotlib.patches import Patch
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

full_generator_list = generator_list + [
    "gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp",
    "gemma3-4b_evaluator_gemma3-27b_rootreward",
    "rag_bm25_gemini-2.5-flash",
    "rag_bm25_gpt-4.1",
    "rag_bm25_gemma-3-12b-it",
    "rag_bm25_gemma-3-4b-it",
    "rag_groundtruth_gemini-2.5-flash",
    "rag_groundtruth_gpt-4.1",
    "rag_groundtruth_gemma-3-12b-it",
    "rag_groundtruth_gemma-3-4b-it",
    "rag_contriever_gemini-2.5-flash",
    "rag_contriever_gpt-4.1",
    "rag_contriever_gemma-3-12b-it",
    "rag_contriever_gemma-3-4b-it",
    "rag_contriever_finetuned_gemini-2.5-flash",
    "rag_contriever_finetuned_gpt-4.1",
    "rag_contriever_finetuned_gemma-3-12b-it",
    "rag_contriever_finetuned_gemma-3-4b-it",
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

rag_comparison_list = [
    "gemma-3-4b-it",
    ""
]

cmap_custom_blue = LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", "#caedec", "#8ed7d7", "#00afbd", "#00383c"], N=256)
cmap_custom_orange = LinearSegmentedColormap.from_list("custom_orange", ["#ffffff", "#feb253", "#fd8c00"], N=256)

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

def krippendorffs_alpha_continuous(rater1, rater2):
    """
    Compute Krippendorff's alpha (interval/continuous version) 
    between two raters who assign numeric values (e.g., 0–10).

    Parameters
    ----------
    rater1 : list or np.ndarray
        Scores from rater 1
    rater2 : list or np.ndarray
        Scores from rater 2

    Returns
    -------
    alpha : float
        Krippendorff's alpha (continuous/interval version)
    """
    rater1 = np.asarray(rater1, dtype=float)
    rater2 = np.asarray(rater2, dtype=float)

    # remove missing values (NaN handling if present)
    mask = ~(np.isnan(rater1) | np.isnan(rater2))
    rater1, rater2 = rater1[mask], rater2[mask]

    n = len(rater1)
    if n == 0:
        raise ValueError("No valid paired data to compute alpha.")

    # Build matrix of all values
    ratings = np.vstack([rater1, rater2]).T  # shape (n, 2)

    # --- Observed disagreement (Do) ---
    Do = 0.0
    count = 0
    for row in ratings:
        # pairwise disagreement for each item
        if np.all(~np.isnan(row)):
            Do += (row[0] - row[1])**2
            count += 1
    Do = Do / count if count > 0 else 0.0

    # --- Expected disagreement (De) ---
    all_values = np.concatenate([rater1, rater2])
    mean = np.mean(all_values)
    De = 2 * np.mean((all_values - mean)**2)

    # Krippendorff’s alpha
    alpha = 1 - (Do / De if De > 0 else np.nan)
    return alpha

# Dict storing:
# evaluator -> generator -> docid -> List[evaluations] (evaluations: Dict[str, Dict[str, bool]])
evaluation_dict = {
    evaluator: {
        generator: {}
        for generator in full_generator_list
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
    for generator in full_generator_list:
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
krippendorff = np.zeros((len(evaluator_list), len(evaluator_list)))
krippendorff_coverage = np.zeros((len(evaluator_list), len(evaluator_list)))
krippendorff_correctness = np.zeros((len(evaluator_list), len(evaluator_list)))
krippendorff_root = np.zeros((len(evaluator_list), len(evaluator_list)))
score_individual_scatter_plots = [("human_group1", "human_group2"), ("human_group1", "gemini-2.0-flash-001"), ("human_group1", "gemma-3-12b-it"), ("human_group2", "gemini-2.0-flash-001")]
scatter_plot_dots = ["o", "D", "^", "v"]

# Top eschelon form
for i, evaluator_i in enumerate(evaluator_list):
    for j, evaluator_j in enumerate(evaluator_list[i:], start=i):
        # print(i, j)
        scores_i = [] # score (0-10)
        scores_j = []
        issue_coverage_i = []
        issue_coverage_j = []
        issue_correctness_i = []
        issue_correctness_j = []
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
                if len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) <= 1:
                    continue
                if len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) <= 1:
                    continue
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
                    result_i_score = 5 if (result_i["contains_issue"] and result_i["correct_conclusion"]) else (2 if result_i["contains_issue"] else 0)
                    result_j_score = 5 if (result_j["contains_issue"] and result_j["correct_conclusion"]) else (2 if result_j["contains_issue"] else 0)
                    doc_results_i.append(result_i_score)
                    doc_results_j.append(result_j_score)

                # partial scores
                issue_coverage_i_doc = 0
                issue_coverage_j_doc = 0
                issue_correctness_i_doc = 0
                issue_correctness_j_doc = 0
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
                    else:
                        if result_i["contains_issue"] and result_i["correct_conclusion"]:
                            issue_correctness_i_doc += 3 / (len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) - 1)
                        if result_i["contains_issue"]:
                            issue_coverage_i_doc += 2 / (len(evaluation_dict[evaluator_i][generator][doc_id]["evaluation_result"]) - 1)

                        if result_j["contains_issue"] and result_j["correct_conclusion"]:
                            issue_correctness_j_doc += 3 / (len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) - 1)
                        if result_j["contains_issue"]:
                            issue_coverage_j_doc += 2 / (len(evaluation_dict[evaluator_j][generator][doc_id]["evaluation_result"]) - 1)
                all_results_i.extend(doc_results_i)
                all_results_j.extend(doc_results_j)
                issue_coverage_i.append(issue_coverage_i_doc)
                issue_coverage_j.append(issue_coverage_j_doc)
                issue_correctness_i.append(issue_correctness_i_doc)
                issue_correctness_j.append(issue_correctness_j_doc)

        # agreed[i, j] = cohens_kappa(all_results_i, all_results_j, [(False, False), (False, True), (True, False), (True, True)])
        agreed[i, j] = krippendorffs_alpha_continuous(all_results_i, all_results_j)
        agreed_root[i, j] = cohens_kappa(root_results_i, root_results_j, [False, True])
        agreed_root_strict[i, j] = cohens_kappa(root_results_strict_i, root_results_strict_j, [False, True])
        pearson[i, j] = np.corrcoef(scores_i, scores_j)[0, 1] if scores_i and scores_j else np.nan
        krippendorff[i, j] = krippendorffs_alpha_continuous(scores_i, scores_j)
        krippendorff_coverage[i, j] = krippendorffs_alpha_continuous(issue_coverage_i, issue_coverage_j)
        krippendorff_correctness[i, j] = krippendorffs_alpha_continuous(issue_correctness_i, issue_correctness_j)
        krippendorff_root[i, j] = krippendorffs_alpha_continuous([(5 if x else 0) for x in root_results_i], [(5 if x else 0) for x in root_results_j])


        if (evaluator_list[i], evaluator_list[j]) in score_individual_scatter_plots:
            plt.figure(figsize=(8, 5))
            plt.xlim(-0.5, 10.5)
            plt.ylim(-0.5, 10.5)
            plt.scatter(scores_i, scores_j, marker = scatter_plot_dots[score_individual_scatter_plots.index((evaluator_list[i], evaluator_list[j]))], color="#00afbd")
            plt.xlabel(evaluator_list[i])
            plt.ylabel(evaluator_list[j])
            plt.title(f"Scatter plot: {evaluator_list[i]} vs {evaluator_list[j]}")
            plt.savefig(f"plots/legit_score_scatter_{evaluator_list[i]}_{evaluator_list[j]}.svg")
            plt.close()

            # Print confusion matrix for all_results_i and all_results_j
            # value being 0, 2, 5
            confusion_matrix = {l1: {l2: 0 for l2 in [0, 2, 5]} for l1 in [0, 2, 5]}
            for a1, a2 in zip(all_results_i, all_results_j):
                confusion_matrix[a1][a2] += 1
            # Plot confusion matrix
            plt.figure(figsize=(4,3))
            plt.imshow(np.array([[confusion_matrix[i][j] for j in [0, 2, 5]] for i in [0, 2, 5]]), vmin=0, vmax=150, cmap=cmap_custom_blue)
            plt.colorbar()
            plt.xticks(ticks=[0, 1, 2], labels=["Not covered", "Covered", "Correct"])
            plt.yticks(ticks=[0, 1, 2], labels=["Not covered", "Covered", "Correct"])
            # data labels
            for x in range(3):
                for y in range(3):
                    plt.text(y, x, confusion_matrix[[0, 2, 5][x]][[0, 2, 5][y]], ha="center", va="center", color="white" if confusion_matrix[[0, 2, 5][x]][[0, 2, 5][y]] > 75 else "black")
            plt.title(f"Confusion matrix: {evaluator_list[i]} vs {evaluator_list[j]}")
            plt.savefig(f"plots/confusion_matrix_{evaluator_list[i]}_{evaluator_list[j]}.svg")
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
# human_avg = (agreed_root[0, 2:] + agreed_root[1, 2:]) / 2
human_avg = agreed_root[0, 2:]
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

# bar graph for krippendorff

plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
# Fisher z transformation between each mode <>human1 and <>human2
human_avg = (krippendorff_coverage[0, 2:] + krippendorff_coverage[1, 2:]) / 2
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#8ed7d7")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("krippendorff's α")
# dotted line at human-human agreement
human_human_krippendorf = krippendorff_coverage[0, 1]
plt.axhline(y=human_human_krippendorf, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_krippendorf + 0.02, f"Human-Human: {human_human_krippendorf:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Average krippendorff's α - coverage")
plt.savefig("plots/PAPER_legit_score_krippendorff_coverage.svg")
plt.close()

plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
# Fisher z transformation between each mode <>human1 and <>human2
human_avg = (krippendorff_correctness[0, 2:] + krippendorff_correctness[1, 2:]) / 2
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#8ed7d7")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("krippendorff's α")
# dotted line at human-human agreement
human_human_krippendorf = krippendorff_correctness[0, 1]
plt.axhline(y=human_human_krippendorf, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_krippendorf + 0.02, f"Human-Human: {human_human_krippendorf:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Average krippendorff's α - coverage")
plt.savefig("plots/PAPER_legit_score_krippendorff_correctness.svg")
plt.close()

plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
# Fisher z transformation between each mode <>human1 and <>human2
human_avg = (krippendorff_root[0, 2:] + krippendorff_root[1, 2:]) / 2
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#8ed7d7")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("krippendorff's α")
# dotted line at human-human agreement
human_human_krippendorf = krippendorff_root[0, 1]
plt.axhline(y=human_human_krippendorf, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_krippendorf + 0.02, f"Human-Human: {human_human_krippendorf:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Average krippendorff's α - coverage")
plt.savefig("plots/PAPER_legit_score_krippendorff_root.svg")
plt.close()



# root
plt.figure(figsize=(8,5))
evaluator_models = evaluator_list[2:]  # exclude humans
human_avg = (krippendorff[0, 2:] + krippendorff[1, 2:]) / 2
x = np.arange(len(evaluator_models))
plt.bar(x, human_avg, color="#00afbd")
plt.xticks(ticks=x, labels=evaluator_models)
plt.ylim(0, 1)
for i in range(len(evaluator_models)):
    plt.text(i, human_avg[i] + 0.02, f"{human_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("krippendorff's α")
# dotted line at human-human agreement
human_human_krippendorf = krippendorff[0, 1]
plt.axhline(y=human_human_krippendorf, color='black', linestyle='--')
plt.text(len(evaluator_models)-1, human_human_krippendorf + 0.02, f"Human-Human: {human_human_krippendorf:.2f}", ha="right", va="bottom", color="black")
plt.legend()
plt.title("Average krippendorff's α against Human Evaluators")
plt.savefig("plots/PAPER_legit_score_krippendorf.svg")
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


# Draw a grouped/stacked bar chart.
# For all mini_generator_list as groups, show average score within easy/medium/hard evaluated by gemini-2.0-flash-001.
# Stack final order correctness, issue coverage, and issue correctness scores.
plt.figure(figsize=(8, 5))
bar_width = 0.2
x = np.arange(len(mini_generator_list))
for offset in [-bar_width, 0, bar_width]:
    final_order_scores_averages = []
    issue_coverage_scores_averages = []
    issue_correctness_scores_averages = []
    for generator in mini_generator_list:
        final_order_scores = []
        issue_coverage_scores = []
        issue_correctness_scores = []
        j = generator_list.index(generator)
        i = evaluator_list.index("gemini-2.0-flash-001") - 2
        final_order_score = gen_eval_root_correct_total[j, i] * 5
        for doc_id in evaluation_dict["gemini-2.0-flash-001"][generator].keys():
            if difficulty_dict[doc_id] != ["easy", "medium", "hard"][int(offset//bar_width) + 1]:
                continue
            num_issues = len(evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"]) - 1
            if num_issues <= 0:
                continue
            try:
                final_order_score = 5 if evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"][""]["correct_conclusion"] else 0
            except KeyError:
                final_order_score = 0
            num_covered = sum(1 for issue_id, result in evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"].items() if issue_id != "" and result["contains_issue"])
            num_correct = sum(1 for issue_id, result in evaluation_dict["gemini-2.0-flash-001"][generator][doc_id]["evaluation_result"].items() if issue_id != "" and result["contains_issue"] and result["correct_conclusion"])
            issue_coverage_score = (num_covered / num_issues) * 2 if num_issues > 0 else 0
            issue_correctness_score = (num_correct / num_issues) * 3 if num_issues > 0 else 0
            final_order_scores.append(final_order_score)
            issue_coverage_scores.append(issue_coverage_score)
            issue_correctness_scores.append(issue_correctness_score)
        final_order_scores_averages.append(sum(final_order_scores) / len(final_order_scores) if final_order_scores else 0)
        issue_coverage_scores_averages.append(sum(issue_coverage_scores) / len(issue_coverage_scores) if issue_coverage_scores else 0)
        issue_correctness_scores_averages.append(sum(issue_correctness_scores) / len(issue_correctness_scores) if issue_correctness_scores else 0)
    plt.bar(x + offset, final_order_scores_averages, width=bar_width, color="#98c126")
    plt.bar(x + offset, issue_coverage_scores_averages, bottom=final_order_scores_averages, width=bar_width, color="#feb253")
    plt.bar(x + offset, issue_correctness_scores_averages, bottom=np.array(final_order_scores_averages) + np.array(issue_coverage_scores_averages), width=bar_width, color="#ff8ca1")
    for i in range(len(final_order_scores_averages)):
        plt.text(x[i] + offset, final_order_scores_averages[i] / 2, f"{final_order_scores_averages[i]:.2f}", ha="center", va="center", color="black")
        plt.text(x[i] + offset, final_order_scores_averages[i] + issue_coverage_scores_averages[i] / 2, f"{issue_coverage_scores_averages[i]:.2f}", ha="center", va="center", color="black")
        plt.text(x[i] + offset, final_order_scores_averages[i] + issue_coverage_scores_averages[i] + issue_correctness_scores_averages[i] / 2, f"{issue_correctness_scores_averages[i]:.2f}", ha="center", va="center", color="black")
        plt.text(x[i] + offset, final_order_scores_averages[i] + issue_coverage_scores_averages[i] + issue_correctness_scores_averages[i] + 0.1, f"{final_order_scores_averages[i] + issue_coverage_scores_averages[i] + issue_correctness_scores_averages[i]:.2f}", ha="center", va="bottom", color="black")
# Add legend
plt.xticks(ticks=x, labels=mini_generator_list)
plt.ylim(0, 10)
plt.ylabel("Score")
plt.title("LEGIT score per difficulty levels of different LLMs")
plt.legend()
plt.savefig("plots/PAPER_legit_score_components_by_generator_and_difficulty.svg")
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
plt.title("Issue Coverage and Correctness by Depth")
plt.xlabel("Depth")
plt.ylabel("Issue coverage")
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
plt.savefig("plots/PAPER_issues_coverage_and_correctness_by_depth_combined.svg", bbox_inches='tight')

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

# Plot root correctness, issue coverage, issue correctness.

generator_root_scores = []
generator_issue_coverage = []
generator_issue_correctness = []
generator_total_scores = []
for generator in generator_list:
    evaluator = "gemini-2.0-flash-001"
    root_scores = []
    issue_coverage = []
    issue_correctness = []
    for doc_id in evaluation_dict[evaluator][generator].keys():
        num_subissues = len(evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]) - 1
        coverage = 0
        correctness = 0
        if num_subissues <= 0:
            continue
        for issue_id, issue in evaluation_dict[evaluator][generator][doc_id]["evaluation_result"].items():
            if issue_id == "":
                root_scores.append(5 if issue["correct_conclusion"] else 0)
            else:
                if issue["contains_issue"]:
                    coverage += 2/num_subissues
                if issue["contains_issue"] and issue["correct_conclusion"]:
                    correctness += 3/num_subissues
        issue_coverage.append(coverage)
        issue_correctness.append(correctness)

    root_scores = sum(root_scores) / len(root_scores) if root_scores else 0
    issue_coverage = sum(issue_coverage) / len(issue_coverage) if issue_coverage else 0
    issue_correctness = sum(issue_correctness) / len(issue_correctness) if issue_correctness else 0
    generator_root_scores.append(root_scores)
    generator_issue_coverage.append(issue_coverage)
    generator_issue_correctness.append(issue_correctness)
    generator_total_scores.append(root_scores + issue_coverage + issue_correctness)
# Stacked bar chart: add reverse to match legend order
plt.figure(figsize=(8, 5))
plt.ylim(0,10)
plt.bar(generator_list, generator_issue_correctness, bottom=[x+y for x, y in zip(generator_root_scores, generator_issue_coverage)], label="Issue Correctness", color="#ff8ca1")
plt.bar(generator_list, generator_issue_coverage, bottom=generator_root_scores, label="Issue Coverage", color="#feb253")
plt.bar(generator_list, generator_root_scores, bottom=0, label="Final order Correctness", color="#98c126")
for i in range(len(generator_list)):
    print(generator_list[i], generator_root_scores[i], generator_issue_coverage[i], generator_issue_correctness[i], generator_root_scores[i] + generator_issue_coverage[i] + generator_issue_correctness[i])
    total_height = generator_root_scores[i] + generator_issue_coverage[i] + generator_issue_correctness[i]
    plt.text(i, total_height + 0.2, f"{total_height:.2f}", ha="center", va="bottom", color="black")
plt.legend(loc="upper right")
plt.savefig("plots/PAPER_LEGIT_scores_stacked.svg")
plt.close()

# Sort by total LEGIT score (root + coverage + correctness), descending
total_scores = [generator_root_scores[i] + generator_issue_coverage[i] + generator_issue_correctness[i] for i in range(len(generator_list))]
sorted_indices = np.argsort(total_scores)[::-1]

sorted_generator_list = [generator_list[i] for i in sorted_indices]
sorted_total_scores = [total_scores[i] for i in sorted_indices]
sorted_root_scores = [generator_root_scores[i] for i in sorted_indices]
sorted_issue_coverage = [generator_issue_coverage[i] for i in sorted_indices]
sorted_issue_correctness = [generator_issue_correctness[i] for i in sorted_indices]

# Assign different colors and hatches to different LLMs
color_map = {
    "o3": "#fd910c",
    "gpt-4.1": "#fd910c",
    "gpt-4.1-mini": "#fd910c",
    "gemini-2.5-pro": "#feb253",
    "gemini-2.5-flash-001": "#feb253",
    "gemini-2.0-flash-001": "#feb253",
    "gemma-3-27b-it": "#ffcd8e",
    "gemma-3-12b-it": "#ffcd8e",
    "gemma-3-4b-it": "#ffcd8e",
    "exaone3.5:32b": "#ffe5c5",
    "exaone3.5:7.8b": "#ffe5c5",
    "EXAONE-3.0-7.8B-Instruct": "#ffe5c5",
}
hatch_map = {
    "o3": "",
    "gpt-4.1": "",
    "gpt-4.1-mini": "",
    "gemini-2.5-pro": "//",
    "gemini-2.5-flash-001": "//",
    "gemini-2.0-flash-001": "//",
    "gemma-3-27b-it": ".",
    "gemma-3-12b-it": ".",
    "gemma-3-4b-it": ".",
    "exaone3.5:32b": "\\\\",
    "exaone3.5:7.8b": "\\\\",
    "EXAONE-3.0-7.8B-Instruct": "\\\\",
}
bar_colors = [color_map[g] for g in sorted_generator_list]
bar_hatches = [hatch_map[g] for g in sorted_generator_list]

plt.figure(figsize=(8, 5))
plt.ylim(0, 10)
# bars = plt.bar(sorted_generator_list, sorted_total_scores, color=bar_colors, hatch=[hatch_map[g] for g in sorted_generator_list], edgecolor="black", alpha=0.99)
bars = plt.bar(sorted_generator_list, sorted_total_scores, color=bar_colors)
# for bar, generator in zip(bars, sorted_generator_list):
#     bar.set_hatch(hatch_map[generator])
for i in range(len(sorted_generator_list)):
    plt.text(i, sorted_total_scores[i] + 0.2, f"{sorted_total_scores[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("LEGIT Score")
plt.title("LEGIT Scores by Generator (sorted)")
plt.xticks(rotation=45)
# Custom legend for colors/hatches
labels = {
    "gpt-4.1": "OpenAI",
    "gemini-2.0-flash-001": "Google Gemini",
    "gemma-3-4b-it": "Gemma-3",
    "exaone3.5:32b": "EXAONE",
}
# legend_handles = [Patch(facecolor=bar_colors[i], edgecolor="black", hatch=bar_hatches[i], label=labels[g]) for g in ["gpt-4.1", "gemini-2.0-flash-001", "gemma-3-4b-it", "exaone3.5:32b"] for i in range(len(bar_colors)) if sorted_generator_list[i] == g]
legend_handles = [Patch(facecolor=bar_colors[i], label=labels[g]) for g in ["gpt-4.1", "gemini-2.0-flash-001", "gemma-3-4b-it", "exaone3.5:32b"] for i in range(len(bar_colors)) if sorted_generator_list[i] == g]
plt.legend(handles=legend_handles, loc="upper right")
plt.tight_layout()
plt.savefig("plots/PAPER_LEGIT_scores.svg")
plt.close()

generator_root_scores = []
generator_issue_coverage = []
generator_issue_correctness = []
# rl_generator_list = ["gemma-3-4b-it", "rag_bm25_gemma-3-4b-it", "rag_groundtruth_gemma-3-4b-it", "gemma3-4b_evaluator_gemma3-27b_rootreward", "gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp"]
rl_generator_list = ["gemma-3-4b-it", "rag_bm25_gemma-3-4b-it", "rag_contriever_gemma-3-4b-it", "rag_groundtruth_gemma-3-4b-it", "gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp"]
for generator in rl_generator_list:
    evaluator = "gemini-2.0-flash-001"
    root_scores = []
    issue_coverage = []
    issue_correctness = []
    for doc_id in evaluation_dict[evaluator][generator].keys():
        num_subissues = len(evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]) - 1
        coverage = 0
        correctness = 0
        if num_subissues <= 0:
            continue
        for issue_id, issue in evaluation_dict[evaluator][generator][doc_id]["evaluation_result"].items():
            if issue_id == "":
                root_scores.append(5 if issue["correct_conclusion"] else 0)
            else:
                if issue["contains_issue"]:
                    coverage += 2/num_subissues
                if issue["contains_issue"] and issue["correct_conclusion"]:
                    correctness += 3/num_subissues
        issue_coverage.append(coverage)
        issue_correctness.append(correctness)

    root_scores = sum(root_scores) / len(root_scores) if root_scores else 0
    issue_coverage = sum(issue_coverage) / len(issue_coverage) if issue_coverage else 0
    issue_correctness = sum(issue_correctness) / len(issue_correctness) if issue_correctness else 0
    generator_root_scores.append(root_scores)
    generator_issue_coverage.append(issue_coverage)
    generator_issue_correctness.append(issue_correctness)
# Dot plot for rl_generators


data = [ # ^ o D -> final order reward, base, LEGIT reward
    generator_root_scores,
    generator_issue_coverage,
    generator_issue_correctness,
    [x+y+z for x, y, z in zip(generator_root_scores, generator_issue_coverage, generator_issue_correctness)]
]

print("=" * 80)
for d, label in zip(data, ["root", "coverage", "correctness", "total"]):
    print(label)
    for gen, score in zip(rl_generator_list, d):
        print(f"  {gen}: {score - d[0]:.2f}")
print("=" * 80)

plt.figure(figsize=(8, 5))
ranges = [(0, 5), (0, 2), (0, 3), (0, 10)]

fig, axes = plt.subplots(nrows=4, sharey=True, figsize=(8, 5))

for i, ax in enumerate(axes):
    y = np.arange(len(data[i]))
    ax.barh(y, list(reversed(data[i])), color=list(reversed(["#7f7f7f", "#caedec", "#8ed7d7", "#007a84", "#feb253"])), height=1)
    ax.set_xlim(ranges[i])
    ax.set_yticks(y)
    ax.set_yticklabels(list(reversed(["Gemma-3-4B", "RAG (BM25)", "RAG (Contriever)", "RAG (Ground truth)", "RL (LEGIT reward)"])))
    # add data labels on the right outside
    for j in range(len(data[i])):
        ax.text(data[i][len(data[i]) - 1 - j] + (ranges[i][1]-ranges[i][0]) / 100, j, f"{data[i][len(data[i]) - 1 - j]:.2f}", ha="left", va="center", color="black")

ax.legend(loc="upper right", bbox_to_anchor=(1.1, 4.5))

plt.tight_layout()
plt.savefig("plots/PAPER_LEGIT_scores_rl_rag_barplot.svg")
plt.close()


################################################################################
# 3 subplots


for model_idx, model_name in enumerate(["gpt-4.1", "gemini-2.5-flash", "gemma-3-4b-it"]):
    generator_root_scores = []
    generator_issue_coverage = []
    generator_issue_correctness = []
    rl_generator_list = [model_name, f"rag_bm25_{model_name}", f"rag_contriever_{model_name}", f"rag_contriever_finetuned_{model_name}", f"rag_groundtruth_{model_name}"]
    if model_name == "gemini-2.5-flash":
        rl_generator_list[0] = "gemini-2.5-flash-001"
    for generator in rl_generator_list:
        evaluator = "gemini-2.0-flash-001"
        root_scores = []
        issue_coverage = []
        issue_correctness = []
        for doc_id in evaluation_dict[evaluator][generator].keys():
            num_subissues = len(evaluation_dict[evaluator][generator][doc_id]["evaluation_result"]) - 1
            coverage = 0
            correctness = 0
            if num_subissues <= 0:
                continue
            for issue_id, issue in evaluation_dict[evaluator][generator][doc_id]["evaluation_result"].items():
                if issue_id == "":
                    root_scores.append(5 if issue["correct_conclusion"] else 0)
                else:
                    if issue["contains_issue"]:
                        coverage += 2/num_subissues
                    if issue["contains_issue"] and issue["correct_conclusion"]:
                        correctness += 3/num_subissues
            issue_coverage.append(coverage)
            issue_correctness.append(correctness)

        root_scores = sum(root_scores) / len(root_scores) if root_scores else 0
        issue_coverage = sum(issue_coverage) / len(issue_coverage) if issue_coverage else 0
        issue_correctness = sum(issue_correctness) / len(issue_correctness) if issue_correctness else 0
        generator_root_scores.append(root_scores)
        generator_issue_coverage.append(issue_coverage)
        generator_issue_correctness.append(issue_correctness)
    # Dot plot for rl_generators


    data = [ # ^ o D -> final order reward, base, LEGIT reward
        generator_root_scores,
        generator_issue_coverage,
        generator_issue_correctness,
        [x+y+z for x, y, z in zip(generator_root_scores, generator_issue_coverage, generator_issue_correctness)]
    ]

    print("=" * 80)
    for d, label in zip(data, ["root", "coverage", "correctness", "total"]):
        print(label)
        for gen, score in zip(rl_generator_list, d):
            print(f"  {gen}: {score - d[0]:.2f}")
    print("=" * 80)

    ranges = [(0, 5), (0, 2), (0, 3), (0, 10)]

    plt.figure(figsize=(8, 5))
    fig, axes = plt.subplots(nrows=4, sharey=True, figsize=(8, 5))

    # for i, ax in enumerate(axes):
    #     y = np.ones_like(data[i])  # put all dots at same vertical position
    #     for x, shape, color in zip(data[i], ['o', '^', 'v', '<', '>'], ["#7f7f7f", "#8ed7d7", "#00afbd", "#009daa", "#007a84"]):
    #         ax.plot(x, 0, shape, color=color)  # scatter along x
    #     ax.set_xlim(ranges[i])     # set different x-range per subplot
    #     ax.set_yticks([])         # remove y-axis ticks
    for i, ax in enumerate(axes):
        y = np.arange(len(data[i]))
        ax.barh(y, list(reversed(data[i])), color=list(reversed(["#7f7f7f", "#caedec", "#8ed7d7", "#00afbd", "#007a84"])), height=1)
        ax.set_xlim(ranges[i])
        ax.set_yticks(y)
        ax.set_yticklabels(list(reversed(["Gemma-3-4B", "RAG (BM25)", "RAG (Contriever)", "RAG (Fine-tuned Contriever)", "RAG (Ground truth)"])))
        # add data labels on the right outside
        for j in range(len(data[i])):
            ax.text(data[i][len(data[i]) - 1 - j] + (ranges[i][1]-ranges[i][0]) / 100, j, f"{data[i][len(data[i]) - 1 - j]:.2f}", ha="left", va="center", color="black")

    plt.tight_layout()
    plt.savefig(f"plots/PAPER_LEGIT_scores_rag_barplot_{model_name}.svg")
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
no_children_identified = {x: [] for x in full_generator_list}
some_children_identified_are_wrong = {x: [] for x in full_generator_list}
all_children_identified_are_correct = {x: [] for x in full_generator_list}
for generator in full_generator_list:
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
                    some_children_identified_are_wrong[generator].append(int(issue["correct_conclusion"] == True))
                elif all([res[0] and res[1] for res in children_results]):
                    all_children_identified_are_correct[generator].append(int(issue["correct_conclusion"] == True))
                
# Bar chart for all three categories, grouped by generator in mini_generator_list
fig = plt.figure(figsize=(8, 5))
bar_width = 0.25
x = np.arange(len(mini_generator_list))
no_children_identified_avg = [sum(no_children_identified[gen]) / len(no_children_identified[gen]) if len(no_children_identified[gen]) > 0 else 0 for gen in mini_generator_list]
some_children_identified_are_wrong_avg = [sum(some_children_identified_are_wrong[gen]) / len(some_children_identified_are_wrong[gen]) if len(some_children_identified_are_wrong[gen]) > 0 else 0 for gen in mini_generator_list]
all_children_identified_are_correct_avg = [sum(all_children_identified_are_correct[gen]) / len(all_children_identified_are_correct[gen]) if len(all_children_identified_are_correct[gen]) > 0 else 0 for gen in mini_generator_list]
plt.bar(x - bar_width, no_children_identified_avg, width=bar_width, label="No children covered", color="#feb253")
plt.bar(x, some_children_identified_are_wrong_avg, width=bar_width, label="Some children covered but incorrect", color="#ffc5d0")
plt.bar(x + bar_width, all_children_identified_are_correct_avg, width=bar_width, label="All covered children are correct", color="#ff8ca1")
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

rl_generator_list = ["gemma-3-4b-it", "gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp", "gemma3-4b_evaluator_gemma3-27b_rootreward"]
fig = plt.figure(figsize=(8, 5))
bar_width = 0.25
x = np.arange(len(rl_generator_list))
no_children_identified_avg = [sum(no_children_identified[gen]) / len(no_children_identified[gen]) if len(no_children_identified[gen]) > 0 else 0 for gen in rl_generator_list]
some_children_identified_are_wrong_avg = [sum(some_children_identified_are_wrong[gen]) / len(some_children_identified_are_wrong[gen]) if len(some_children_identified_are_wrong[gen]) > 0 else 0 for gen in rl_generator_list]
all_children_identified_are_correct_avg = [sum(all_children_identified_are_correct[gen]) / len(all_children_identified_are_correct[gen]) if len(all_children_identified_are_correct[gen]) > 0 else 0 for gen in rl_generator_list]
plt.bar(x - bar_width, no_children_identified_avg, width=bar_width, label="No children identified", color="#feb253")
plt.bar(x, some_children_identified_are_wrong_avg, width=bar_width, label="Some children identified are wrong", color="#ffc5d0")
plt.bar(x + bar_width, all_children_identified_are_correct_avg, width=bar_width, label="All children identified are correct", color="#ff8ca1")
plt.xticks(ticks=x, labels=rl_generator_list)
plt.ylim(0, 1)
for i in range(len(rl_generator_list)):
    plt.text(i - bar_width, no_children_identified_avg[i] + 0.02, f"{no_children_identified_avg[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i, some_children_identified_are_wrong_avg[i] + 0.02, f"{some_children_identified_are_wrong_avg[i]:.2f}", ha="center", va="bottom", color="black")
    plt.text(i + bar_width, all_children_identified_are_correct_avg[i] + 0.02, f"{all_children_identified_are_correct_avg[i]:.2f}", ha="center", va="bottom", color="black")
plt.ylabel("Correctness of identified parent issue")
plt.title("Effects of Child Issue Results on Parent Issue Correctness")
plt.legend(loc="lower left")
plt.savefig("plots/PAPER_effects_of_children_node_results_rl.svg")
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

# Draw a heatmap based on child issue coverage and correctness
# x-axis: child issue coverage (0, 0.2, 0.4, 0.6, 0.8, 1)
# y-axis: child issue correctness (0, 0.2, 0.4, 0.6, 0.8, 1)
heatmap_data = {gen: np.zeros((5, 5)) for gen in full_generator_list}
heatmap_counts = {gen: np.zeros((5, 5)) for gen in full_generator_list}
for generator in full_generator_list:
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
                coverage = sum([1 for res in children_results if res[0]]) / len(children_results)
                correctness = sum([1 for res in children_results if res[0] and res[1]]) / len(children_results)
                # correctness = sum([1 for res in children_results if res[0] and res[1]]) / sum([1 for res in children_results if res[0]]) if sum([1 for res in children_results if res[0]]) > 0 else 0
                coverage_bin = min(int(coverage * 5), 4)
                correctness_bin = 4 - min(int(correctness * 5), 4) #origin goes to left corner
                heatmap_data[generator][correctness_bin, coverage_bin] += int(issue["correct_conclusion"] == True)
                heatmap_counts[generator][correctness_bin, coverage_bin] += 1
# Average the heatmap data
for generator in mini_generator_list + ["gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp", "gemma3-4b_evaluator_gemma3-27b_rootreward"]:
    for i in range(5):
        for j in range(5):
            if heatmap_counts[generator][i, j] > 0:
                heatmap_data[generator][i, j] /= heatmap_counts[generator][i, j]
            else:
                heatmap_data[generator][i, j] = np.nan
# Draw heatmaps
for generator in mini_generator_list + ["gemma3-4b_evaluator_gemma3-27b_fullreward_new_largelr_fixtemp", "gemma3-4b_evaluator_gemma3-27b_rootreward"]:
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap_data[generator], cmap=cmap_custom_orange, vmin=0, vmax=1)
    plt.colorbar(label="Parent Issue Correctness")
    plt.xticks(ticks=np.arange(5), labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    plt.yticks(ticks=np.arange(5), labels=reversed(["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]))
    plt.xlabel("Identified issue ratio")
    plt.ylabel("Identified and correct issue ratio")
    plt.title(f"Parent Issue Correctness by Child Issue Coverage and Correctness ({generator})")
    # Add counts to each cell
    for i in range(5):
        for j in range(5):
            if not np.isnan(heatmap_data[generator][i, j]):
                plt.text(j, i, f"{float(heatmap_data[generator][i, j]):2f}\n({int(heatmap_counts[generator][i, j])})", ha="center", va="center", color="black")
    plt.savefig(f"plots/parent_issue_correctness_by_child_issue_coverage_and_correctness_{generator}.svg")
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
evaluation_dict_legit = evaluation_dict
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
krippendorff_likert = np.zeros((len(evaluator_list), len(evaluator_list)))
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
        krippendorff_likert[i, j] = krippendorffs_alpha_continuous(scores_i, scores_j) if scores_i and scores_j else np.nan

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



# Place LLM-LLM agreement (krippendorff) of LEGIT score and likert score side by side for comparison
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.nan_to_num(krippendorff[2:, 2:]), cmap=cmap_custom_blue, vmin=0, vmax=1)
plt.xticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
plt.yticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
# add data labels inside boxes
for i in range(len(evaluator_models)):
    for j in range( i, len(evaluator_models)):
        if not np.isnan(krippendorff[i+2, j+2]):
            plt.text(j, i, f"{krippendorff[i+2, j+2]:.2f}", ha="center", va="center", color="white" if krippendorff[i+2, j+2] > 0.11 else "black")
plt.title("Krippendorff's Alpha between Evaluators (LEGIT score)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(np.nan_to_num(krippendorff_likert[2:, 2:]), cmap=cmap_custom_blue, vmin=0, vmax=1)
plt.xticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
plt.yticks(ticks=np.arange(len(evaluator_models)), labels=evaluator_models)
# add data labels inside boxes
for i in range(len(evaluator_models)):
    for j in range(i, len(evaluator_models)):
        if not np.isnan(krippendorff_likert[i+2, j+2]):
            plt.text(j, i, f"{krippendorff_likert[i+2, j+2]:.2f}", ha="center", va="center", color="white" if krippendorff_likert[i+2, j+2] > 0.11 else "black")
plt.title("Krippendorff's Alpha between Evaluators (Likert scale)")
# plt.xlabel("Evaluator")
# plt.ylabel("Evaluator")
plt.colorbar()
plt.savefig("plots/PAPER_likert_vs_legit_krippendorff_comparison.svg")

# Scatter plot between LEGIT score and likert score for gemini-2.0-flash-001
scores_legit = []
scores_likert = []

# extract generator, doc_id pairs from human eval
for generator, doc_id in evaluation_dict["human_group1"].items():
    evaluator = "gemini-2.0-flash-001"
    evaluator_docids = set(evaluation_dict["human_group1"][generator].keys())
    for doc_id in evaluator_docids:
        if doc_id in evaluation_dict_legit[evaluator][generator] and doc_id in evaluation_dict[evaluator][generator]:
            scores_legit.append(evaluation_dict_legit[evaluator][generator][doc_id]["score"])
            scores_likert.append(evaluation_dict[evaluator][generator][doc_id]["score"])
plt.figure(figsize=(8, 5))
plt.scatter(scores_legit, scores_likert)
plt.xlabel("LEGIT Score")
plt.ylabel("Likert Score")
plt.title("Scatter plot: LEGIT Score vs Likert Score (gemini-2.0-flash-001)")
# Fit a linear regression line
if scores_legit and scores_likert:
    m, b = np.polyfit(scores_legit, scores_likert, 1)
    plt.plot(sorted(scores_legit), [m*x + b for x in sorted(scores_legit)], color="red")
plt.savefig("plots/likert_vs_legit_scatter_gemini-2.0-flash-001.svg")
plt.close()