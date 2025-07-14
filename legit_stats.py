import json
import re
import numpy as np

with open("data/legit.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f.readlines()]
print(f"Total number of data points: {len(data)}")

# Check if datum["legit"] is not empty
data = [datum for datum in data if datum["issues"]]
print(f"Number of data points with non-empty 'issues': {len(data)}")

# Check if "issues" have correct component types:
# claim: List[dict[str, str]]
# relevant_law, analysis: List[str]
# summary, conclusion: str
def check_types(issues):
    for issue in issues:
        for keys in ["id", "summary", "claim", "relevant_law", "analysis", "conclusion"]:
            if keys not in issue:
                return False
        if not isinstance(issue["summary"], str):
            return False
        if not isinstance(issue["claim"], list) or any(not isinstance(item, dict) or not isinstance(item.get("claimer"), str) for item in issue["claim"]) or any(not isinstance(item.get("content"), str) for item in issue["claim"]):
            return False
        if not isinstance(issue["relevant_law"], list) or any(not isinstance(item, str) for item in issue["relevant_law"]):
            return False
        if not isinstance(issue["analysis"], list) or any(not isinstance(item, str) for item in issue["analysis"]):
            return False
        if not isinstance(issue["conclusion"], str):
            return False
    return True
data = [datum for datum in data if check_types(datum["issues"])]
print(f"Number of data points with correct types: {len(data)}")

# Check if issue IDs are unique within each datum
def check_unique_ids(issues):
    seen_ids = set()
    for issue in issues:
        if issue["id"] in seen_ids:
            return False
        seen_ids.add(issue["id"])
    return True
data = [datum for datum in data if check_unique_ids(datum["issues"])]
print(f"Number of data points with unique issue IDs: {len(data)}")

# Check if "issues" have correct issue IDs
def check_hierarchy(issues):
    # collect IDs
    ids = set([issue["id"] for issue in issues])
    for issue in issues:
        # find parent ID
        if issue["id"] == "":
            continue
        elif re.fullmatch(r"[0-9]+(\.[0-9]+)*", issue["id"]):
            # replace (\.?[0-9]+)$ with empty string
            parent_id = re.sub(r"(\.?[0-9]+)$", "", issue["id"])
        else:
            return False
        # check if parent ID is in the set of IDs
        if parent_id not in ids:
            return False
    return True
data = [datum for datum in data if check_hierarchy(datum["issues"])]
print(f"Number of data points with correct issue IDs: {len(data)}")


# Check events
def check_events(datum):
    if "events" not in datum or not len(datum["events"]) > 0:
        return False
    for event in datum["events"]:
        if not isinstance(event, dict):
            return False
        if list(event.keys()) != ["event"]:
            return False
    return True
data = [datum for datum in data if check_events(datum)]
print(f"Number of data points with valid events: {len(data)}")

# Check if event summaries are not empty
def check_event_summary(datum):
    return "event_summary" in datum and isinstance(datum["event_summary"], str) and len(datum["event_summary"]) > 0
data = [datum for datum in data if check_event_summary(datum)]
print(f"Number of data points with non-empty 'event_summary': {len(data)}")


# Check event summary containing keywords from example
def check_event_summary_keywords(datum):
    if "event_summary" not in datum or not isinstance(datum["event_summary"], str):
        return False
    keywords = ["주식매매계약 제3.1조", "서울중앙지법 2017가합558413호", "약 54억 6천만 원", "코뼈골절", "안와골절"]
    return not any(keyword in datum["event_summary"].lower() for keyword in keywords)
data = [datum for datum in data if check_event_summary_keywords(datum)]
print(f"Number of data points with event summary containing keywords: {len(data)}")

print("\n------ Statistics ------")

# Average number of issues per datum
total_issues = sum(len(datum["issues"]) for datum in data)
average_issues = total_issues / len(data) if data else 0
print(f"Number of issues (per doc, average)  : {average_issues:.2f}")
# min, Q1, Q2, Q3, max
issue_counts = [len(datum["issues"]) for datum in data]
issue_counts_stats = np.percentile(issue_counts, [0, 25, 50, 75, 100])
print(f"Number of issues (per doc, quartiles): min={issue_counts_stats[0]}, Q1={issue_counts_stats[1]}, Q2={issue_counts_stats[2]}, Q3={issue_counts_stats[3]}, max={issue_counts_stats[4]}")
print()
# Draw issue count histogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(issue_counts, bins=range(1, max(issue_counts) + 2),
            edgecolor='black', alpha=0.7)
plt.title('Issue Count Histogram')
plt.xlabel('Number of Issues')
plt.ylabel('Frequency')
plt.xticks(range(1, max(issue_counts) + 2))
plt.grid(axis='y', alpha=0.75)
plt.savefig('issue_count_histogram.png')

# Average max depth (max number of numbers when splitting issue IDs by '.')
def max_depth(issue_id):
    if issue_id == "":
        return 0
    return len(issue_id.split("."))
max_depths = [max(max_depth(issue["id"]) for issue in datum["issues"]) for datum in data]
average_max_depth = sum(max_depths) / len(data) if data else 0
print(f"Depth of issue trees (per doc, average)  : {average_max_depth:.2f}")
# min, Q1, Q2, Q3, max
max_depths_stats = np.percentile(max_depths, [0, 25, 50, 75, 100])
print(f"Depth of issue trees (per doc, quartiles): min={max_depths_stats[0]}, Q1={max_depths_stats[1]}, Q2={max_depths_stats[2]}, Q3={max_depths_stats[3]}, max={max_depths_stats[4]}")
print()

# Collect leaf issues
leaf_issues = []
for datum in data:
    for issue in datum["issues"]:
        if issue["id"] == "":
            continue
        if not any(other_issue["id"].startswith(issue["id"] + ".") for other_issue in datum["issues"]):
            leaf_issues.append(issue)
# Leaf issue statistics
print(f"Number of leaf issues: {len(leaf_issues)}")
print()
# Average number of claims per leaf issue
leaf_claim_counts = [len(issue["claim"]) for issue in leaf_issues]
average_leaf_claims = sum(leaf_claim_counts) / len(leaf_issues) if leaf_issues else 0
print(f"Number of claims (per leaf issue, average)  : {average_leaf_claims:.2f}")
# min, Q1, Q2, Q3, max
leaf_claim_counts_stats = np.percentile(leaf_claim_counts, [0, 25, 50, 75, 100])
print(f"Number of claims (per leaf issue, quartiles): min={leaf_claim_counts_stats[0]}, Q1={leaf_claim_counts_stats[1]}, Q2={leaf_claim_counts_stats[2]}, Q3={leaf_claim_counts_stats[3]}, max={leaf_claim_counts_stats[4]}")
print()

# Average number of relevant laws per leaf issue
leaf_relevant_law_counts = [len(issue["relevant_law"]) for issue in leaf_issues]
average_leaf_relevant_law = sum(leaf_relevant_law_counts) / len(leaf_issues) if leaf_issues else 0
print(f"Number of relevant laws (per leaf issue, average)  : {average_leaf_relevant_law:.2f}")
# min, Q1, Q2, Q3, max
leaf_relevant_law_counts_stats = np.percentile(leaf_relevant_law_counts, [0, 25, 50, 75, 100])
print(f"Number of relevant laws (per leaf issue, quartiles): min={leaf_relevant_law_counts_stats[0]}, Q1={leaf_relevant_law_counts_stats[1]}, Q2={leaf_relevant_law_counts_stats[2]}, Q3={leaf_relevant_law_counts_stats[3]}, max={leaf_relevant_law_counts_stats[4]}")
print()

# Average number of analyses per leaf issue
leaf_analysis_counts = [len(issue["analysis"]) for issue in leaf_issues]
average_leaf_analysis = sum(leaf_analysis_counts) / len(leaf_issues) if leaf_issues else 0
print(f"Number of analyses (per leaf issue, average)  : {average_leaf_analysis:.2f}")
# min, Q1, Q2, Q3, max
leaf_analysis_counts_stats = np.percentile(leaf_analysis_counts, [0, 25, 50, 75, 100])
print(f"Number of analyses (per leaf issue, quartiles): min={leaf_analysis_counts_stats[0]}, Q1={leaf_analysis_counts_stats[1]}, Q2={leaf_analysis_counts_stats[2]}, Q3={leaf_analysis_counts_stats[3]}, max={leaf_analysis_counts_stats[4]}")
print()


# print("\n------ Anomal datapoints check ------")
# # Find leaf issue that has 11 claims
# for datum in data:
#     for issue in datum["issues"]:
#         if len(issue["claim"]) == 11:
#             print(f"Found anomal issue with 11 claims: {datum['doc_id']} - {issue['id']}")
#             # print(f"Claims: {issue['claim']}")
#             # print(f"Relevant law: {issue['relevant_law']}")
#             # print(f"Analysis: {issue['analysis']}")
#             # print(f"Conclusion: {issue['conclusion']}")
#             print()

# # Find leaf issue that has 65 relevant laws
# for datum in data:
#     for issue in datum["issues"]:
#         if len(issue["relevant_law"]) == 65:
#             print(f"Found anomal issue with 65 relevant laws: {datum['doc_id']} - {issue['id']}")
#             # print(f"Claims: {issue['claim']}")
#             # print(f"Relevant law: {issue['relevant_law']}")
#             # print(f"Analysis: {issue['analysis']}")
#             # print(f"Conclusion: {issue['conclusion']}")
#             print()

# # Find example
# # Issue ids should be "", "1", "1.1", "2"
# for datum in data:
#     if len(datum["issues"]) == 4:
#         issue_ids = [issue["id"] for issue in datum["issues"]]
#         if set(issue_ids) == set(["", "1", "1.1", "2"]) and len(datum["event_summary"]) < 500:
#             print(f"Found example with issue IDs='', '1', '1.1', '2': {datum['doc_id']}")
#             print(datum["event_summary"])
#             print(json.dumps(datum["issues"], indent=2, ensure_ascii=False))

# Check if issues have specific pattern in summary
# summary_pattern = re.compile(r"[^\s]+의 ([^\s]+ ){0,2}(주장|청구|항변|재항변)(에 대(한 판단|하여))")
# def is_leaf(issue, datum):
#     for x in datum["issues"]:
#         if x["id"] != issue["id"] and x["id"].startswith(issue["id"]):
#             return False
#     return True
# cnt = 0
# for datum in data:
#     for issue in datum["issues"]:
#         if summary_pattern.fullmatch(issue["summary"]) and is_leaf(issue, datum):
#             print(f"Found issue with invalid summary pattern: {datum['doc_id']} - {issue['id']}")
#             print(f"Summary: {issue['summary']}")
#             print()
#             cnt += 1
# print("Insufficient summary", cnt)


# Save the cleaned data
with open("data/legit_backup0701_filtered.jsonl", "w", encoding="utf-8") as f:
    for datum in data:
        f.write(json.dumps(datum, ensure_ascii=False) + "\n")