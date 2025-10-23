import json
from networkx import radius
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Any
from pandas import read_csv, isna
plt.rcParams['svg.fonttype'] = 'none'

data = read_csv("data/LEGIT_casenames.csv")
print(data)

case_type_counter = defaultdict(int)
case_subtype_counter = defaultdict(int)
case_name_counter = defaultdict(int)
raw_casename_set = set()
for _, row in data.iterrows():
    raw_casename_set.add(row["case_name_ko"])
    case_type_counter[row["case_type"]] += row["count"]
    if isna(row["case_subtype"]) or "etc" in row["case_subtype"].lower() or "confirmation" in row["case_subtype"].lower() or "기타" in row["case_name_ko"]:
        row["case_subtype"] = "etc."
        row["case_name_ko"] = ""
    case_subtype_counter[(row["case_type"], row["case_subtype"])] += row["count"]
    case_name_counter[(row["case_type"], row["case_subtype"], row["case_name_ko"])] += row["count"]

# Draw pie chart for case_name_counter.
# Group by case_types, sort in descending order.
# For each case_type, group by case_subtypes in descending order.
# For each case_subtype, plot all case_names in descending order. Label only ones with count >= 30.
case_name_order = sorted(case_name_counter.keys(), key=lambda x: (case_type_counter[x[0]], 1 if x[1] != "etc." else 0, case_subtype_counter[(x[0], x[1])], case_name_counter[x]), reverse=True)
case_name_labels = [f"{name} ({count})" if count >= 200 and name else "" for i, (type_, subtype, name), count in zip(range(len(case_name_order)), case_name_order, [case_name_counter[x] for x in case_name_order])]
case_name_sizes = [case_name_counter[x] for x in case_name_order]
# each subtype has different alphas, decreasing from 1.0 in first of case_type and multiplying by 0.8 for each subsequent subtype.
case_name_alphas = []
current_alpha = 1.0
for i, (type_, subtype, name) in enumerate(case_name_order):
    if i > 0:
        prev_type, prev_subtype, prev_name = case_name_order[i-1]
        if type_ != prev_type:
            current_alpha = 1.0
        elif subtype != prev_subtype:
            current_alpha *= 0.7
    case_name_alphas.append(current_alpha)
# plot pie chart
plt.figure(figsize=(8, 6))
# adjust colors based on alpha, smaller alpha -> lighter color
colors = [None for _ in range(len(case_name_order))]
for i, (type_, subtype, name) in enumerate(case_name_order):
    high_level_colors = {
        "civil": np.array([0x98/256, 0xc1/256, 0x26/256, 1]),
        "administration": np.array([0x00/256, 0xaf/256, 0xbd/256, 1]),
    }
    colors[i] = high_level_colors.get(type_, np.array([1, 1, 1, 1])) * np.array([1, 1, 1, case_name_alphas[i]])
# Make labels be perpendicular along the pie chart, and make it closer to the pie chart.
wedges, texts = plt.pie(case_name_sizes, labels=case_name_labels, colors=colors, startangle=90, counterclock=False, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}, textprops={'fontsize': 8})
for wedge, text in zip(wedges, texts):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))
    horizontalalignment = 'left' if x > 0 else 'right'
    connectionstyle = "angle,angleA=0,angleB={}".format(angle)
    text.set_horizontalalignment(horizontalalignment)
    text.set_rotation(angle if x > 0 else angle + 180)
    text.set_rotation_mode('anchor')
    text.set_position((1.0 * x, 1.0 * y))

plt.tight_layout()
plt.savefig("plots/legit_casename_distribution.svg")

print(f"civil: {case_type_counter['civil'] / sum(case_type_counter.values()) * 100}%, administration: {case_type_counter['administration'] / sum(case_type_counter.values()) * 100}%")
for (type_, subtype), count in case_subtype_counter.items():
    print(f"{type_} - {subtype}: {count / sum(case_type_counter.values()) * 100}%")

# Calculate the bisector angle for each case_subtype.
# Find the end angle of first and last wedges in the same case_subtype, and calculate the bisector angle.
case_subtype_bisectors = {}
for i, (type_, subtype, name) in enumerate(case_name_order):
    if (type_, subtype) not in case_subtype_bisectors:
        # find the first wedge with the same case_subtype
        for j in range(i, -1, -1):
            if case_name_order[j][0] == type_ and case_name_order[j][1] == subtype:
                first_wedge = j
                break
        # find the last wedge with the same case_subtype
        for j in range(i, len(case_name_order)):
            if case_name_order[j][0] == type_ and case_name_order[j][1] == subtype:
                last_wedge = j
        # calculate the bisector angle
        first_wedge_angle = (wedges[first_wedge].theta2 + wedges[first_wedge].theta1) / 2
        last_wedge_angle = (wedges[last_wedge].theta2 + wedges[last_wedge].theta1) / 2
        bisector_angle = (first_wedge_angle + last_wedge_angle) / 2
        case_subtype_bisectors[(type_, subtype)] = bisector_angle
# print the bisector angles
for (type_, subtype), angle in case_subtype_bisectors.items():
    if angle < -92:
        angle += 180
    print(f"{type_} - {subtype}: {angle} degrees")

# Print how many cases that have more than 200 instance, more than 10 instances
print("Total cases:", sum(case_name_counter.values()))
print(f"Number of case names with more than 200 instances: {sum(1 for count in case_name_counter.values() if count >= 200)}")
print(f"Number of case names with more than 10 instances: {sum(1 for count in case_name_counter.values() if count > 10)}")
print(f"Number of unique raw case names: {len(raw_casename_set)}")