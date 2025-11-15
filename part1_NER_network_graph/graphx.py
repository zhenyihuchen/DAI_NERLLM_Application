import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from collections import Counter, defaultdict

def build_knowledge_graph(
    input_json,
    output_gexf="part1_NER_network_graph/results/professor_network.gexf",
    save_plot=True,
    threshold_ratio=0.15,
    teacher_sample_ratio=0.01  # <== NEW PARAMETER
):
    """
    Build a knowledge graph from merged entity results.
    Aggregates entities with <10% of max frequency before building the graph,
    and includes only a sample (default 20%) of professors to reduce clutter.
    """
    print(f"📘 Loading merged entity data from {input_json}...")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input file not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------------------------------
    # 🔹 STEP 1: Gather all entities by category (same as before)
    # ------------------------------------------------------------------
    print("🔍 Collecting entity frequencies...")
    entities_by_type = defaultdict(list)

    for prof in data:
        # Academic Experience
        for category, items in prof.get("academic_experience", {}).items():
            for val in items:
                val = val.strip()
                if val:
                    if category.lower() == "course":
                        entities_by_type["course"].append(val)
                    elif category.lower() == "program":
                        entities_by_type["program"].append(val)
                    elif category.lower() == "organization":
                        entities_by_type["university"].append(val)

        # Academic Background
        for category, items in prof.get("academic_background", {}).items():
            for val in items:
                val = val.strip()
                if val:
                    if category.lower() == "organization":
                        entities_by_type["university"].append(val)
                    elif category.lower() == "education":
                        entities_by_type["degree"].append(val)
                    elif category.lower() == "period":
                        entities_by_type["year"].append(val)
                    elif category.lower() == "location":
                        entities_by_type["location"].append(val)

        # Corporate Experience
        for category, items in prof.get("corporate_experience", {}).items():
            for val in items:
                val = val.strip()
                if val:
                    if category.lower() == "organization":
                        entities_by_type["company"].append(val)
                    elif category.lower() == "location":
                        entities_by_type["location"].append(val)

    freq_tables = {etype: Counter(vals) for etype, vals in entities_by_type.items()}
    thresholds = {
        etype: (max(cnt.values()) * threshold_ratio if cnt else 0)
        for etype, cnt in freq_tables.items()
    }

    print("📊 Frequency thresholds (min counts to keep):")
    for etype, thr in thresholds.items():
        print(f"  {etype:<10}: {thr:.2f}")

    # ------------------------------------------------------------------
    # 🔹 STEP 2: Replace rare entities (same as before)
    # ------------------------------------------------------------------
    print("🧹 Aggregating low-frequency entities before graph construction...")

    def aggregate_entity(entity, etype):
        if freq_tables[etype][entity] < thresholds[etype]:
            return f"Other_{etype.capitalize()}"
        return entity

    for prof in data:
        # Academic Experience
        for category, items in prof.get("academic_experience", {}).items():
            new_list = []
            for val in items:
                val = val.strip()
                if not val:
                    continue
                if category.lower() == "course":
                    val = aggregate_entity(val, "course")
                elif category.lower() == "program":
                    val = aggregate_entity(val, "program")
                elif category.lower() == "organization":
                    val = aggregate_entity(val, "university")
                new_list.append(val)
            prof["academic_experience"][category] = new_list

        # Academic Background
        for category, items in prof.get("academic_background", {}).items():
            new_list = []
            for val in items:
                val = val.strip()
                if not val:
                    continue
                if category.lower() == "organization":
                    val = aggregate_entity(val, "university")
                elif category.lower() == "education":
                    val = aggregate_entity(val, "degree")
                elif category.lower() == "period":
                    val = aggregate_entity(val, "year")
                elif category.lower() == "location":
                    val = aggregate_entity(val, "location")
                new_list.append(val)
            prof["academic_background"][category] = new_list

        # Corporate Experience
        for category, items in prof.get("corporate_experience", {}).items():
            new_list = []
            for val in items:
                val = val.strip()
                if not val:
                    continue
                if category.lower() == "organization":
                    val = aggregate_entity(val, "company")
                elif category.lower() == "location":
                    val = aggregate_entity(val, "location")
                new_list.append(val)
            prof["corporate_experience"][category] = new_list

    print("✅ Entities aggregated. Proceeding to graph construction...")

    # ------------------------------------------------------------------
    # 🔹 STEP 3: Sample professors (NEW)
    # ------------------------------------------------------------------
    total_profs = len(data)
    sample_size = max(1, int(total_profs * teacher_sample_ratio))
    sampled_data = random.sample(data, sample_size)
    print(f"🎯 Using {sample_size} professors out of {total_profs} ({teacher_sample_ratio*100:.0f}%) for the graph.")

    # ------------------------------------------------------------------
    # 🔹 STEP 4: Build Graph (same as before, but with sampled_data)
    # ------------------------------------------------------------------
    G = nx.DiGraph()

    for prof in sampled_data:
        prof_node = f"Prof_{prof.get('alias', f'ID_{prof.get('id', 'Unknown')}')}"
        G.add_node(prof_node, type="professor")

        # Academic Experience
        for category, items in prof.get("academic_experience", {}).items():
            for value in items:
                if not value:
                    continue
                if category.lower() == "course":
                    G.add_node(value, type="course")
                    G.add_edge(prof_node, value, relation="teaches")
                elif category.lower() == "program":
                    G.add_node(value, type="program")
                    G.add_edge(prof_node, value, relation="teaches_in_program")
                elif category.lower() == "organization":
                    G.add_node(value, type="university")
                    G.add_edge(prof_node, value, relation="teaches_at")

        # Academic Background
        for category, items in prof.get("academic_background", {}).items():
            for value in items:
                if not value:
                    continue
                if category.lower() == "organization":
                    G.add_node(value, type="university")
                    G.add_edge(prof_node, value, relation="studied_at")
                elif category.lower() == "education":
                    G.add_node(value, type="degree")
                    G.add_edge(prof_node, value, relation="has_degree")
                elif category.lower() == "period":
                    G.add_node(value, type="year")
                    G.add_edge(prof_node, value, relation="graduated_in")
                elif category.lower() == "location":
                    G.add_node(value, type="location")
                    G.add_edge(prof_node, value, relation="studied_in")

        # Corporate Experience
        for category, items in prof.get("corporate_experience", {}).items():
            for value in items:
                if not value:
                    continue
                if category.lower() == "organization":
                    G.add_node(value, type="company")
                    G.add_edge(prof_node, value, relation="worked_at")
                elif category.lower() == "location":
                    G.add_node(value, type="location")
                    G.add_edge(prof_node, value, relation="worked_in")

    print(f"✅ Graph built (sampled 20% professors): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # 🔹 STEP 5: Save & Visualize (Enhanced)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_gexf), exist_ok=True)
    nx.write_gexf(G, output_gexf)
    print(f"💾 Graph saved to: {output_gexf}")

    if save_plot:
        try:
            print("🎨 Generating enhanced visualization...")
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.6, seed=42)

            # Define colors for node types
            color_map = {
                "professor": "skyblue",
                "university": "lightgreen",
                "company": "orange",
                "course": "violet",
                "program": "turquoise",
                "degree": "pink",
                "location": "lightgray",
                "year": "beige"
            }

            # Assign colors per node
            node_colors = [color_map.get(G.nodes[n]["type"], "gray") for n in G.nodes]
            node_sizes = [max(100, 80 + 10 * G.degree(n)) for n in G.nodes]

            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.5, arrows=False)

            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.9
            )

            # Prepare labels (name + edge count for non-professors)
            labels = {}
            for n in G.nodes:
                n_type = G.nodes[n]["type"]
                if n_type == "professor":
                    labels[n] = n.replace("Prof_", "")  # just the name
                else:
                    # Show global frequency (from full dataset)
                    global_count = 0
                    for fdict in freq_tables.values():
                        if n in fdict:
                            global_count = fdict[n]
                            break
                    labels[n] = f"{n} ({global_count})"

            # Draw labels (small font, slight offset)
            nx.draw_networkx_labels(
                G, pos, labels=labels, font_size=8, font_color="black",
                verticalalignment="bottom"
            )

            # Create legend automatically
            for t, c in color_map.items():
                plt.scatter([], [], c=c, label=t, s=200, edgecolors="none")
            plt.legend(
                scatterpoints=1, frameon=False, fontsize=9,
                loc="upper right", title="Node Type", title_fontsize=10
            )

            plt.title("Knowledge Graph (Sampled Professors + Global Edge Counts)", fontsize=13)
            plt.axis("off")
            plt.tight_layout()

            output_img = "part1_NER_network_graph/results/professor_network_enhanced.png"
            plt.savefig(output_img, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"📊 Enhanced visualization saved as {output_img}")

        except Exception as e:
            print(f"⚠️ Visualization skipped: {e}")


    return G
