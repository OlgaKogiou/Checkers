import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt


METRIC_COLUMNS = [
    ("mean", "mean_similarity_pct", "mean_weight"),
    ("stddev", "std_similarity_pct", "std_weight"),
    ("skewness", "skew_similarity_pct", "skew_weight"),
    ("excess kurtosis", "kurt_similarity_pct", "kurt_weight"),
    ("median", "median_similarity_pct", "median_weight"),
    ("iqr", "iqr_similarity_pct", "iqr_weight"),
    ("K-S", "ks_similarity_pct", "ks_weight"),
]

FINGERPRINT_COLUMNS = [
    ("mean", "mean", "higher mean"),
    ("stddev", "stddev", "broader spread"),
    ("skewness", "skewness", "more asymmetric tail balance"),
    ("excess kurtosis", "excess_kurtosis", "heavier tails / more outliers"),
    ("median", "median", "higher median"),
    ("iqr", "iqr", "wider middle 50%"),
]


@dataclass
class GroupSummary:
    group_id: int
    layers: list
    representative_layer: int
    average_internal_similarity: float
    description: str
    internal_pair_count: int
    metric_breakdown: list
    reconstructed_internal_similarity: float
    raw_metric_profile: list


def aggregate_similarity(component_scores, component_weights):
    weight_sum = component_weights.sum()

    if weight_sum <= 0.0:
        return 0.0

    return float(np.dot(component_scores, component_weights) / weight_sum)


def normalize_weight_vector(values):
    total = values.sum()

    if total <= 0.0:
        return np.full(len(values), 1.0 / len(values))

    return values / total


def recommend_weights_from_sensitivity(elasticity_matrix):
    mean_elasticity = elasticity_matrix.mean(axis=0)
    recommended_weights = normalize_weight_vector(mean_elasticity)
    return mean_elasticity, recommended_weights


def compute_metric_sensitivity_cube(row_dicts, sweep_values):
    sensitivity_rows = []
    elasticity_rows = []

    for row in row_dicts:
        component_scores = np.array([
            float(row[column_name]) for _, column_name, _ in METRIC_COLUMNS
        ])
        base_weights = np.full(len(METRIC_COLUMNS), 1.0 / len(METRIC_COLUMNS))

        metric_sweeps = []
        metric_elasticities = []
        for metric_index in range(len(METRIC_COLUMNS)):
            other_weights = base_weights.copy()
            other_weights[metric_index] = 0.0
            other_sum = other_weights.sum()
            if other_sum > 0.0:
                other_weights = other_weights / other_sum

            sweep_scores = []
            for target_weight in sweep_values:
                effective_weights = other_weights * (1.0 - target_weight)
                effective_weights[metric_index] = target_weight
                sweep_scores.append(aggregate_similarity(component_scores, effective_weights))

            sweep_scores = np.asarray(sweep_scores)
            metric_sweeps.append(sweep_scores)
            metric_elasticities.append(float(sweep_scores.max() - sweep_scores.min()))

        sensitivity_rows.append(metric_sweeps)
        elasticity_rows.append(metric_elasticities)

    return np.asarray(sensitivity_rows), np.asarray(elasticity_rows)


def load_similarity_rows(csv_path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_fingerprint_rows(csv_path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_manifest_rows(csv_path):
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_recommended_weights_csv(output_path, recommendation_rows):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "state_type",
            "category",
            "metric",
            "average_elasticity",
            "recommended_weight",
        ])
        writer.writerows(recommendation_rows)


def write_group_summary_csv(output_path, summary_rows):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "state_type",
            "category",
            "group_id",
            "layer_count",
            "layers",
            "representative_layer",
            "average_internal_similarity",
            "comparison_total_time_ms",
            "grouping_time_ms",
            "description",
        ])
        writer.writerows(summary_rows)


def write_unclassified_layers(output_path, unclassified_rows):
    with output_path.open("w") as handle:
        for state_type, category, layers in unclassified_rows:
            if layers:
                handle.write(f"{state_type} :: {category}: {', '.join(str(layer) for layer in layers)}\n")
            else:
                handle.write(f"{state_type} :: {category}: none\n")


def build_fingerprint_lookup(rows):
    lookup = {}
    for row in rows:
        lookup[(row["state_type"], row["category"], int(row["layer"]))] = row
    return lookup


def build_manifest_lookup(rows, export_root):
    lookup = {}
    for row in rows:
        lookup[(row["state_type"], row["category"], int(row["layer"]))] = export_root / row["relative_path"]
    return lookup


def build_similarity_matrix(group_rows):
    layers = sorted({int(row["anchor_layer"]) for row in group_rows} | {int(row["compare_layer"]) for row in group_rows})
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    matrix = np.full((len(layers), len(layers)), np.nan)
    np.fill_diagonal(matrix, 100.0)

    for row in group_rows:
        anchor_layer = int(row["anchor_layer"])
        compare_layer = int(row["compare_layer"])
        similarity = float(row["similarity_pct"])
        anchor_index = layer_to_index[anchor_layer]
        compare_index = layer_to_index[compare_layer]
        matrix[anchor_index, compare_index] = similarity
        matrix[compare_index, anchor_index] = similarity

    return layers, matrix


def summarize_group_internal_metrics(group_layers, group_rows):
    metric_breakdown = []
    internal_rows = []
    group_layer_set = set(group_layers)

    for row in group_rows:
        anchor_layer = int(row["anchor_layer"])
        compare_layer = int(row["compare_layer"])
        if anchor_layer in group_layer_set and compare_layer in group_layer_set:
            internal_rows.append(row)

    if not internal_rows:
        return 0, [], 0.0

    reconstructed_internal_similarity = 0.0
    for metric_name, similarity_column, weight_column in METRIC_COLUMNS:
        avg_similarity = float(np.mean([float(row[similarity_column]) for row in internal_rows]))
        avg_weight = float(np.mean([float(row[weight_column]) for row in internal_rows]))
        avg_contribution = float(np.mean([
            float(row[similarity_column]) * float(row[weight_column])
            for row in internal_rows
        ]))
        reconstructed_internal_similarity += avg_contribution
        metric_breakdown.append({
            "metric": metric_name,
            "average_similarity": avg_similarity,
            "average_weight": avg_weight,
            "average_contribution": avg_contribution,
        })

    return len(internal_rows), metric_breakdown, reconstructed_internal_similarity


def summarize_group_raw_metrics(state_type, category, group_layers, all_layers, fingerprint_lookup):
    group_rows = [fingerprint_lookup[(state_type, category, layer)] for layer in group_layers]
    rest_layers = [layer for layer in all_layers if layer not in group_layers]
    rest_rows = [fingerprint_lookup[(state_type, category, layer)] for layer in rest_layers]

    raw_metric_profile = []
    for display_name, column_name, _ in FINGERPRINT_COLUMNS:
        group_values = np.array([float(row[column_name]) for row in group_rows], dtype=float)
        representative_value = float(group_values[0]) if group_values.size == 1 else float(np.mean(group_values))
        group_average = float(np.mean(group_values))

        if rest_rows:
            rest_values = np.array([float(row[column_name]) for row in rest_rows], dtype=float)
            rest_average = float(np.mean(rest_values))
        else:
            rest_average = float("nan")

        raw_metric_profile.append({
            "metric": display_name,
            "group_average": group_average,
            "rest_average": rest_average,
            "delta_vs_rest": group_average - rest_average if np.isfinite(rest_average) else float("nan"),
            "representative_value": representative_value,
        })

    return raw_metric_profile


def average_internal_similarity(matrix, layers, group_layers):
    if len(group_layers) < 2:
        return 0.0

    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    values = []
    for lhs, rhs in combinations(group_layers, 2):
        values.append(matrix[layer_to_index[lhs], layer_to_index[rhs]])
    return float(np.nanmean(values)) if values else 0.0


def bron_kerbosch(r_set, p_set, x_set, adjacency, cliques):
    if not p_set and not x_set:
        cliques.append(sorted(r_set))
        return

    for vertex in list(p_set):
        bron_kerbosch(
            r_set | {vertex},
            p_set & adjacency[vertex],
            x_set & adjacency[vertex],
            adjacency,
            cliques,
        )
        p_set.remove(vertex)
        x_set.add(vertex)


def find_similarity_groups(layers, matrix, threshold):
    adjacency = {layer: set() for layer in layers}
    layer_to_index = {layer: index for index, layer in enumerate(layers)}

    for lhs, rhs in combinations(layers, 2):
        if matrix[layer_to_index[lhs], layer_to_index[rhs]] >= threshold:
            adjacency[lhs].add(rhs)
            adjacency[rhs].add(lhs)

    cliques = []
    bron_kerbosch(set(), set(layers), set(), adjacency, cliques)
    candidate_cliques = [clique for clique in cliques if len(clique) >= 2]
    candidate_cliques.sort(
        key=lambda clique: (
            -len(clique),
            -average_internal_similarity(matrix, layers, clique),
            clique[0],
        )
    )

    assigned = set()
    chosen_groups = []
    for clique in candidate_cliques:
        if any(layer in assigned for layer in clique):
            continue
        chosen_groups.append(clique)
        assigned.update(clique)

    unclassified = [layer for layer in layers if layer not in assigned]
    return chosen_groups, unclassified


def pick_representative_layer(layers, matrix, group_layers):
    if len(group_layers) == 1:
        return group_layers[0]

    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    best_layer = group_layers[0]
    best_score = -np.inf
    for layer in group_layers:
        similarities = []
        for other_layer in group_layers:
            if layer == other_layer:
                continue
            similarities.append(matrix[layer_to_index[layer], layer_to_index[other_layer]])
        score = float(np.nanmean(similarities)) if similarities else 0.0
        if score > best_score:
            best_score = score
            best_layer = layer
    return best_layer


def describe_group(state_type, category, group_layers, all_layers, fingerprint_lookup):
    group_rows = [fingerprint_lookup[(state_type, category, layer)] for layer in group_layers]
    rest_layers = [layer for layer in all_layers if layer not in group_layers]
    if not rest_layers:
        return "This group spans all layers in the category, so there is no remaining reference set to contrast against."

    rest_rows = [fingerprint_lookup[(state_type, category, layer)] for layer in rest_layers]
    contrast_rows = []
    for display_name, column_name, semantic_hint in FINGERPRINT_COLUMNS:
        group_values = np.array([float(row[column_name]) for row in group_rows], dtype=float)
        rest_values = np.array([float(row[column_name]) for row in rest_rows], dtype=float)
        overall_values = np.concatenate([group_values, rest_values])
        scale = float(np.std(overall_values))
        delta = float(group_values.mean() - rest_values.mean())
        normalized_delta = abs(delta) / scale if scale > 1.0e-12 else abs(delta)
        direction = "higher" if delta >= 0.0 else "lower"
        if display_name == "skewness":
            semantic = f"{direction} skewness"
        elif display_name == "excess kurtosis":
            semantic = f"{direction} excess kurtosis ({semantic_hint})"
        elif display_name == "stddev":
            semantic = f"{direction} stddev ({semantic_hint})"
        elif display_name == "iqr":
            semantic = f"{direction} IQR ({semantic_hint})"
        else:
            semantic = f"{direction} {display_name}"
        contrast_rows.append((normalized_delta, semantic))

    contrast_rows.sort(key=lambda item: item[0], reverse=True)
    top_descriptors = [semantic for _, semantic in contrast_rows[:3]]
    return "This group separates from the remaining layers mainly through " + ", ".join(top_descriptors) + "."


def sampled_ecdf_curve(values, max_points=2048):
    sorted_values = np.sort(values)
    if sorted_values.size == 0:
        return np.array([]), np.array([])

    step = max(1, sorted_values.size // max_points)
    sampled_values = sorted_values[::step]
    sampled_y = np.linspace(0.0, 1.0, sampled_values.size, endpoint=True)
    return sampled_values, sampled_y


def sampled_pdf_curve(values, max_bins=256):
    if values.size == 0:
        return np.array([]), np.array([])

    if np.allclose(values.min(), values.max()):
        return np.array([values[0]]), np.array([1.0])

    bin_count = min(max_bins, max(32, int(np.sqrt(values.size))))
    density, bin_edges = np.histogram(values, bins=bin_count, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, density


def plot_group_distributions(state_type,
                             category,
                             group_summaries,
                             unclassified_layers,
                             manifest_lookup,
                             state_output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    cdf_axis, pdf_axis = axes

    for summary in group_summaries:
        representative_path = manifest_lookup[(state_type, category, summary.representative_layer)]
        representative_values = np.fromfile(representative_path, dtype=np.float32)
        cdf_x, cdf_y = sampled_ecdf_curve(representative_values)
        pdf_x, pdf_y = sampled_pdf_curve(representative_values)
        cdf_axis.plot(
            cdf_x,
            cdf_y,
            linewidth=2.0,
            label=f"group {summary.group_id} rep layer {summary.representative_layer}",
        )
        pdf_axis.plot(
            pdf_x,
            pdf_y,
            linewidth=2.0,
            label=f"group {summary.group_id} rep layer {summary.representative_layer}",
        )

    for layer in unclassified_layers:
        representative_path = manifest_lookup[(state_type, category, layer)]
        representative_values = np.fromfile(representative_path, dtype=np.float32)
        cdf_x, cdf_y = sampled_ecdf_curve(representative_values)
        pdf_x, pdf_y = sampled_pdf_curve(representative_values)
        cdf_axis.plot(
            cdf_x,
            cdf_y,
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            label=f"unclassified layer {layer}",
        )
        pdf_axis.plot(
            pdf_x,
            pdf_y,
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            label=f"unclassified layer {layer}",
        )

    cdf_axis.set_xlabel("Tensor value")
    cdf_axis.set_ylabel("Empirical CDF")
    cdf_axis.set_title("Representative group CDFs")
    cdf_axis.grid(True, alpha=0.3)
    cdf_axis.legend(fontsize=8)

    pdf_axis.set_xlabel("Tensor value")
    pdf_axis.set_ylabel("Probability density")
    pdf_axis.set_title("Representative group PDFs")
    pdf_axis.grid(True, alpha=0.3)
    pdf_axis.legend(fontsize=8)

    figure.suptitle(f"{state_type} :: {category} representative group distributions", fontsize=16)
    figure.savefig(state_output_dir / f"{category}_group_distributions.png", dpi=150)
    plt.close(figure)


def write_group_report(state_type,
                       category,
                       group_summaries,
                       unclassified_layers,
                       comparison_total_time_ms,
                       grouping_time_ms,
                       output_path):
    with output_path.open("w") as handle:
        handle.write(f"{state_type} :: {category}\n")
        handle.write(f"comparison_total_time_ms: {comparison_total_time_ms:.6f}\n")
        handle.write(f"grouping_time_ms: {grouping_time_ms:.6f}\n")
        handle.write(f"group_count: {len(group_summaries)}\n")
        handle.write(f"unclassified_layers: {', '.join(str(layer) for layer in unclassified_layers) if unclassified_layers else 'none'}\n\n")

        for summary in group_summaries:
            handle.write(f"group_{summary.group_id}:\n")
            handle.write(f"  layers: {', '.join(str(layer) for layer in summary.layers)}\n")
            handle.write(f"  layer_count: {len(summary.layers)}\n")
            handle.write(f"  internal_pair_count: {summary.internal_pair_count}\n")
            handle.write(f"  representative_layer: {summary.representative_layer}\n")
            handle.write(f"  average_internal_similarity: {summary.average_internal_similarity:.4f}\n")
            handle.write(f"  reconstructed_internal_similarity: {summary.reconstructed_internal_similarity:.4f}\n")
            handle.write("  internal_metric_breakdown:\n")
            for metric_row in summary.metric_breakdown:
                handle.write(
                    "    "
                    f"{metric_row['metric']}: similarity={metric_row['average_similarity']:.4f}, "
                    f"weight={metric_row['average_weight']:.6f}, "
                    f"weighted_contribution={metric_row['average_contribution']:.4f}\n"
                )
            handle.write("  raw_metric_profile:\n")
            for metric_row in summary.raw_metric_profile:
                rest_average = (
                    f"{metric_row['rest_average']:.6g}"
                    if np.isfinite(metric_row['rest_average'])
                    else "n/a"
                )
                delta_vs_rest = (
                    f"{metric_row['delta_vs_rest']:.6g}"
                    if np.isfinite(metric_row['delta_vs_rest'])
                    else "n/a"
                )
                handle.write(
                    "    "
                    f"{metric_row['metric']}: group_average={metric_row['group_average']:.6g}, "
                    f"representative_value={metric_row['representative_value']:.6g}, "
                    f"rest_average={rest_average}, "
                    f"delta_vs_rest={delta_vs_rest}\n"
                )
            handle.write("  note: K-S is a pairwise distribution-comparison metric, so it appears in the internal similarity breakdown above rather than as a per-tensor raw scalar here.\n")
            handle.write(f"  description: {summary.description}\n\n")


def plot_similarity_groups(rows, fingerprint_rows, manifest_rows, export_root, output_dir, similarity_threshold):
    grouped_rows = defaultdict(list)
    for row in rows:
        key = (row["state_type"], row["category"])
        grouped_rows[key].append(row)

    recommendation_rows = []
    grouping_summary_rows = []
    unclassified_rows = []
    fingerprint_lookup = build_fingerprint_lookup(fingerprint_rows)
    manifest_lookup = build_manifest_lookup(manifest_rows, export_root)

    for (state_type, category), group_rows in grouped_rows.items():
        ordered_rows = sorted(group_rows, key=lambda row: (int(row["anchor_layer"]), int(row["compare_layer"])))
        layers, similarity_matrix = build_similarity_matrix(ordered_rows)

        state_output_dir = output_dir / state_type
        state_output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 7))
        image = plt.imshow(similarity_matrix, vmin=0.0, vmax=100.0, cmap="viridis")
        plt.colorbar(image, label="overall similarity")
        plt.xticks(range(len(layers)), layers)
        plt.yticks(range(len(layers)), layers)
        plt.xlabel("Layer")
        plt.ylabel("Layer")
        plt.title(f"{state_type} :: {category} all-to-all similarity matrix")
        plt.tight_layout()

        plot_path = state_output_dir / f"{category}_similarity.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        weight_sweep = np.linspace(0.0, 1.0, 41)
        sensitivity_cube, elasticity_matrix = compute_metric_sensitivity_cube(ordered_rows, weight_sweep)

        figure, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
        axes = axes.flatten()
        for metric_index, (metric_name, _, _) in enumerate(METRIC_COLUMNS):
            axis = axes[metric_index]
            metric_heatmap = sensitivity_cube[:, metric_index, :]
            image = axis.imshow(
                metric_heatmap,
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=100.0,
                cmap="viridis",
            )
            axis.set_title(metric_name)
            axis.set_xlabel("Metric weight sweep")
            axis.set_ylabel("Pair index")

        axes[len(METRIC_COLUMNS)].axis("off")
        figure.colorbar(image, ax=axes[: len(METRIC_COLUMNS)], label="overall similarity")
        figure.suptitle(f"{state_type} :: {category} global component sensitivity", fontsize=16)
        global_heatmap_path = state_output_dir / f"{category}_global_sensitivity_heatmaps.png"
        figure.savefig(global_heatmap_path, dpi=150)
        plt.close(figure)

        plt.figure(figsize=(12, 7))
        elasticity_image = plt.imshow(
            elasticity_matrix.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
        )
        plt.yticks(range(len(METRIC_COLUMNS)), [metric_name for metric_name, _, _ in METRIC_COLUMNS])
        plt.colorbar(elasticity_image, label="overall similarity range across sweep")
        plt.xlabel("Pair index")
        plt.ylabel("Metric")
        plt.title(f"{state_type} :: {category} elasticity matrix")
        plt.tight_layout()
        elasticity_path = state_output_dir / f"{category}_elasticity_matrix.png"
        plt.savefig(elasticity_path, dpi=150)
        plt.close()

        mean_elasticity, recommended_weights = recommend_weights_from_sensitivity(elasticity_matrix)

        plt.figure(figsize=(12, 7))
        plt.bar([metric_name for metric_name, _, _ in METRIC_COLUMNS], mean_elasticity)
        plt.ylabel("Average similarity range across sweep")
        plt.xlabel("Metric whose weight is swept")
        plt.title(f"{state_type} :: {category} average metric sensitivity")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        summary_path = state_output_dir / f"{category}_metric_sensitivity_summary.png"
        plt.savefig(summary_path, dpi=150)
        plt.close()

        plt.figure(figsize=(12, 7))
        plt.bar(
            [metric_name for metric_name, _, _ in METRIC_COLUMNS],
            recommended_weights,
        )
        plt.ylabel("Recommended weight from sensitivity")
        plt.xlabel("Metric")
        plt.title(f"{state_type} :: {category} recommended metric weights")
        plt.ylim(0.0, 1.0)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        recommendation_plot_path = state_output_dir / f"{category}_recommended_weights.png"
        plt.savefig(recommendation_plot_path, dpi=150)
        plt.close()

        for metric_index, (metric_name, _, _) in enumerate(METRIC_COLUMNS):
            recommendation_rows.append([
                state_type,
                category,
                metric_name,
                float(mean_elasticity[metric_index]),
                float(recommended_weights[metric_index]),
            ])

        grouping_start = time.perf_counter()
        chosen_groups, unclassified_layers = find_similarity_groups(layers, similarity_matrix, similarity_threshold)
        grouping_time_ms = (time.perf_counter() - grouping_start) * 1000.0
        comparison_total_time_ms = sum(
            float(row["ks_sort_time_ms"]) +
            float(row["ks_gap_time_ms"]) +
            float(row["comparison_time_ms"])
            for row in ordered_rows
        )

        group_summaries = []
        for group_id, group_layers in enumerate(chosen_groups, start=1):
            representative_layer = pick_representative_layer(layers, similarity_matrix, group_layers)
            internal_pair_count, metric_breakdown, reconstructed_internal_similarity = summarize_group_internal_metrics(
                group_layers,
                ordered_rows,
            )
            raw_metric_profile = summarize_group_raw_metrics(
                state_type,
                category,
                group_layers,
                layers,
                fingerprint_lookup,
            )
            group_summaries.append(GroupSummary(
                group_id=group_id,
                layers=group_layers,
                representative_layer=representative_layer,
                average_internal_similarity=average_internal_similarity(similarity_matrix, layers, group_layers),
                description=describe_group(state_type, category, group_layers, layers, fingerprint_lookup),
                internal_pair_count=internal_pair_count,
                metric_breakdown=metric_breakdown,
                reconstructed_internal_similarity=reconstructed_internal_similarity,
                raw_metric_profile=raw_metric_profile,
            ))

        plot_group_distributions(
            state_type,
            category,
            group_summaries,
            unclassified_layers,
            manifest_lookup,
            state_output_dir,
        )
        write_group_report(
            state_type,
            category,
            group_summaries,
            unclassified_layers,
            comparison_total_time_ms,
            grouping_time_ms,
            state_output_dir / f"{category}_group_report.txt",
        )

        for summary in group_summaries:
            grouping_summary_rows.append([
                state_type,
                category,
                summary.group_id,
                len(summary.layers),
                " ".join(str(layer) for layer in summary.layers),
                summary.representative_layer,
                summary.average_internal_similarity,
                comparison_total_time_ms,
                grouping_time_ms,
                summary.description,
            ])

        unclassified_rows.append((state_type, category, unclassified_layers))

    write_recommended_weights_csv(output_dir / "recommended_metric_weights.csv", recommendation_rows)
    write_group_summary_csv(output_dir / "grouping_summary.csv", grouping_summary_rows)
    write_unclassified_layers(output_dir / "unclassified_layers.txt", unclassified_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TensorAnalyzer layer similarity results from the analyzer CSV output."
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="Directory containing TensorAnalyzer's pairwise_similarity.csv output.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=85.0,
        help="Minimum overall similarity required for two layers to belong to the same group.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    csv_path = analysis_dir / "pairwise_similarity.csv"
    fingerprints_path = analysis_dir / "fingerprints.csv"
    manifest_path = analysis_dir.parent / "manifest.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing similarity CSV: {csv_path}")
    if not fingerprints_path.exists():
        raise FileNotFoundError(f"Missing fingerprints CSV: {fingerprints_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest CSV: {manifest_path}")

    rows = load_similarity_rows(csv_path)
    fingerprint_rows = load_fingerprint_rows(fingerprints_path)
    manifest_rows = load_manifest_rows(manifest_path)
    plot_output_dir = analysis_dir / "plots"
    plot_similarity_groups(
        rows,
        fingerprint_rows,
        manifest_rows,
        analysis_dir.parent,
        plot_output_dir,
        args.similarity_threshold,
    )
    print(f"Wrote plots to {plot_output_dir}")


if __name__ == "__main__":
    main()