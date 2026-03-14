import json
import csv
import ast
from pathlib import Path
from collections import Counter
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ===================== YOUR PATHS (hardcoded) =====================
# I will point to the AudioSet label CSV file.
LABEL_CSV = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/egs/audioset/data/class_labels_indices.csv")
# I will point to the training manifest JSON file.
TRAIN_JSON = Path("/WAVE/projects/oignat_lab/Parth-Personal/Dataset/balanced_train_converted.json")
# I will define the output directory where the heatmap will be saved.
OUT_DIR = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/data_viz")
# ================================================================


def read_label_map(label_csv_path: Path):
    # I will map class index to display name.
    idx_to_name = {}
    # I will map class MID to class index.
    mid_to_idx = {}
    # I will open the CSV and read rows as dicts.
    with label_csv_path.open("r", newline="", encoding="utf-8") as f:
        # I will create a DictReader for the CSV.
        reader = csv.DictReader(f)
        # I will iterate over each row in the label CSV.
        for row in reader:
            # I will extract the numeric index field (handles different column casing).
            if "index" in row:
                idx = int(row["index"])
            elif "Index" in row:
                idx = int(row["Index"])
            else:
                continue

            # I will read the MID field if present.
            mid = row.get("mid") or row.get("MID") or row.get("Mid") or ""
            # I will read the display name field with fallbacks.
            name = row.get("display_name") or row.get("DisplayName") or row.get("name") or str(mid) or str(idx)

            # I will store the mapping from index to name.
            idx_to_name[idx] = name
            # I will store the mapping from MID to index when MID exists.
            if mid:
                mid_to_idx[mid] = idx

    # I will return the two mappings for later use.
    return idx_to_name, mid_to_idx


def _maybe_extract_list(obj):
    # I will return the object directly if it is already a list.
    if isinstance(obj, list):
        return obj
    # I will unwrap common wrapper keys if this is a dict container.
    if isinstance(obj, dict):
        for k in ["data", "items", "audios", "clips", "entries"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    # I will return the original object if nothing matches.
    return obj


def load_manifest_any(path: Path):
    # I will read the entire file as text.
    text = path.read_text(encoding="utf-8").strip()
    # I will return empty if the file is empty.
    if not text:
        return []

    # I will try parsing the full file as JSON.
    try:
        obj = json.loads(text)
        obj = _maybe_extract_list(obj)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # I will try parsing the full file as a Python literal.
    try:
        obj = ast.literal_eval(text)
        obj = _maybe_extract_list(obj)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # I will fall back to parsing line-by-line as JSONL or Python literals.
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
                continue
            except Exception:
                pass
            try:
                items.append(ast.literal_eval(line))
                continue
            except Exception:
                continue

    # I will return all parsed items.
    return items


def extract_labels(item: dict, mid_to_idx: dict):
    # I will find the first matching label key in the item.
    for key in ["labels", "label", "target", "targets", "class_indices", "classes"]:
        if key in item:
            raw = item[key]
            break
    else:
        return []

    # I will normalize the raw value into a list.
    if isinstance(raw, list):
        vals = raw
    elif isinstance(raw, (tuple, set)):
        vals = list(raw)
    elif isinstance(raw, str):
        vals = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        return []

    # I will convert labels to integer class indices.
    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str):
            if v in mid_to_idx:
                out.append(mid_to_idx[v])
            else:
                try:
                    out.append(int(v))
                except ValueError:
                    pass
        else:
            try:
                out.append(int(v))
            except Exception:
                pass

    # I will deduplicate labels while preserving order.
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    # I will return unique label indices for the clip.
    return uniq


def compute_label_freq_and_per_clip(items, mid_to_idx):
    # I will count overall label frequencies.
    label_freq = Counter()
    # I will store per-clip label lists for co-occurrence computation.
    per_clip_labels = []

    # I will iterate through all manifest items.
    for it in items:
        # I will extract the label list for this item.
        labs = extract_labels(it, mid_to_idx)
        # I will store it for later co-occurrence counting.
        per_clip_labels.append(labs)
        # I will update frequency counts if labels exist.
        if labs:
            label_freq.update(labs)

    # I will return frequency and per-clip labels.
    return label_freq, per_clip_labels


def shorten_name(name: str, max_len: int = 18):
    # I will trim whitespace from the class name.
    name = name.strip()
    # I will shorten long names to keep axis labels readable.
    return name if len(name) <= max_len else (name[: max_len - 3] + "...")


def plot_cooccurrence_heatmap(per_clip_labels, label_freq: Counter, idx_to_name: dict, out_path: Path, top_k=5):
    # I will pick the top-K labels by frequency.
    top = [i for i, _ in label_freq.most_common(top_k)]
    # I will stop if there are not enough classes to form pairs.
    if len(top) < 2:
        return

    # I will create a mapping from label index to matrix position.
    idx_pos = {idx: p for p, idx in enumerate(top)}
    # I will allocate the co-occurrence matrix.
    mat = np.zeros((len(top), len(top)), dtype=np.int64)

    # I will count co-occurrences among the top-K labels per clip.
    for labs in per_clip_labels:
        s = [x for x in labs if x in idx_pos]
        if len(s) < 2:
            continue
        for a, b in itertools.combinations(sorted(set(s)), 2):
            ia, ib = idx_pos[a], idx_pos[b]
            mat[ia, ib] += 1
            mat[ib, ia] += 1

    # I will zero out the diagonal since self-co-occurrence is not needed.
    np.fill_diagonal(mat, 0)

    # I will create shortened names for axis ticks.
    names = [shorten_name(idx_to_name.get(i, str(i)), 18) for i in top]

    # I will compute vmax for scaling.
    vmax = int(mat.max()) if mat.size else 0
    if vmax < 1:
        vmax = 1

    # I will define bin edges and truncate them based on vmax.
    base_bounds = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    bounds = [b for b in base_bounds if b <= vmax] + [vmax + 1]
    n_bins = len(bounds) - 1

    # I will build a solid discrete palette with enough colors for all bins.
    palette = [
        "#0b0b0b", "#1b1b5a", "#2b1055", "#5a189a", "#7b2cbf",
        "#9d4edd", "#c77dff", "#f72585", "#ff7b00", "#ffd166",
        "#caffbf", "#2a9d8f", "#e9ecef"
    ]
    # I will ensure we have at least n_bins colors (repeat if needed).
    colors = (palette * ((n_bins // len(palette)) + 1))[:n_bins]

    # I will construct the colormap and matching boundary normalization.
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # I will set bold fonts for readability.
    plt.rcParams.update({
        "font.size": 12,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    # I will plot with tight layout and minimal whitespace.
    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")

    # I will add a colorbar and bolden its text.
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Co-occurrence count (binned)", fontweight="bold")
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    # I will set axis ticks bold and readable.
    plt.xticks(range(len(names)), names, rotation=90, fontsize=11, fontweight="bold")
    plt.yticks(range(len(names)), names, fontsize=11, fontweight="bold")

    # I will set a bold title.
    plt.title(f"Co-occurrence Heatmap (Top {top_k} classes)", fontsize=14, fontweight="bold")

    # I will save with almost no outer whitespace.
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    # I will ensure the output directory exists.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # I will read label mappings from the CSV.
    idx_to_name, mid_to_idx = read_label_map(LABEL_CSV)

    # I will load the training manifest.
    train_items = load_manifest_any(TRAIN_JSON)

    # I will compute label frequencies and per-clip labels.
    label_freq, per_clip_labels = compute_label_freq_and_per_clip(train_items, mid_to_idx)

    # I will define the output file path for the heatmap.
    out_path = OUT_DIR / "train_cooccurrence_top5_heatmap_solid.png"

    # I will plot and save the top-5 co-occurrence heatmap.
    plot_cooccurrence_heatmap(per_clip_labels, label_freq, idx_to_name, out_path, top_k=5)

    # I will print the saved path for confirmation.
    print(f"Saved heatmap into: {out_path}")


if __name__ == "__main__":
    main()









# import json
# import csv
# import ast
# from pathlib import Path
# from collections import Counter
# import itertools

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm


# # ===================== YOUR PATHS (hardcoded) =====================
# LABEL_CSV = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/egs/audioset/data/class_labels_indices.csv")
# TRAIN_JSON = Path("/WAVE/projects/oignat_lab/Parth-Personal/Dataset/balanced_train_converted.json")

# OUT_DIR = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/data_viz")
# # ================================================================


# def read_label_map(label_csv_path: Path):
#     idx_to_name = {}
#     mid_to_idx = {}
#     with label_csv_path.open("r", newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             # I will parse the class index from the CSV.
#             if "index" in row:
#                 idx = int(row["index"])
#             elif "Index" in row:
#                 idx = int(row["Index"])
#             else:
#                 continue

#             # I will read the class MID and display name (fallback to MID/index).
#             mid = row.get("mid") or row.get("MID") or row.get("Mid") or ""
#             name = row.get("display_name") or row.get("DisplayName") or row.get("name") or str(mid) or str(idx)

#             # I will store index->name and MID->index mappings.
#             idx_to_name[idx] = name
#             if mid:
#                 mid_to_idx[mid] = idx

#     # I will return both mappings for later label decoding.
#     return idx_to_name, mid_to_idx


# def _maybe_extract_list(obj):
#     # I will unwrap common wrapper keys if the JSON is nested.
#     if isinstance(obj, list):
#         return obj
#     if isinstance(obj, dict):
#         for k in ["data", "items", "audios", "clips", "entries"]:
#             if k in obj and isinstance(obj[k], list):
#                 return obj[k]
#     return obj


# def load_manifest_any(path: Path):
#     # I will read the file text first.
#     text = path.read_text(encoding="utf-8").strip()
#     if not text:
#         return []

#     # I will try parsing as JSON first.
#     try:
#         obj = json.loads(text)
#         obj = _maybe_extract_list(obj)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass

#     # I will try parsing as a Python literal as a fallback.
#     try:
#         obj = ast.literal_eval(text)
#         obj = _maybe_extract_list(obj)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass

#     # I will finally try JSONL / line-by-line parsing.
#     items = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 items.append(json.loads(line))
#                 continue
#             except Exception:
#                 pass
#             try:
#                 items.append(ast.literal_eval(line))
#                 continue
#             except Exception:
#                 continue
#     return items


# def extract_labels(item: dict, mid_to_idx: dict):
#     # I will locate the label field in a flexible way.
#     for key in ["labels", "label", "target", "targets", "class_indices", "classes"]:
#         if key in item:
#             raw = item[key]
#             break
#     else:
#         return []

#     # I will normalize raw labels into a list of tokens.
#     if isinstance(raw, list):
#         vals = raw
#     elif isinstance(raw, (tuple, set)):
#         vals = list(raw)
#     elif isinstance(raw, str):
#         vals = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
#     else:
#         return []

#     # I will convert MIDs or numeric strings into integer indices.
#     out = []
#     for v in vals:
#         if isinstance(v, int):
#             out.append(v)
#         elif isinstance(v, str):
#             if v in mid_to_idx:
#                 out.append(mid_to_idx[v])
#             else:
#                 try:
#                     out.append(int(v))
#                 except ValueError:
#                     pass
#         else:
#             try:
#                 out.append(int(v))
#             except Exception:
#                 pass

#     # I will deduplicate while preserving order.
#     seen = set()
#     uniq = []
#     for x in out:
#         if x not in seen:
#             seen.add(x)
#             uniq.append(x)

#     # I will return the final unique label index list for this clip.
#     return uniq


# def compute_label_freq_and_per_clip(items, mid_to_idx):
#     # I will compute label frequencies and store per-clip label lists.
#     label_freq = Counter()
#     per_clip_labels = []

#     for it in items:
#         # I will extract labels for the current item.
#         labs = extract_labels(it, mid_to_idx)
#         # I will store labels for later co-occurrence counting.
#         per_clip_labels.append(labs)
#         # I will update overall frequency counts.
#         if labs:
#             label_freq.update(labs)

#     # I will return frequency and per-clip labels.
#     return label_freq, per_clip_labels


# def shorten_name(name: str, max_len: int = 18):
#     # I will shorten long class names for cleaner axis labels.
#     name = name.strip()
#     return name if len(name) <= max_len else (name[: max_len - 3] + "...")


# def plot_cooccurrence_heatmap(per_clip_labels, label_freq: Counter, idx_to_name: dict, out_path: Path, top_k=10):
#     # I will pick the top-K most frequent labels to include in the heatmap.
#     top = [i for i, _ in label_freq.most_common(top_k)]
#     if len(top) < 2:
#         return

#     # I will build an index-to-position lookup for the top labels.
#     idx_pos = {idx: p for p, idx in enumerate(top)}
#     # I will create a square matrix for co-occurrence counts.
#     mat = np.zeros((len(top), len(top)), dtype=np.int64)

#     # I will count co-occurrences within each clip for labels in the top-K set.
#     for labs in per_clip_labels:
#         s = [x for x in labs if x in idx_pos]
#         if len(s) < 2:
#             continue
#         for a, b in itertools.combinations(sorted(set(s)), 2):
#             ia, ib = idx_pos[a], idx_pos[b]
#             mat[ia, ib] += 1
#             mat[ib, ia] += 1

#     # I will remove diagonal entries since self-co-occurrence is not needed.
#     np.fill_diagonal(mat, 0)

#     # I will prepare shortened axis labels.
#     names = [shorten_name(idx_to_name.get(i, str(i)), 18) for i in top]

#     # I will set log-scale bounds for a stable visualization.
#     vmax = int(mat.max()) if mat.size else 1
#     if vmax <= 1:
#         vmax = 2

#     # I will render and save the heatmap with log normalization.
#     plt.figure(figsize=(10, 9))
#     plt.imshow(
#         mat,
#         aspect="auto",
#         cmap="magma",
#         norm=LogNorm(vmin=1, vmax=vmax),
#         interpolation="nearest",
#     )
#     plt.colorbar(label="Co-occurrence count (log)")
#     plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
#     plt.yticks(range(len(names)), names, fontsize=8)
#     plt.title(f"Co-occurrence Heatmap (Top {top_k} classes, log-scaled)")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def main():
#     # I will create the output directory if it does not exist.
#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     # I will read label mappings from the AudioSet CSV.
#     idx_to_name, mid_to_idx = read_label_map(LABEL_CSV)

#     # I will load the training manifest items.
#     train_items = load_manifest_any(TRAIN_JSON)

#     # I will compute label frequencies and per-clip label lists.
#     label_freq, per_clip_labels = compute_label_freq_and_per_clip(train_items, mid_to_idx)

#     # I will save the co-occurrence heatmap for top-10 classes.
#     out_path = OUT_DIR / "train_cooccurrence_top10_heatmap_log.png"
#     plot_cooccurrence_heatmap(per_clip_labels, label_freq, idx_to_name, out_path, top_k=10)

#     # I will print where the image was saved.
#     print(f"Saved heatmap into: {out_path}")


# if __name__ == "__main__":
#     main()


# import json
# import csv
# import ast
# import math
# from pathlib import Path
# from collections import Counter
# import itertools
# import textwrap

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm


# # ===================== YOUR PATHS (hardcoded) =====================
# LABEL_CSV = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/egs/audioset/data/class_labels_indices.csv")
# TRAIN_JSON = Path("/WAVE/projects/oignat_lab/Parth-Personal/Dataset/balanced_train_converted.json")
# EVAL_JSON  = Path("/WAVE/projects/oignat_lab/Parth-Personal/Dataset/eval_converted_clean.json")

# OUT_DIR = Path("/WAVE/projects/oignat_lab/Parth-Personal/AudioProj/DTFAT/data_viz")
# # ================================================================


# def read_label_map(label_csv_path: Path):
#     idx_to_name = {}
#     mid_to_idx = {}
#     with label_csv_path.open("r", newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if "index" in row:
#                 idx = int(row["index"])
#             elif "Index" in row:
#                 idx = int(row["Index"])
#             else:
#                 continue

#             mid = row.get("mid") or row.get("MID") or row.get("Mid") or ""
#             name = row.get("display_name") or row.get("DisplayName") or row.get("name") or str(mid) or str(idx)

#             idx_to_name[idx] = name
#             if mid:
#                 mid_to_idx[mid] = idx
#     return idx_to_name, mid_to_idx


# def _maybe_extract_list(obj):
#     if isinstance(obj, list):
#         return obj
#     if isinstance(obj, dict):
#         for k in ["data", "items", "audios", "clips", "entries"]:
#             if k in obj and isinstance(obj[k], list):
#                 return obj[k]
#     return obj


# def load_manifest_any(path: Path):
#     text = path.read_text(encoding="utf-8").strip()
#     if not text:
#         return []

#     try:
#         obj = json.loads(text)
#         obj = _maybe_extract_list(obj)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass

#     try:
#         obj = ast.literal_eval(text)
#         obj = _maybe_extract_list(obj)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass

#     items = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 items.append(json.loads(line))
#                 continue
#             except Exception:
#                 pass
#             try:
#                 items.append(ast.literal_eval(line))
#                 continue
#             except Exception:
#                 continue
#     return items


# def extract_labels(item: dict, mid_to_idx: dict):
#     for key in ["labels", "label", "target", "targets", "class_indices", "classes"]:
#         if key in item:
#             raw = item[key]
#             break
#     else:
#         return []

#     if isinstance(raw, list):
#         vals = raw
#     elif isinstance(raw, (tuple, set)):
#         vals = list(raw)
#     elif isinstance(raw, str):
#         vals = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
#     else:
#         return []

#     out = []
#     for v in vals:
#         if isinstance(v, int):
#             out.append(v)
#         elif isinstance(v, str):
#             if v in mid_to_idx:
#                 out.append(mid_to_idx[v])
#             else:
#                 try:
#                     out.append(int(v))
#                 except ValueError:
#                     pass
#         else:
#             try:
#                 out.append(int(v))
#             except Exception:
#                 pass

#     seen = set()
#     uniq = []
#     for x in out:
#         if x not in seen:
#             seen.add(x)
#             uniq.append(x)
#     return uniq


# def extract_duration(item: dict):
#     for key in ["duration", "dur", "clip_duration", "audio_duration", "seconds"]:
#         if key in item:
#             try:
#                 return float(item[key])
#             except Exception:
#                 return None
#     if "start" in item and "end" in item:
#         try:
#             return float(item["end"]) - float(item["start"])
#         except Exception:
#             return None
#     return None


# def extract_sample_rate(item: dict):
#     for key in ["sample_rate", "sr", "sampling_rate", "fs"]:
#         if key in item:
#             try:
#                 return int(item[key])
#             except Exception:
#                 return None
#     return None


# def entropy_from_counts(counts: np.ndarray):
#     if counts.size == 0:
#         return 0.0
#     p = counts / counts.sum()
#     p = p[p > 0]
#     return float(-(p * np.log(p)).sum())


# def gini_from_counts(counts: np.ndarray):
#     if counts.size == 0:
#         return 0.0
#     x = np.sort(counts.astype(float))
#     if x.sum() == 0:
#         return 0.0
#     n = x.size
#     cumx = np.cumsum(x)
#     g = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
#     return float(g)


# def compute_stats(items, mid_to_idx):
#     label_freq = Counter()
#     label_cardinality = []
#     durations = []
#     sample_rates = []
#     per_clip_labels = []

#     for it in items:
#         labs = extract_labels(it, mid_to_idx)
#         per_clip_labels.append(labs)
#         label_cardinality.append(len(labs))
#         if labs:
#             label_freq.update(labs)

#         d = extract_duration(it)
#         if d is not None and d > 0:
#             durations.append(d)

#         sr = extract_sample_rate(it)
#         if sr is not None and sr > 0:
#             sample_rates.append(sr)

#     label_cardinality = np.array(label_cardinality, dtype=int)
#     durations = np.array(durations, dtype=float)
#     sample_rates = np.array(sample_rates, dtype=int)

#     freq_vals = np.array(list(label_freq.values()), dtype=float)
#     ent = entropy_from_counts(freq_vals) if freq_vals.size else 0.0
#     gini = gini_from_counts(freq_vals) if freq_vals.size else 0.0

#     return {
#         "n_items": len(items),
#         "label_freq": label_freq,
#         "label_cardinality": label_cardinality,
#         "durations": durations,
#         "sample_rates": sample_rates,
#         "per_clip_labels": per_clip_labels,
#         "entropy": ent,
#         "gini": gini,
#     }


# def shorten_name(name: str, max_len: int = 22):
#     name = name.strip()
#     return name if len(name) <= max_len else (name[: max_len - 3] + "...")


# def plot_top_class_freq_hbar(label_freq: Counter, idx_to_name: dict, out_path: Path, mapping_path: Path, top_k=20):
#     top = label_freq.most_common(top_k)
#     if not top:
#         return

#     idxs = [i for i, _ in top][::-1]
#     vals = [c for _, c in top][::-1]

#     full_names = [idx_to_name.get(i, str(i)) for i in idxs]
#     short_names = [shorten_name(n, 26) for n in full_names]

#     with mapping_path.open("w", encoding="utf-8") as f:
#         f.write("Top class label mapping (short -> full)\n")
#         for s, full in zip(short_names[::-1], full_names[::-1]):
#             f.write(f"{s}\t{full}\n")

#     plt.figure(figsize=(11, 7))
#     plt.barh(range(len(vals)), vals)
#     plt.xscale("log")
#     plt.yticks(range(len(vals)), short_names)
#     plt.title(f"Top {top_k} Class Frequencies (log x-scale)")
#     plt.xlabel("Count (log)")
#     plt.ylabel("Class (shortened)")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def plot_label_cardinality_hist(cardinality: np.ndarray, out_path: Path):
#     if cardinality.size == 0:
#         return
#     max_k = int(cardinality.max())
#     bins = np.arange(0, max_k + 2) - 0.5

#     plt.figure(figsize=(9, 5))
#     plt.hist(cardinality, bins=bins)
#     plt.title("Label Cardinality per Clip (#labels per clip)")
#     plt.xlabel("#labels")
#     plt.ylabel("#clips")
#     plt.xticks(range(0, max_k + 1))
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def plot_duration_hist(durations: np.ndarray, out_path: Path):
#     if durations.size == 0:
#         return
#     plt.figure(figsize=(9, 5))
#     plt.hist(durations, bins=30)
#     plt.title("Clip Duration Distribution")
#     plt.xlabel("Duration (seconds)")
#     plt.ylabel("#clips")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def plot_rank_frequency(label_freq: Counter, out_path: Path):
#     if not label_freq:
#         return
#     counts = np.array(sorted(label_freq.values(), reverse=True), dtype=float)
#     ranks = np.arange(1, counts.size + 1)

#     plt.figure(figsize=(9, 5))
#     plt.plot(ranks, counts)
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.title("Class Frequency Long-Tail (Rank vs Count)")
#     plt.xlabel("Rank (log)")
#     plt.ylabel("Count (log)")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def plot_cumulative_coverage(label_freq: Counter, out_path: Path):
#     if not label_freq:
#         return
#     counts = np.array(sorted(label_freq.values(), reverse=True), dtype=float)
#     cum = np.cumsum(counts) / counts.sum()
#     k = np.arange(1, counts.size + 1)

#     plt.figure(figsize=(9, 5))
#     plt.plot(k, cum)
#     plt.title("Cumulative Label Occurrence Coverage vs Top-K Classes")
#     plt.xlabel("Top-K classes")
#     plt.ylabel("Cumulative fraction of label occurrences")
#     plt.ylim(0, 1.01)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def plot_cooccurrence_heatmap(per_clip_labels, label_freq: Counter, idx_to_name: dict, out_path: Path, top_k=25):
#     top = [i for i, _ in label_freq.most_common(top_k)]
#     if len(top) < 2:
#         return

#     idx_pos = {idx: p for p, idx in enumerate(top)}
#     mat = np.zeros((len(top), len(top)), dtype=np.int64)

#     for labs in per_clip_labels:
#         s = [x for x in labs if x in idx_pos]
#         if len(s) < 2:
#             continue
#         for a, b in itertools.combinations(sorted(set(s)), 2):
#             ia, ib = idx_pos[a], idx_pos[b]
#             mat[ia, ib] += 1
#             mat[ib, ia] += 1

#     np.fill_diagonal(mat, 0)
#     names = [shorten_name(idx_to_name.get(i, str(i)), 18) for i in top]

#     vmax = int(mat.max()) if mat.size else 1
#     if vmax <= 1:
#         vmax = 2

#     plt.figure(figsize=(11, 10))
#     plt.imshow(
#         mat,
#         aspect="auto",
#         cmap="magma",
#         norm=LogNorm(vmin=1, vmax=vmax),
#         interpolation="nearest",
#     )
#     plt.colorbar(label="Co-occurrence count (log)")
#     plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
#     plt.yticks(range(len(names)), names, fontsize=7)
#     plt.title(f"Co-occurrence Heatmap (Top {top_k} classes, log-scaled)")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=220)
#     plt.close()


# def write_summary(train_stats, eval_stats, out_path: Path):
#     def _dur_stats(durs: np.ndarray):
#         if durs.size == 0:
#             return "duration: not found in manifest\n"
#         return (
#             f"duration_seconds: count={durs.size}, "
#             f"min={durs.min():.3f}, mean={durs.mean():.3f}, median={np.median(durs):.3f}, max={durs.max():.3f}\n"
#         )

#     def _sr_stats(srs: np.ndarray):
#         if srs.size == 0:
#             return "sample_rate_hz: not found in manifest\n"
#         uniq, cnt = np.unique(srs, return_counts=True)
#         pairs = ", ".join([f"{u}({c})" for u, c in sorted(zip(uniq.tolist(), cnt.tolist()))])
#         return f"sample_rate_hz: {pairs}\n"

#     def _basic(name, stats):
#         freq = stats["label_freq"]
#         card = stats["label_cardinality"]
#         n = stats["n_items"]

#         uniq_classes = len(freq)
#         card_nonzero = card[card > 0]
#         avg_card = float(card_nonzero.mean()) if card_nonzero.size else 0.0
#         p0 = float((card == 0).mean()) if card.size else 0.0

#         maxf = max(freq.values()) if freq else 0
#         minf = min(freq.values()) if freq else 0
#         imbalance = (maxf / minf) if (maxf and minf) else 0.0

#         ent = stats.get("entropy", 0.0)
#         gini = stats.get("gini", 0.0)

#         top5 = freq.most_common(5)
#         top5_str = ", ".join([f"{k}:{v}" for k, v in top5]) if top5 else "n/a"

#         return (
#             f"{name}:\n"
#             f"clips = {n}\n"
#             f"unique_classes_present = {uniq_classes}\n"
#             f"avg_labels_per_clip = {avg_card:.3f}\n"
#             f"fraction_clips_with_0_labels = {p0:.3f}\n"
#             f"class_freq_max_min_ratio = {imbalance:.3f}\n"
#             f"class_entropy = {ent:.4f}\n"
#             f"class_gini = {gini:.4f}\n"
#             f"top5_class_indices:counts = {top5_str}\n"
#             f"{_dur_stats(stats['durations'])}"
#             f"{_sr_stats(stats['sample_rates'])}"
#         )

#     out_path.write_text(_basic("TRAIN", train_stats) + "\n" + _basic("EVAL/VAL", eval_stats), encoding="utf-8")


# def main():
#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     idx_to_name, mid_to_idx = read_label_map(LABEL_CSV)

#     train_items = load_manifest_any(TRAIN_JSON)
#     eval_items = load_manifest_any(EVAL_JSON)

#     train_stats = compute_stats(train_items, mid_to_idx)
#     eval_stats = compute_stats(eval_items, mid_to_idx)

#     plot_top_class_freq_hbar(
#         train_stats["label_freq"],
#         idx_to_name,
#         OUT_DIR / "train_top20_class_freq_log_hbar.png",
#         OUT_DIR / "train_top20_label_mapping.txt",
#         top_k=20,
#     )
#     plot_top_class_freq_hbar(
#         eval_stats["label_freq"],
#         idx_to_name,
#         OUT_DIR / "eval_top20_class_freq_log_hbar.png",
#         OUT_DIR / "eval_top20_label_mapping.txt",
#         top_k=20,
#     )

#     plot_label_cardinality_hist(train_stats["label_cardinality"], OUT_DIR / "train_label_cardinality_hist.png")
#     plot_label_cardinality_hist(eval_stats["label_cardinality"], OUT_DIR / "eval_label_cardinality_hist.png")

#     plot_cooccurrence_heatmap(
#         train_stats["per_clip_labels"],
#         train_stats["label_freq"],
#         idx_to_name,
#         OUT_DIR / "train_cooccurrence_top25_heatmap_log.png",
#         top_k=25,
#     )

#     plot_duration_hist(train_stats["durations"], OUT_DIR / "train_duration_hist.png")
#     plot_duration_hist(eval_stats["durations"], OUT_DIR / "eval_duration_hist.png")

#     plot_rank_frequency(train_stats["label_freq"], OUT_DIR / "train_rank_frequency_loglog.png")
#     plot_cumulative_coverage(train_stats["label_freq"], OUT_DIR / "train_cumulative_coverage.png")

#     write_summary(train_stats, eval_stats, OUT_DIR / "dataset_stats_summary.txt")

#     print(f"Saved everything into: {OUT_DIR}")


# if __name__ == "__main__":
#     main()