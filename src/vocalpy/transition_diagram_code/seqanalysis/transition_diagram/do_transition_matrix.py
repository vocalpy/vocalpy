import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython import embed

from seqanalysis.util.get_data_transition_diagram import get_labels, get_bouts, replace_chunks, get_analyse_files_data
from seqanalysis.util.calc_matrix_transition_diagram import get_transition_matrix, get_node_matrix
import seqanalysis.util.plot_transition_diagram_functions as pf
from seqanalysis.util.logging import config_logging

log = config_logging()


def get_data(cfg, analyse_files: str):
    folder_path = pathlib.Path((cfg["paths"]["folder_path"]))
    if not folder_path.exists():
        log.error(f"Path {folder_path} does not exist")
        raise FileNotFoundError(f"Path {folder_path} does not exist")

    file_list = get_analyse_files_data(folder_path, analyse_files)
    if not file_list:
        log.error(f"No files found in {file_list}")
        raise FileNotFoundError(f"No files found in {file_list}")
    log.info(f"Files found: {len(file_list)} in {analyse_files} files")

    seqs = get_labels(
        file_list,
        cfg["labels"]["intro_notes"],
        cfg["labels"]["intro_notes_replacement"],
    )

    cfg["data"]["bouts"], cfg["data"]["noise"] = get_bouts(
        seqs, cfg["labels"]["bout_chunk"]
    )

    if cfg["labels"]["double_syl"] is not None:
        cfg["data"]["bouts_rep"] = cfg["data"]["bouts"]
        log.info("Replacing double syllables")
        for i, (double_syll, renamed_double_syll) in enumerate(
                zip(cfg["labels"]["double_syl"], cfg["labels"]["double_syl_rep"])
        ):
            log.info(f"Replacing {double_syll} with {renamed_double_syll}")
            cfg["data"]["bouts_rep"] = re.sub(
                str(double_syll),
                renamed_double_syll,
                cfg["data"]["bouts_rep"],
            )
    else:
        cfg["data"]["bouts_rep"] = cfg["data"]["bouts"]

    log.info("Replacing chunks")
    cfg["data"]["chunk_bouts"], cfg["labels"]["chunks_renamed"] = replace_chunks(
        cfg["data"]["bouts_rep"], cfg["labels"]["chunks"]
    )

    return cfg


def make_first_plots(cfg):
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    bouts = cfg["data"]["bouts_rep"]
    # unique_labels in bouts
    unique_labels = sorted(list(set(bouts)))
    log.info(f"Unique labels of Chunks from bouts_rep: {unique_labels}\n")
    tm, _ = get_transition_matrix(bouts, unique_labels)

    # U2 is the size of the strings in zeros
    label_matrix = np.zeros((len(unique_labels), len(unique_labels)), "U2")
    for i, labely in enumerate(unique_labels):
        for j, labelx in enumerate(unique_labels):
            labelylabelx = str(labely + labelx)
            label_matrix[i, j] = labelylabelx

    # NOTE: Sort tm by the transitions with the highest probability
    tm_prob = np.around((tm.T / np.sum(tm, axis=0)).T, 2) * 100
    tm_sorted = np.zeros(tm.shape)
    label_matrix_sorted = np.zeros((len(unique_labels), len(unique_labels)), "U2")
    _multiple_index = [0]
    # NOTE: Add the first element befor entering the loop
    tm_sorted[0] = tm[0]
    label_matrix_sorted[0] = label_matrix[0]
    for i in range(1, tm.shape[0]):
        for sort in np.argsort(tm_sorted[i - 1])[::-1]:
            log.debug(sort)
            if sort not in _multiple_index:
                log.debug(_multiple_index)
                tm_sorted[i] = tm[sort]
                label_matrix_sorted[i] = label_matrix[sort]
                _multiple_index.append(sort)
                log.debug(_multiple_index)
                break
            else:
                continue

    # Sort the columns of the matrix
    multiple_index_col_shift = np.roll(_multiple_index, -1)

    tm_sorted_shift = tm_sorted[:, multiple_index_col_shift]
    label_matrix_shift = label_matrix_sorted[:, multiple_index_col_shift]
    tm_sorted_no_shift = tm_sorted[:, _multiple_index]
    label_matrix_no_shift = label_matrix_sorted[:, _multiple_index]

    tmd = tm_sorted_shift.astype(int)
    tmd_no_shift = tm_sorted_no_shift.astype(int)

    tmpd = (tmd.T / np.sum(tmd, axis=1)).T
    tmpd_no_shift = (tmd_no_shift.T / np.sum(tmd_no_shift, axis=1)).T
    tmpd = get_node_matrix(tmpd, 0)
    tmpd_no_shift = get_node_matrix(
        tmpd_no_shift, 0)

    # "Plot Transition Matrix and Transition Diagram"
    node_size = (
            np.round(
                np.sum(tmpd_no_shift, axis=1) / np.max(np.sum(tmpd_no_shift, axis=1)), 2
            )
            * 500
    )
    # get them into the right order
    # nice labels
    xlabels = []
    ylabels = []
    for x, y in zip(label_matrix_shift[0, :], label_matrix_shift[:, 0]):
        xlabels.append(x[1])
        ylabels.append(y[0])

    pf.plot_transition_diagram(
        tmpd_no_shift,
        ylabels,
        node_size,
        cfg["constants"]["edge_width"],
        cfg["paths"]["save_path"] + cfg["title_figures"] + "_graph_simple.pdf",
        cfg["title_figures"] + " simple",
    )

    pf.plot_transition_matrix(
        tmpd,
        xlabels,
        ylabels,
        cfg["paths"]["save_path"] + cfg["title_figures"] + "_matrix_simple.pdf",
        cfg["title_figures"] + " simple",
    )
    # log.info("Suggestion for labels")
    # for lab in ylabels:
    #     print(f"- {lab}")
    # for ch in cfg["labels"]["chunks_renamed"]:
    #     print(f"- {ch[1]}")
    plt.show()


def make_final_plots(cfg):
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    bouts = cfg["data"]["chunk_bouts"]
    unique_labels = sorted(list(set(bouts)))
    # place "_" at the beginning of the list
    unique_labels.remove("_")
    unique_labels.insert(0, "_")
    unique_labels = np.array(unique_labels)
    label_matrix_check = np.zeros((len(unique_labels), len(unique_labels)), "U2")
    for i, labely in enumerate(unique_labels):
        for j, labelx in enumerate(unique_labels):
            labelylabelx = str(labely + labelx)
            label_matrix_check[i, j] = labelylabelx

    log.info(f"Unique labels of Chunks: {unique_labels}\n")
    log.info(f"Label matrix:\n {label_matrix_check[:, 0]}")
    tm, tmp = get_transition_matrix(bouts, unique_labels)

    # Filter out nodes with low occurrence
    k = np.where(
        (np.sum(tm, axis=0) / np.sum(tm)) * 100 <= cfg["constants"]["node_threshold"]
    )
    tmd = np.delete(tm, k, axis=1)
    tmd = np.delete(tmd, k, axis=0)
    label_matrix_check = np.delete(label_matrix_check, k, axis=0)
    log.info(f"Updated label matrix:\n {label_matrix_check[:, 0]}")

    unique_labels = np.delete(unique_labels, k)
    # U2 is the size of the strings in zeros
    label_matrix = np.zeros((len(unique_labels), len(unique_labels)), "U2")
    for i, labely in enumerate(unique_labels):
        for j, labelx in enumerate(unique_labels):
            labelylabelx = str(labely + labelx)
            label_matrix[i, j] = labelylabelx

    tm_sorted = np.zeros(tmd.shape)
    label_matrix_sorted = np.zeros((len(unique_labels), len(unique_labels)), "U2")
    _multiple_index = [0]
    # NOTE: Add the first element befor entering the loop
    tm_sorted[0] = tmd[0]
    label_matrix_sorted[0] = label_matrix[0]
    for i in range(1, tmd.shape[0]):
        for sort in np.argsort(tm_sorted[i - 1])[::-1]:
            log.debug(sort)
            if sort not in _multiple_index:
                # log.debug(_multiple_index)
                tm_sorted[i] = tmd[sort]
                label_matrix_sorted[i] = label_matrix[sort]
                _multiple_index.append(sort)
                # log.debug(_multiple_index)
                break
            else:
                continue

    # Sort the columns of the matrix
    multiple_index_col_shift = np.roll(_multiple_index, -1)

    tm_sorted_shift = tm_sorted[:, multiple_index_col_shift]
    label_matrix_shift = label_matrix_sorted[:, multiple_index_col_shift]
    tm_sorted_no_shift = tm_sorted[:, _multiple_index]
    label_matrix_no_shift = label_matrix_sorted[:, _multiple_index]

    tmd = tm_sorted_shift.astype(int)
    tmd_no_shift = tm_sorted_no_shift.astype(int)

    # Normalize transition matrix and create node matrix
    tmpd = (tm_sorted_shift.T / np.sum(tm_sorted_shift, axis=1)).T
    tmpd_no_shift = (tm_sorted_no_shift.T / np.sum(tm_sorted_no_shift, axis=1)).T

    tmpd = get_node_matrix(tmpd, cfg["constants"]["edge_threshold"])
    tmpd_no_shift = get_node_matrix(
        tmpd_no_shift, cfg["constants"]["edge_threshold"]
    )
    node_size = (
            np.round(np.sum(tmd_no_shift, axis=1) / np.min(np.sum(tmd_no_shift, axis=1)), 2)
            * cfg["constants"]["node_size"]
    )

    xlabels = []
    ylabels = []

    for x, y in zip(label_matrix_shift[0, :], label_matrix_shift[:, 0]):
        renamedch = [ch[1] for ch in cfg["labels"]["chunks_renamed"]]
        ch = [ch[0] for ch in cfg["labels"]["chunks_renamed"]]
        if x[1] in renamedch:
            xlabels.append(ch[renamedch.index(x[1])])
        elif x[1] == "_":
            xlabels.append("End")
        else:
            xlabels.append(x[1])

        if y[0] in renamedch:
            ylabels.append(ch[renamedch.index(y[0])])
        elif y[0] == "_":
            ylabels.append("Start")
        else:
            ylabels.append(y[0])

    pf.plot_transition_diagram(
        tmpd_no_shift,
        ylabels,
        node_size,
        cfg["constants"]["edge_width"],
        cfg["paths"]["save_path"] + cfg["title_figures"] + "_graph.pdf",
        cfg["title_figures"],
    )

    pf.plot_transition_matrix(
        tmpd,
        xlabels,
        ylabels,
        cfg["paths"]["save_path"] + cfg["title_figures"] + "_matrix.pdf",
        cfg["title_figures"],
    )
    plt.show()

    # for lab in ylabels:
    #     print(f"- {lab}")
    # if cfg["labels"]["unique_labels"]:
    #     for lab in cfg["labels"]["unique_labels"]:
    #         renamedch = [ch[1] for ch in cfg["labels"]["chunks_renamed"]]
    #         ch = [ch[0] for ch in cfg["labels"]["chunks_renamed"]]


def main(yaml_file, analyse_files):
    with open(yaml_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

        if not cfg["data"]["bouts"]:
            cfg = get_data(cfg, analyse_files)
            log.info("No bouts found in yaml file - created")

        if cfg["nonchunk_plot"]:
            make_first_plots(cfg)

        make_final_plots(cfg)

        user_input = input("Do you want to save the yaml file? (y/n): ")

        match user_input:
            case "y" | "Y":
                with open(yaml_file, "w") as f:
                    yaml.dump(cfg, f)
                    # print(yaml.dump(cfg))
                    f.close()
            case "n" | "N":
                log.info("Yaml file not saved")
            case _:
                log.error("Invalid input")
                raise ValueError("Invalid input")


if __name__ == "__main__":
    # this script plots transition matrix and diagrams
    #
    # INPUT:
    # sys.argv[1] = yaml file of the bird, example: example_yaml.yaml
    # sys.argv[2] = analysis catch or all files: input: catch, batch, notcatch
    #
    # OUTPUT:
    # figures

    main(sys.argv[1], sys.argv[2])
