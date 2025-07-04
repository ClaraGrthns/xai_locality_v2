import argparse
import numpy as np
from src.utils.plotting_utils import plot_accuracy_vs_radius
import os.path as osp
from src.utils.misc import load_and_get_non_zero_cols

def parse_args():
    parser = argparse.ArgumentParser(description="Plot accuracy vs fraction for different kernel widths.")
    parser.add_argument('--kernel_widths', nargs='+', type=float, default = [1.0, 2.5, 5.3, 16.0],   help='List of kernel widths.')
    parser.add_argument("--results_folder", type=str, help="Path to the results folder")
    parser.add_argument("--results_path", type=str, help="Path to save results", default="/home/grotehans/xai_locality/results/LightGBM/jannis")
    parser.add_argument('--setting', type=str,  help='Setting string.')
    parser.add_argument("--experiment_setting", type=str, default = "thresholds-0-4.0-max63.0num_tresh-150_only_test_kernel_width-{kw}_model_regr-ridge_model_type-LightGBM_accuracy_fraction.npy.npz", help="Leave kernel_width as")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    def get_path(base_folder, base_path, setting, suffix=""):
        if base_folder is None:
            return base_path
        assert setting is not None, "Setting must be specified if folder is provided"
        return osp.join(base_folder, f"{suffix}{setting}")
    
    results_path = get_path(args.results_folder, args.results_path, args.setting)
    experiment_result_paths = [osp.join(results_path, args.experiment_setting.format(kw=kw)) for kw in args.kernel_widths]
    accuracy_arrays = []
    fraction_arrays = []
    non_zero_cols_list = []

    for result_path in experiment_result_paths:
        accuracy_array, fraction_array, _, non_zero_cols = load_and_get_non_zero_cols(result_path)
        accuracy_arrays.append(accuracy_array)
        fraction_arrays.append(fraction_array)
        non_zero_cols_list.append(non_zero_cols)

    min_cols = min(non_zero_cols_list)
    accuracies_concat = np.stack([acc[:, :min_cols] for acc in accuracy_arrays], axis=0)
    unique_arrays = set([tuple(arr.flatten()) for arr in accuracies_concat])

    print(f"Number of unique arrays: {len(unique_arrays)}")
    print(f"Expected length: {len(accuracy_arrays)}")
    if len(unique_arrays) != len(accuracy_arrays):
        print("Warning: Some arrays are identical!")
    else:
        print("All arrays are unique.")

    fraction_concat = np.stack([frac[:, :min_cols] for frac in fraction_arrays], axis=0)

    max_accuracies = np.max(accuracies_concat, axis=0)
    max_indices = np.argmax(accuracies_concat, axis=0)
    print("unqiue values of kernel indices", np.unique(max_indices))
    selected_fractions = fraction_concat[max_indices, np.arange(fraction_concat.shape[1])[:, None], np.arange(fraction_concat.shape[2])]

    kernel_id_to_width = {i: str(kw) for i, kw in enumerate(args.kernel_widths)}
    graphics_path = osp.join(results_path, "graphics", f"max_accuracy_vs_fraction_kernel.pdf")
    plot_accuracy_vs_radius(max_accuracies, selected_fractions, 
                              kernel_ids=max_indices, 
                              kernel_id_to_width=kernel_id_to_width, 
                              title_add_on=" - max over kernel widths",
                              save_path=graphics_path,
                              alpha=0.5)
    print("Saved plot to", graphics_path)

if __name__ == "__main__":
    main()