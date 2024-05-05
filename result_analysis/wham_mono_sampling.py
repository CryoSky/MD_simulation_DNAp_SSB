import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(ax, data, bins, color, label=None, linestyle='-', linewidth=3):
    counts, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(bin_centers, counts, linestyle, color=color, linewidth=linewidth, label=label)

def main():
    parser = argparse.ArgumentParser(description='Plot Q and E distributions.')
    parser.add_argument('q_file', type=str, help='Path to the Q data file.')
    parser.add_argument('e_file', type=str, help='Path to the E data file.')
    parser.add_argument('output_file', type=str, help='Path to save the output plot.')
    args = parser.parse_args()

    repeat = 50
    k_bias = 500
    fsize = 20
    mr = 3
    mc = 1
    color = plt.cm.jet(np.linspace(0, 1, repeat))

    # This is used for plot only a sub-region of the plot
    window_index_list = [0]
    name_array_list = ['Test']

    for simulation_name in name_array_list:
        for window_index in window_index_list:
            fig, axes = plt.subplots(mr, mc, figsize=(20, 15))

            # Load q
            q_filename = args.q_file
            q = np.loadtxt(q_filename, dtype=float)
            Nline = len(q) // repeat

            # Plot q
            axes[0].grid(True)
            axes[0].plot(q, 'k', linewidth=1)
            axes[0].set_xlabel('Frames', fontsize=fsize)
            axes[0].set_ylabel('Q', fontsize=fsize)
            axes[0].set_title(f'{simulation_name}  N-qbias={repeat} k-bias={k_bias}', fontsize=fsize)

            # Load energy
            e_filename = args.e_file
            E = np.loadtxt(e_filename, dtype=float)

            # Plot Q distribution
            axes[1].grid(True)
            axes[1].set_xlabel('Q', fontsize=fsize)
            axes[1].set_ylabel('P(Q)', fontsize=fsize)
            binN_Qi = 30
            for i_repeat in range(1, repeat + 1):
                i_start = 1 + Nline * (i_repeat - 1)
                i_end = i_start + Nline - 1
                qwindow = q[i_start:i_end]
                plot_histogram(axes[1], qwindow, bins=binN_Qi, color=color[i_repeat - 1], linestyle='-', linewidth=3)

            # Create a secondary y-axis for the Q distribution
            secondary_y_axis = axes[1].twinx()
            p_q_total, qgrid_total = np.histogram(q, bins=30, range=(min(q), max(q)), density=False)
            secondary_y_axis.plot(qgrid_total[:-1], p_q_total, '--k', linewidth=5, label='Total')
            secondary_y_axis.legend()

            # Plot E distribution
            axes[2].grid(True)
            axes[2].set_xlabel('E (kcal/mol)', fontsize=fsize)
            axes[2].set_ylabel('P(E)', fontsize=fsize)
            binN_Ei = 50
            for i_repeat in range(1, repeat + 1):
                i_start = 1 + (i_repeat - 1) * Nline
                i_end = i_repeat * Nline
                Ewindow = E[i_start:i_end]
                plot_histogram(axes[2], Ewindow, bins=binN_Ei, color=color[i_repeat - 1], linestyle='-', linewidth=3)

            plt.savefig(args.output_file, dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    main()
