__author__ = 'Shikai Jin'
__date__ = '2024-Feb-21'
__version__ = '7.0'

# Cite Gallicchio, Emilio, et al. "Temperature weighted histogram analysis method,
# replica exchange, and transition paths." The Journal of Physical Chemistry B
# 109.14 (2005): 6722-6731.

# Usage Example: python wham_calculation.py pd_chain_com_distance_all.txt total_energy_new_all.txt test 50 300 250 345 5 5 12.5 37 -o test.png

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap


def min_max(input):
    input_array = np.array(input)
    return np.max(input_array), np.min(input_array)


def wham_generate_input(E_min, E_max, Q_min, Q_max, bin_n_Q, bin_n_E, T_target, T_sample, Q_list, k_bias, L_bias,
                        Q_E_file):
    # Target temperature is determined by what you used to generate the umbrella_potential 
    KB = 0.001987 # Universal gas constant in the unit of kcal·mol-1·K-1
    beta_0 = 1 / (KB * T_target)
    beta_i = 1 / (KB * T_sample)

    E_median = (E_max + E_min) / 2
    E_half = E_min - E_median # E_median is the median of energy value

    print("The number of windows in this WHAM calculation is %d" % len(Q_list))

    Q_bin_size = (Q_max - Q_min) / bin_n_Q
    E_bin_size = (E_max - E_min) / bin_n_E

    # Generate the wham counts file
    histogram_counts = np.zeros(bin_n_Q * bin_n_E, dtype=int)  # Use NumPy for faster array operations
    print(f"The length of 2-D histogram is {len(histogram_counts)}")
    print(f"The expected Q value for each window is listed here {Q_list}")

    # Generate the histogram counts file
    for each_line in Q_E_file:
        Q = each_line[0]
        E = each_line[1]

        # Use NumPy functions for bin calculations, discretize coordinate and energy
        # T-WHAM paper section 2, paragraph 1
        x_index = np.clip(int((Q - Q_min) / Q_bin_size), 0, bin_n_Q - 1)
        y_index = np.clip(int((E - E_min) / E_bin_size), 0, bin_n_E - 1)

        k = x_index + y_index * bin_n_Q
        histogram_counts[k] += 1

    # Generate umbrella potential file, this file saves the c_il value for each window / histogram pair
    umbrella_potential = np.zeros((len(Q_list), len(histogram_counts)),
                                       dtype=float)  # shape(number of q bins, number of sampled size)

    for window_index in range(len(Q_list)):
        c_max = 0
        histogram_index = 0
        Q_each_window = Q_list[window_index] # Represents how many simulation and the expected Q value in the lowest energy
        for _ in histogram_counts:
            x_index = histogram_index % bin_n_Q
            y_index = int((histogram_index - x_index) / bin_n_Q) # In case any error happens
            Q_tmp = Q_min + Q_bin_size * (x_index + 0.5) # Expected q value in this given 2D discretization bin
            E_tmp = E_half + E_bin_size * (y_index + 0.5) # Expected E value in this given 2D discretization bin

            # V is for biasing potential
            # Q_each_window is the bottom of each window, here we don't include 0.5
            # Thus, the k0 value need to multiple 0.5 and make sure it converts to kcal
            V = k_bias * ((Q_tmp - Q_each_window) ** L_bias)

            # Corresponding to the equation 4 in the original article
            # c_il is the bias factor which accounts for the effect of temperature and any biasing potentials
            c_il = np.exp(-(beta_i - beta_0) * E_tmp) * np.exp(-beta_i * V)
            if c_max < c_il:
                c_max = c_il  # The overall maximum c_il value for each window, not used if not verbose
            umbrella_potential[window_index, histogram_index] = c_il
            histogram_index += 1

        # print("The cij_max for the current simulation window is %6.2f" %c_max)
    return histogram_counts, umbrella_potential


# The original file is wham.c from unknown
def wham(histogram_counts, simcount, umbrella_potential, tolerance, max_iterations):
    """
      Samples WHAM distribution and returns standard deviation of each
      population.

      P({p0}|{n}) = [Prod_i (f_{i})^N_{i}] [Prod_l (p0{l})^n_{l}] - Equation 6 in the original paper

        f_{i} = Sum_l c_{i,l} p_{l}

      where:
      i: simulation index
      N_{i}: total count for simulation i
      l: bin index
      p0_{l}: probability of bin l
      n_{l}: total count in bin l (from any simulation)

      Samples by rejection using proposals from Dirichlet distribution



    p0{l} = N_{i,l} / Sum_i N_{i} f_{i} c_{i,l} - Equation 7 in the original paper

    exp(-f_{i}) = Sum_l c_{i,l} p0{l} - Equation 8 in the original paper

    p0{l}: unbiased distribution at bin l, the probability of finding the system in bin l
    N_{i,l}: total count at bin l from all simulations, represents to the Sum_i n_{i,l} (from counts_file)
    N_{i}: total samples from simulation i (from counts_file)
    c_{i,l}: umbrella potential in bin l of simulation i (from umbrella_potential)

    c_{i,l} = exp[-(betai-beta0)El] exp(-betai wi(x_l))

    where betai is temperature of simulation of i
    and wi(x_l) is umbrella potential at bin l from simulation i.
    If the temperature is different in each simulation, it is assumed
    that the potential energy E is one of the binned quantities.
    Therefore, El is potential energy in bin l. If the temperature
    is the same in all simulations, the first term in the right-hand side of
    the equation for c_{i,l} is 1 (betai = beta).


    Iterate p and f until their multiplication reach the maximum (Equation 6 in the original paper)

    In the original C file
    line 341 - 366 read counts file
    line 367 - 392 read simulation counts file
    line 393 - 426 read umbrella potential file

    This function reproduces the code in line 427 - 497
    """


    # nsamples = 128 # Leave for error check
    # passes_per_sample = 10 # Leave for error check

    simcount_len = len(simcount) # simcount_len is the same as args.repeat

    # start setting f_{i} = 1
    # simcount is a table that converts the variable file size into NumberOfWindow * PerWindowSize
    u = simcount * 1 # N_{i} * f_{i} in the beginning

    # calculate p until rms is satisfied
    rms = 2 * tolerance
    iter = 1

    # This part should represent Equations 7 and 8 in T-WHAM paper
    # A total of four variables, t, p, inverse_f, u
    # We want to iterate f to minimize the statistical error of the normalized frequency
    # of finding the system in the vicinity of a given value of x
    # Once we have f, we can combine windows to one global free energy
    # This process aims to find the set of weights that best reproduces the observed histograms.
    while rms >= tolerance and iter <= max_iterations:
        # Finish the calculation in the Equation 7
        # Matrix-vector multiplication a @ b = a.matmul(b), @ plays the role of Sum_i
        # histogram_counts already did the sum of n_{i,l} along with Sum_i to n_{l}, mean total samples of bin l
        t = u @ umbrella_potential  # In the equation 7, t = Sum_i N_{i} * f_{i} * c_{i,l}
        p = histogram_counts / t  # for each l, we have n_{l} / Sum_i N_{i} * f_{i} * c_{i,l}

        # Normalize p
        p /= np.sum(p) # Under any conditions, the sum of the probability of a system must be 1. Check Equation 15 and 16 in DOI: 10.1002/wcms.66

        # recalculate f, for Equation 8
        inverse_f = umbrella_potential @ p # c_{i,l} * p0_{l}, remember this is f_{i}^{-1}
        u = simcount / inverse_f # N_{i} * f_{i} update u

        # update t, check the rmsd
        # Matrix-vector multiplication a * b = np.multiply(a, b)
        update_t = u @ umbrella_potential * p # N_{i} * f_{i} * c_{i,l} * p0_{l} = updated n_{l}

        # The histogram_counts is always fixed, but updated_t is not.
        # We iterate p and f but their product is fixed
        rms = np.sqrt(np.mean(np.square(histogram_counts - update_t)))

        print("Iteration %d. RMSD = %6.4f" % (iter, rms))

        if np.isnan(rms):
            print("WARNING: The final RMSD value is NaN.") # The inverse f is too small, reaches the limit of numpy double
            break
        iter += 1

    if iter >= max_iterations:
        print("WARNING: Maximum number of iterations exceeded.")

    print("Simulation     -log(f)    f")
    for i in range(simcount_len):
        print("%d %16.8e %.8f" % (i+1, -np.log(inverse_f[i]), inverse_f[i]))

    # Error analysis part include line 26 - 183, line 500 - 520, all skipped
    # The output is unbiased probabilistic distribution
    return p  # line 523


def wham_pmf(Q_min, Q_max, bin_n_Q, bin_n_E, T_target, prob_per_bin):
    KB = 0.001987
    beta_i = 1 / (KB * T_target)

    Q_bin_size = (Q_max - Q_min) / bin_n_Q
    # print(f'The Q_bin_size is {Q_bin_size}')

    assert len(prob_per_bin) == bin_n_Q * bin_n_E
    p_check = np.sum(prob_per_bin) # Check whether the sum of probability is still 1
    print("check p_check = %f" % p_check)

    # Transfer the 2d discretization back and sum up
    # F means column-major order, C means row-major order
    p_Q = np.sum(prob_per_bin.reshape((bin_n_Q, bin_n_E), order='F'), axis=1) / Q_bin_size # transform probability to density
    print("The p_Q is:")
    print(p_Q)

    print("check p_Q = %f" % np.sum(p_Q))
    print(f'The beta under the current temperature is {beta_i}')

    output = []
    Q_values = np.linspace(Q_min, Q_max, bin_n_Q + 1)
    Q_midpoints = 0.5 * (Q_values[:-1] + Q_values[1:])

    for i, Q_midpoint in enumerate(Q_midpoints):
        if p_Q[i] != 0:
            pmf = -np.log(p_Q[i]) / beta_i
            print("%f %f" % (Q_midpoint, pmf))
            output.append([Q_midpoint, pmf])
        else:
            print("%f inf" % Q_midpoint)
            output.append([Q_midpoint, np.inf])

    return output


def make_colormap(color_bins, min_value, max_value):
    colors = [(60/255, 99/255, 190/255), (102/255, 168/255, 176/255), (187/255, 224/255, 152/255), (247/255, 249/255, 182/255), (243/255, 216/255, 154/255), (226/255, 160/255, 94/255), (211/255, 83/255, 71/255), (151/255, 32/255, 58/255)]
    # For color check this page https://www.html.am/html-codes/color/color-scheme.cfm?rgbcolor=245,245,245
    cmap_name = 'my_list_from_leo'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=color_bins)
    # bounds = [0, 1]
    bounds = [min_value, max_value]
    # cm.set_over('black')

    # norm = matplotlib.colors.BoundaryNorm(bounds, color_bins, extend='max')
    return cm

def pmf_2d_plot(T_min, T_max, dT, whole_wham_out, variable_x, variable_y, extra_sampling_temp, x_label, y_label, output_prefix):
    F_re_all = []
    binN = 50
    lower_cutoff = 2
    upper_cutoff = 12
    energy_label = 'F (kcal/mol)'
    n_contour = 21
    T_2d = extra_sampling_temp
    font_size = 12

    assert len(variable_x) == len(variable_y), "length between two variables not matched"

    for i in range(int((T_max - T_min) / dT)):
        T = T_min + dT * i
        if T == T_2d:
            FF = np.array(whole_wham_out[i])
            qx, Fy = FF[:, 0], FF[:, 1]
            nbin = len(qx)  # nbin is the same as args.repeat
            # print(f'The length of qx is {nbin}')
            dq = qx[1] - qx[0]
            qmin, qmax = qx[0] - dq / 2, qx[-1] + dq / 2

            # Calculate and normalize the probability in case the sum is not 1
            # Along with the temperature increase, the rare probability will increase the weight
            # The most frequent bin will lower the weight, thus, the depth difference between two basins will decrease
            Py = np.exp(-Fy / (0.001987 * T))
            P_norm = np.sum(Py)
            Py /= P_norm

            pi_sample = np.zeros(len(variable_x))
            ni_sample = np.zeros(nbin)

            # calculate pi_sample, the weight per window that comes from p
            # ni_sample is counts per window
            for i_bin in range(nbin):
                qi_min, qi_max = qmin + i_bin * dq, qmin + (i_bin + 1) * dq # Here we may have some issues
                ids = np.where(np.logical_and(variable_x >= qi_min, variable_x < qi_max))[0]
                ni_sample[i_bin] = len(ids)
                if ni_sample[i_bin] > 0:
                    pi_sample[ids] = Py[i_bin] / ni_sample[i_bin]

            #print(pi_sample[:20]) # pi_sample is matched old code

            print(f'probability = {np.sum(pi_sample):.3f}')
            # Recalculate the probability for coordinate in axis Y
            # The weight of each data point has been calculated by the pi_sample
            qa_lin = np.linspace(np.min(variable_x), np.max(variable_x), binN)
            qb_lin = np.linspace(np.min(variable_y), np.max(variable_y), binN)
            bin_index_x = np.digitize(variable_x, qa_lin) - 1
            bin_index_y = np.digitize(variable_y, qb_lin) - 1

            H = np.zeros((binN, binN))

            for i_sample in range(len(variable_x)):
                x = bin_index_x[i_sample]
                y = bin_index_y[i_sample]
                H[x, y] = H[x, y] + pi_sample[i_sample]

            H = H.transpose()
            pd.DataFrame(H).to_csv('H.csv')

            print(f'sum(sum(H)) = {np.sum(H):.3f}')
            F = -0.001987 * T * np.log(H)
            F[F >= upper_cutoff] = upper_cutoff

            pd.DataFrame(F).to_csv('F.csv')

            # fig, ax = plt.subplots(111)
            fig = plt.figure()

            # Set up the colormap
            cm = make_colormap(n_contour, lower_cutoff, upper_cutoff)
            cm.set_over('white')
            cm.set_under((43 / 255, 94 / 255, 209 / 255))
            # matplotlib.cm.colors.Normalize(0, 20)

            # contourf draws filled contours
            plt.contourf(qa_lin, qb_lin, F, n_contour, alpha=1, cmap=cm, vmin=lower_cutoff, vmax=upper_cutoff, levels=np.linspace(lower_cutoff, upper_cutoff - 0.01, n_contour), extend='both')
            plt.colorbar(ticks=np.linspace(lower_cutoff, upper_cutoff, n_contour))
            # C = plt.contour(qa_lin, qb_lin, F, n_contour, alpha=1, colors='dimgrey', linewidths=.35, vmax=np.matrix(F).max() - 0.01)

            # contour draws contour lines
            plt.contour(qa_lin, qb_lin, F, n_contour, alpha=1, colors='dimgrey', linewidths=.35,
                        vmin=lower_cutoff, vmax=upper_cutoff, levels=np.linspace(lower_cutoff, upper_cutoff, n_contour))
            plt.grid(visible=True, which='major', axis='both', linestyle=':')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim([np.min(variable_x), np.max(variable_x)])
            plt.ylim([np.min(variable_y), np.max(variable_y)])
            fig.savefig(f'2D_energy_{output_prefix}.png', dpi=600, bbox_inches='tight')
            fig.savefig(f'2D_energy_{output_prefix}.eps', dpi=600, bbox_inches='tight')
            plt.close()

def pmf_1d_plot(T_min, T_max, dT, whole_wham_out, variable, cv_for_compute, x_label, output_prefix):
    # cv_for_compute is the variable for a new set of reaction coordinates if you want to
    # use the pmf from one collective variable to calculate the profile for another one

    # Future color ID: #61d836, #00a2ff, #FF6A6A

    F_re_all = []
    binN = 50
    cutoff = 20
    curve_shift_flag = True
    q0_shift = 0.2
    y_label = 'F (kcal/mol)'


    # Deal with the colormap
    original_cmap = plt.get_cmap('jet')
    indices = np.linspace(0, 1, int((T_max - T_min) / dT))
    colors = original_cmap(indices)
    new_cmap = ListedColormap(colors)

    plt.figure()

    for i in range(int((T_max - T_min) / dT)):
        T = T_min + dT * i
        FF = np.array(whole_wham_out[i])
        qx, Fy = FF[:, 0], FF[:, 1]

        nbin = len(qx) # nbin is the same as args.repeat
        # print(f'The length of qx is {nbin}')
        dq = qx[1] - qx[0]
        qmin, qmax = qx[0] - dq / 2, qx[-1] + dq / 2

        # Calculate and normalize the probability in case the sum is not 1
        # Along with the temperature increase, the rare probability will increase the weight
        # The most frequent bin will lower the weight, thus, the depth difference between two basins will decrease
        Py = np.exp(-Fy / (0.001987 * T))
        P_norm = np.sum(Py)
        Py /= P_norm

        pi_sample = np.zeros(len(variable))
        ni_sample = np.zeros(nbin)

        # calculate pi_sample, the weight per window that comes from p
        # ni_sample is counts per window
        for i_bin in range(nbin):
            qi_min, qi_max = qmin + i_bin * dq, qmin + (i_bin + 1) * dq
            ids = np.where(np.logical_and(variable >= qi_min, variable < qi_max))[0]
            ni_sample[i_bin] = len(ids)
            if ni_sample[i_bin] > 0:
                pi_sample[ids] = Py[i_bin] / ni_sample[i_bin]

        # Recalculate the probability for new coordinate
        # The weight of each data point has been calculated by the pi_sample
        qa_lin = np.linspace(np.min(cv_for_compute), np.max(cv_for_compute), binN)
        bin_index_x = np.digitize(cv_for_compute, qa_lin) - 1
        # print(bin_index_x)
        count_qa = np.bincount(bin_index_x, weights=pi_sample, minlength=binN)

        F_qa = -0.001987 * T * np.log(count_qa)
        F_qa[F_qa >= cutoff] = cutoff

        if curve_shift_flag:
            id_shift = np.argmin(np.abs(qa_lin - q0_shift))
            # id_shift = np.argmin(qa_lin - q0_shift)
        else:
            id_shift = np.argmin(F_qa)

        F_re = F_qa - F_qa[id_shift]
        F_re_all.append(F_re)
        plt.plot(qa_lin, F_re, color=new_cmap(i), linewidth=4, label=f'{str(T)}K')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(f'1D_energy_{output_prefix}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'1D_energy_{output_prefix}.eps', dpi=600, bbox_inches='tight')

    return F_re_all

def read_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]

def navigate_to_or_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

    os.chdir(folder_path)
    print(f"Changed current working directory to '{folder_path}'.")

def generate_q_list(Q_min, Q_max, repeat):
    delta_Q = Q_max - Q_min
    dq = delta_Q / (repeat - 1)
    return [Q_min + dq * j for j in range(repeat)]

def plot_histogram(ax, data, bins, color, label=None, linestyle='-', linewidth=3):
    counts, edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(bin_centers, counts, linestyle, color=color, linewidth=linewidth, label=label)


def main():
    parser = argparse.ArgumentParser(
        description="This script do the WHAM based on Weihua's old code and T-WHAM code")
    parser.add_argument("variable_file", help='The variable file path', type=str)
    parser.add_argument("energy_file", help='The energy file path', type=str)
    parser.add_argument("folder", help='The folder for the calculation', type=str)
    parser.add_argument("repeat", help='How many simulation windows that you used', type=int)
    parser.add_argument("T_sample", help='The sampled temperature during the simulation', type=int, default=300)
    parser.add_argument("T_min", help='The minimum temperature that you want to elongate', type=int)
    parser.add_argument("T_max", help='The maximum temperature that you want to elongate', type=int)
    parser.add_argument("dT", help='The temperature delta during the range', type=int)
    parser.add_argument("k0", help='The k0, need to multiple 0.5 and make sure it converts to kcal', type=float)
    parser.add_argument("q_min", help='The expected q value in the first window', type=float)
    parser.add_argument("q_max", help='The expected q value in the last window', type=float)
    parser.add_argument("--q_bins", help='The number of variable bins that used in 2D discretization', type=int, default=50)
    parser.add_argument("--e_bins", help='The number of energy bins that used in 2D discretization', type=int, default=100)
    parser.add_argument("--twod_flag", help='The 2D flag', action='store_true')
    parser.add_argument("--extra_sampling_temp", help='The extra sampling temperature for 2D', type=int, default=300)
    parser.add_argument("--x_label", help='The name of the reaction coordinate in X axis', type=str,
                        default='Reaction Coordinate 1')
    parser.add_argument("--y_label", help='The name of the reaction coordinate in Y axis in 2D mode', type=str,
                        default='Reaction Coordinate 2')
    parser.add_argument("-e", "--extra_variable_file", help='The extra variable file path', default=None)
    parser.add_argument("-o", "--output_prefix", help='The output prefix', default='output')

    args = parser.parse_args()

    variable = read_file(args.variable_file)
    energy = read_file(args.energy_file)
    assert len(variable) == len(energy), 'The length of variable file is different than energy'

    folder = args.folder.lower()
    navigate_to_or_create_folder('./%s' % folder)

    Q_list = generate_q_list(args.q_min, args.q_max, args.repeat)
    # Notice the q_min and q_max from input only use for the calculation of q_list
    # After that, we will use real min/max from variable file to compute umbrella file
    tolerance = 0.00001
    max_iterations = 5000
    L0 = 2
    n_T = int((args.T_max - args.T_min) / args.dT)
    variable_file_length = len(variable)
    data_number_per_window = int(variable_file_length / args.repeat)
    print(f"The number of data in each window is ground to {data_number_per_window}")

    # simcount is a table that converts the variable file size into NumberOfWindow * PerWindowSize
    simcount = np.ones(args.repeat) * data_number_per_window
    q_E_file = list(zip(variable, energy))

    Q_max, Q_min = min_max(variable) # The real minimum and maximum values of q
    E_max, E_min = min_max(energy) # The real minimum and maximum values of E

    whole_wham_out = []

    for i_T in range(n_T):
        T_target = args.T_min + (args.T_max - args.T_min) / n_T * i_T
        print("The current temperature is %d" % T_target)
        # Generate input files
        histogram_counts, umbrella_potential = wham_generate_input(E_min, E_max, Q_min, Q_max,
                                                                   args.q_bins, args.e_bins, T_target, args.T_sample,
                                                                   Q_list, args.k0, L0, q_E_file)

        pd.DataFrame(histogram_counts, columns=None).to_csv(f'./historgram_counts_{args.output_prefix}.csv')
        pd.DataFrame(umbrella_potential, columns=None).to_csv(f'./umbrella_potential_{args.output_prefix}.csv')
        plt.figure()
        plt.imshow(histogram_counts.reshape((args.q_bins, args.e_bins), order='F'), cmap='binary')
        plt.xlabel('Energy bin index')
        plt.ylabel('Reaction coordinate bin index')
        plt.colorbar()
        plt.savefig(f'./histogram_counts_{args.output_prefix}.png', dpi=600)
        plt.close()

        fig = plt.figure()
        axes = fig.add_subplot(111)

        line_color = plt.cm.jet(np.linspace(0, 1, args.q_bins))
        for i_repeat in range(args.q_bins):
            qwindow = np.log(umbrella_potential)[i_repeat]
            plot_histogram(axes, qwindow, bins=args.q_bins, color=line_color[i_repeat - 1], label=i_repeat+1, linestyle='-', linewidth=0.5)
        plt.xlabel('2D discretization bin distribution')
        plt.ylabel('Histogram count')
        # legend = axes.legend(labelspacing=0.5)
        # legend.get_title().set_fontsize('5')
        plt.savefig(f'./umbrella_potential_{args.output_prefix}.png', dpi=600)
        plt.close()


        plt.figure()
        prob_per_bin = wham(histogram_counts, simcount, umbrella_potential, tolerance, max_iterations)
        pd.DataFrame(prob_per_bin, columns=None).to_csv(f'./prob_per_bin_{args.output_prefix}.csv')
        plt.imshow(prob_per_bin.reshape((args.q_bins, args.e_bins)), cmap='viridis')
        plt.xlabel('Energy bin index')
        plt.ylabel('Reaction coordinate bin index')
        plt.colorbar()
        plt.savefig(f'./prob_per_bin_{args.output_prefix}.png', dpi=600)
        plt.close()

        pmf_under_current_temp = wham_pmf(Q_min, Q_max, args.q_bins, args.e_bins, T_target, prob_per_bin)
        whole_wham_out.append(pmf_under_current_temp)

    # print(whole_wham_out)
    if args.extra_variable_file is not None:
        extra_variable = read_file(args.extra_variable_file)
    else:
        extra_variable = variable

    if args.twod_flag:
        pmf_2d_plot(args.T_min, args.T_max, args.dT, whole_wham_out, variable, extra_variable, args.extra_sampling_temp, args.x_label, args.y_label, args.output_prefix)
    else:
        pmf_1d_plot(args.T_min, args.T_max, args.dT, whole_wham_out, variable, extra_variable, args.x_label, args.output_prefix)


if __name__ == '__main__':
    main()
