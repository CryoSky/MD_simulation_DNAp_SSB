import MDAnalysis as mda
import argparse
import numpy as np
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
import pandas as pd
import matplotlib.pyplot as plt

def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates - center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
    # square root and return
    return np.sqrt(rog_sq)


def compute_traj(trajectory, chain_id, output):
    '''
        Please check this page for the details of the function.
        https://userguide.mdanalysis.org/stable/examples/analysis/custom_trajectory_analysis.html#Radius-of-gyration
    '''
    u = mda.Universe(trajectory, format='PDB')
    if chain_id == None:
        ag = u.select_atoms('name CA')
    else:
        ag = u.select_atoms(f'chainID {chain_id} and name CA')

    rog = AnalysisFromFunction(radgyr, u.trajectory,
                           ag, ag.masses,
                           total_mass=np.sum(ag.masses))
    rog.run()
    # print(rog.results['timeseries'].shape)
    # print(rog.results['timeseries'][:,0])
    np.savetxt(output, rog.results['timeseries'][:,0], fmt='%.2f')


def main():
    parser = argparse.ArgumentParser(
        description="This script calculates the rg value for a given chain using MDanalysis")
    parser.add_argument("trajectory", help="The file name of trajectory in pdb format", type=str)
    parser.add_argument('--chain', help="The chain ID for extraction, no show means all, only --chain_id means A, otherwise use given ID", nargs='?', const='A', type=str)
    parser.add_argument("output", help="The file name of output",
                        type=str)
    args = parser.parse_args()

    compute_traj(args.trajectory, args.chain, args.output)


if __name__ == '__main__':
    main()