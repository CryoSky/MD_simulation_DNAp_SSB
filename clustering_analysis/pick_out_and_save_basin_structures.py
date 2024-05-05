__author__ = 'Shikai Jin'
__date__ = '2024-Mar-4'
__version__ = '1.0'

import argparse
import math
from collections import defaultdict
import MDAnalysis as mda

def main():
    parser = argparse.ArgumentParser(
        description="This script indicates the frame of a given basin, 1 indexed")
    parser.add_argument("list_file", help='The x_axis variable file path', type=str)
    parser.add_argument("trajectory_directory", help='The path to the directory that contains trajectory', type=str)
    parser.add_argument("frames_per_file", help='The x_axis variable file path', type=int)
    args = parser.parse_args()

    data = defaultdict(list)
    with open(args.list_file, 'r') as fopen:
        lines = fopen.readlines()
        for line in lines:
            traj_index = math.floor(int(line) / args.frames_per_file)
            frame_index = int(line) % args.frames_per_file
            data[traj_index].append(frame_index)
    for each_traj_index in data.keys():
        u = mda.Universe(f'{args.trajectory_directory}/output_{each_traj_index}.pdb', format='PDB')
        ag = u.select_atoms('all')
        for each_frame_index in data[each_traj_index]:
            print(type(u))
            ag.write(f'./output_{each_traj_index}_{each_frame_index}.pdb', frames=[each_frame_index])
            # written_frame = u.trajectory[each_frame_index].select_atoms('all')
            # written_frame.write(f'./output_{each_traj_index}_{each_frame_index}.pdb')


if __name__ == '__main__':
    main()


