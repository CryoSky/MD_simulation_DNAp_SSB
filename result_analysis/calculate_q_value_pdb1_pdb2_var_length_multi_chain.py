#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# A greatly expansion of the calculate Q script, now support multiple chains
# Remeber only works for ATOM, not for HETATM
# Example in Linux: python calculate_q_value_pdb1_pdb2_var_length_multi_chain.py T1003/3D-JIGSAW_SL1_TS1.pdb T1003_reference_6hrh_A.pdb 0 --start1 30 --end1 465


__AUTHOR__ = 'Shikai Jin'
__DATE__ = '2022-06-11'
__VERSION__ = '1.1'

import argparse
import math
import sys
from Bio.PDB.PDBParser import PDBParser


def vector(p1, p2):
    return [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]


def vabs(a):
    return math.sqrt(pow(a[0], 2) + pow(a[1], 2) + pow(a[2], 2))


# For the equation of calculate the Q value read dx.doi.org/10.1021/jp212541y

def compute_Q_pdb1_pdb2(ca_atoms_pdb1, ca_atoms_pdb2, q_type, sigma_sq, contact, cutoff, minimum_separation):
    Q_value = 0
    count_a = 0
    ca_length = len(ca_atoms_pdb2)
    if q_type == 1:
        minimum_separation = 4
        cutoff = 9.5
        contact = True
    elif q_type == 0:
        minimum_separation = 3
        contact = False  # Temporality save to avoid q=0 and contact=true
        cutoff = 9.5
    elif q_type == 2:
        pass
    else:
        sys.exit("The Q value type %s should be 2, 1 or 0" % q_type)

    if len(ca_atoms_pdb1) != len(ca_atoms_pdb2):
        sys.exit("The number of C alpha atoms in pdb1 file %s doesn't match the number in pdb2 file %s" % (
            len(ca_atoms_pdb1), len(ca_atoms_pdb2)))

    # print(contact)
    # print(cutoff)
    for ia in range(ca_length):
        for ja in range(ia + minimum_separation, ca_length):
            if (ia + 1) in ca_atoms_pdb1 and (ja + 1) in ca_atoms_pdb1:
                rij_N = vabs(
                    vector(ca_atoms_pdb1[ia + 1], ca_atoms_pdb1[ja + 1]))
                rij = vabs(
                    vector(ca_atoms_pdb2[ia + 1], ca_atoms_pdb2[ja + 1]))
                dr = rij - rij_N
                if contact == True:
                    if abs(rij_N) < cutoff:
                        Q_value = Q_value + \
                            math.exp(-dr * dr / (2 * sigma_sq[ja - ia]))
                        count_a = count_a + 1
                else:
                    Q_value = Q_value + \
                        math.exp(-dr * dr / (2 * sigma_sq[ja - ia]))
                    count_a = count_a + 1
    try:
        Q_value = Q_value / count_a
    except:
        Q_value = 0
    return Q_value


def calc_sigma_sq(ca_atoms_pdb):
    sigma = []
    sigma_sq = []
    sigma_exp = 0.15

    for i in range(0, len(ca_atoms_pdb) + 1):
        sigma.append((1 + i) ** sigma_exp)
        sigma_sq.append(sigma[-1] * sigma[-1])

    return sigma_sq


def pdb_load(pdb_file, chain_label, start_text, end_text, verbose):
    if start_text == None:
        start_pair = [None]
    else:
        start_pair = start_text.split(',')
    if end_text == None:
        end_pair = [None]
    else:
        end_pair = end_text.split(',')

    if len(start_pair) != len(end_pair):
        sys.exit("The pair number of start point and end point are different")

    p = PDBParser(PERMISSIVE=1)  # Useful when find errors in PDB
    s = p.get_structure('pdb', pdb_file)
    # Compatible for multiple chains, but only for first model
    chains = s[0].get_list()
    chain_id = 0

    ca_atoms_pdb_all_chains = {}
    pdb_chain_id = []

    calculated_chain_list = chain_label

    for chain in chains:
        if chain.id in calculated_chain_list:
            ca_atoms_pdb = {}  # A new dictionary to record the C_alpha atom coordinate in pdb file
            for pair_index in range(len(start_pair)):
                chain_id = chain_id + 1

                # contains protein, water, even ligand
                total_residue_list = chain.get_unpacked_list()
                protein_residue_list = []
                for all_res in total_residue_list:
                    # print(chain_id)
                    # print(all_res.get_id()[0])

                    is_protein_hetero_flag = (
                        all_res.get_id()[0] in [' ', 'H_MSE', 'H_M3L', 'H_CAS', 'H_NGP', 'H_IPR', 'H_IGL'])
                    if is_protein_hetero_flag:
                        # print(all_res.get_id()[0])
                        # purify to only protein chain
                        protein_residue_list.append(all_res)
                # print("Checkpoint 1")
                if protein_residue_list != []:
                    first_residue = protein_residue_list[
                        0]  # Attention! If there is a chain only contains water then this will output error
                    last_residue = protein_residue_list[-1]
                else:
                    if verbose:
                        print("WARNING: chain has no protein part, skip this")
                    continue
                if verbose:
                    print("first residue in protein part is ")
                    print(first_residue)
                    print("last residue in protein part is *important*")
                    print(last_residue)

                sequence_id_flag = 0
                if end_pair[pair_index] is None:  # Default end point is last residue
                    end = int(last_residue.get_id()[1])
                    if verbose:
                        print("End value is %s" %end)
                    # print(chain[353]['CA'].get_coord())
                else:
                    end = int(end_pair[pair_index])
                if start_pair[pair_index] is None:  # Some fxxking pdbs start from -7
                    start = int(first_residue.get_id()[1])
                else:
                    start = int(start_pair[pair_index])
                for res in protein_residue_list:
                    is_regular_res = (res.has_id('CA') and res.has_id('O')) or (res.get_id()[1] == end and res.has_id(
                        'CA'))  # Some pdbs lack the O atom of the last one residue...interesting, now fixed bug and correctly pick last residue
                    # is_regular_res = res.has_id('CA') and res.has_id('O')
                    hetero_flag = res.get_id()[
                        0]  # Get a list for each residue, include hetero flag, sequence identifier and insertion code
                    # https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
                    # print(is_regular_res)
                    if res.get_id()[1] == end and verbose:
                        print("Now is last residue")
                    if (hetero_flag in [' ', 'H_MSE', 'H_M3L', 'H_CAS', 'H_NGP', 'H_IPR', 'H_IGL']) \
                            and is_regular_res:
                        # The residue_id indicates that there is not a hetero_residue or water ('W')
                        sequence_id = res.get_id()[1]
                        # print(sequence_id)
                        if sequence_id_flag == 0:
                            last_sequence_id = sequence_id
                        else:
                            if last_sequence_id != int(sequence_id) - 1:
                                if verbose:
                                    print(
                                        "WARNING: the residues between %s and %s are lost" % (last_sequence_id, sequence_id))
                                else:
                                    pass
                                    # Some fxxking pdbs lost residues halfway
                            last_sequence_id = sequence_id
                            # print(res.get_id())
                        if sequence_id >= start and sequence_id <= end:
                            ca_atoms_pdb[sequence_id] = res[
                                'CA'].get_coord()  # Biopython only gets first CA if occupancy is not 1
                            # print(ca_atoms_pdb.keys())
                            # ca_atoms_pdb.append(res['CA'].get_coord())
                            pdb_chain_id.append(chain_id)
                        sequence_id_flag = sequence_id_flag + 1
            # print(ca_atoms_pdb.keys())
            ca_atoms_pdb_all_chains[chain.id] = ca_atoms_pdb

    return ca_atoms_pdb_all_chains


# def check_residue(ca_atoms_pdb1, ca_atoms_pdb2):


def main():
    parser = argparse.ArgumentParser(
        description="This script calculates Q value for two pdbs, currently only supports the first model and first chain.")
    parser.add_argument(
        "PDB_filename1", help="The file name of input pdb1, set as reference structure if you use Qo mode", type=str)
    parser.add_argument(
        "--start1", help="The start residue of protein1", type=str)
    parser.add_argument("--end1", help="The end residue of protein1", type=str)
    parser.add_argument(
        "PDB_filename2", help="The file name of input pdb2", type=str)
    parser.add_argument(
        "--start2", help="The start residue of protein2", type=str)
    parser.add_argument("--end2", help="The end residue of protein2", type=str)
    # parser.add_argument("output_filename", help="The file name of output file", type=str)
    parser.add_argument(
        "q_type", help="The Q value type, 0 for Q_wolynes, 1 for Q_onuchic", type=int, default=0)
    parser.add_argument("--contact", help="The contact mode with centain threshold",
                        action="store_true", default=False)
    parser.add_argument(
        "--cutoff", help="The cutoff value of contact", type=float, default=9.5)
    parser.add_argument(
        "--separation", help="The minimum separation of calculation", type=int, default=3)
    parser.add_argument(
        "--chain_list", help="The minimum separation of calculation", type=str, default='all')
    parser.add_argument("-v", "--verbose", help="The print or mute mode",
                        action="store_true", default=False)
    args = parser.parse_args()
    pdb1_file = args.PDB_filename1
    pdb2_file = args.PDB_filename2
    q_type = args.q_type
    start1 = args.start1
    end1 = args.end1
    start2 = args.start2
    end2 = args.end2
    contact = args.contact
    cutoff = args.cutoff
    verbose = args.verbose
    separation = args.separation
    chain_list = args.chain_list

    if pdb1_file[-4:].lower() != ".pdb" or pdb2_file[-4:].lower() != ".pdb":
        sys.exit("It must be a pdb file.")

    chain_label = []
    if chain_list == 'all':

        p = PDBParser(PERMISSIVE=1)  # Useful when find errors in PDB
        s = p.get_structure('pdb', pdb1_file)
        # Compatible for multiple chains, but only for first model
        chains = s[0].get_list()
        for chain in chains:
            chain_label.append(chain.id)
    else:
        for letter in chain_list:
            chain_label.append(letter)

    # print(chain_label)
    ca_atoms_pdb1_all_chain = pdb_load(
        pdb1_file, chain_label, start1, end1, verbose)
    ca_atoms_pdb2_all_chain = pdb_load(
        pdb2_file, chain_label, start2, end2, verbose)
    # print(ca_atoms_pdb1_all_chain)
    # print(ca_atoms_pdb2)

    q_total_chains = []
    length_total_chains = []
    sum_q = 0

    for each_chain_atoms_pdb1, each_chain_atoms_pdb2 in zip(ca_atoms_pdb1_all_chain.values(), ca_atoms_pdb2_all_chain.values()):
        sorted_keys1 = sorted(each_chain_atoms_pdb1.keys())
        sorted_keys2 = sorted(each_chain_atoms_pdb2.keys())
        # print(sorted_keys1)
        # print(sorted_keys2)

        range_pdb1 = int(list(sorted_keys1)[-1]) - int(list(sorted_keys1)[0])
        range_pdb2 = int(list(sorted_keys2)[-1]) - int(list(sorted_keys2)[0])

        if verbose:
            print("first PDB range %s" % range_pdb1)
            print("second PDB range %s" % range_pdb2)

        if len(each_chain_atoms_pdb1) != len(each_chain_atoms_pdb2):
            if range_pdb1 == range_pdb2:
                if verbose:
                    print("The two pdb have same residue range but different length.\
                    So there are some residues lost in the halfway. Now checking...")
                difference = sorted_keys1[0] - sorted_keys2[0]
                for i in sorted_keys1:
                    if i - difference in sorted_keys2:
                        continue
                    else:
                        if verbose:
                            print(
                                "The residue %s in protein1 has no corresponding residue in protein2, the value of it %s has been removed" % (
                                    i, each_chain_atoms_pdb1[i]))
                        each_chain_atoms_pdb1.pop(i)
                for j in sorted_keys2:
                    if j + difference in sorted_keys1:
                        continue
                    else:
                        if verbose:
                            print(
                                "The residue %s in protein2 has no corresponding residue in protein1, the value of it %s has been removed" % (
                                    j, each_chain_atoms_pdb2[j]))
                        each_chain_atoms_pdb2.pop(j)
                if len(each_chain_atoms_pdb1) != len(each_chain_atoms_pdb2):
                    sys.exit("No idea of fixing corresponding region.")
            else:
                print("Error: Two PDB structures have different lengths and ranges!")
                print(each_chain_atoms_pdb1.keys())
                print(len(each_chain_atoms_pdb1))
                print(each_chain_atoms_pdb2.keys())
                print(len(each_chain_atoms_pdb2))
                exit()

        new_each_chain_atoms_pdb1 = {
            i + 1: each_chain_atoms_pdb1[k] for i, k in enumerate(sorted(each_chain_atoms_pdb1.keys()))}
        # print(new_ca_atoms_pdb1.keys())  # Check the number whether start from 1 and match the length
        new_each_chain_atoms_pdb2 = {
            i + 1: each_chain_atoms_pdb2[k] for i, k in enumerate(sorted(each_chain_atoms_pdb2.keys()))}
        # Change the keys of ca_atoms_pdb dict into natural increment\
        # Since we trim the model so the real residue number may not work, if we have same length just reorder from 0
        # https://stackoverflow.com/questions/39126272/reset-new-keys-to-a-dictionary
        # print(new_each_chain_atoms_pdb1)

        sigma_sq = calc_sigma_sq(new_each_chain_atoms_pdb1)

        if len(new_each_chain_atoms_pdb1) > 0:
            # print(new_each_chain_atoms_pdb1)
            # print(new_each_chain_atoms_pdb2)
            q = compute_Q_pdb1_pdb2(new_each_chain_atoms_pdb1, new_each_chain_atoms_pdb2, q_type, sigma_sq, contact,
                                    cutoff, separation)  # For last frame
            # with open(output_file, 'a+') as out:
            #    out.write(str(round(q, 3)))
            #    out.write(' ')
            #    out.write('\n')
            q_total_chains.append(q)
            length_total_chains.append(len(each_chain_atoms_pdb1))

    sum_length = sum(length_total_chains)
    for each_q, each_length in zip(q_total_chains, length_total_chains):
        sum_q += each_q*each_length/sum_length

    # Cancel line break, this line will report error in Python2
    print(str(round(sum_q, 3)), end='')


if __name__ == '__main__':
    main()
