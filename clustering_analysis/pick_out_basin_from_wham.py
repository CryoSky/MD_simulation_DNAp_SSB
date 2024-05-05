import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="This script indicates the frame of a given basin, 1 indexed")
    parser.add_argument("x_axis_file", help='The x_axis variable file path', type=str)
    parser.add_argument("y_axis_file", help='The y_axis variable file path', type=str)
    parser.add_argument("wham_basin_center_x", help='The wham_basin_center_x', type=float)
    parser.add_argument("wham_basin_range_x", help='The wham_basin_range_x', type=float)
    parser.add_argument("wham_basin_center_y", help='The wham_basin_center_y', type=float)
    parser.add_argument("wham_basin_range_y", help='The wham_basin_range_y', type=float)
    args = parser.parse_args()

    x_data_list = []
    y_data_list = []

    with open(args.x_axis_file, 'r') as fopenx:
        for line in fopenx:
            x_data_list.append(float(line.strip()))

    with open(args.y_axis_file, 'r') as fopeny:
        for line in fopeny:
            y_data_list.append(float(line.strip()))


    df = pd.DataFrame({'x_axis': x_data_list, 'y_axis': y_data_list})

    mask_x = (df['x_axis'] >= (args.wham_basin_center_x - args.wham_basin_range_x)) & (df['x_axis'] <= (args.wham_basin_center_x + args.wham_basin_range_x))
    mask_y = (df['y_axis'] >= (args.wham_basin_center_y - args.wham_basin_range_y)) & (
                df['y_axis'] <= (args.wham_basin_center_y + args.wham_basin_range_y))

    filtered_df = df[mask_x & mask_y]
    row_numbers = filtered_df.index.tolist()

    row_numbers = [x+1 for x in row_numbers]
    print(row_numbers)

if __name__ == '__main__':
    main()