import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def parse_xvg_data(filename):
    """Parses the XVG-like data file, extracting time and data columns.

    Args:
        filename (str): The path to the XVG-like file.

    Returns:
        tuple: A tuple containing:
            - times (numpy.ndarray): Array of time values.
            - data (numpy.ndarray): 2D array of data columns.  data[:, i] is
              the i-th data column.
            - legends (list): List of legend labels for each data column.
            - units (list): List of units for each data column.
    """
    times = []
    data_list = []  # Store data as lists initially for appending
    legends = []
    units = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('@'):
                if line.startswith('@') and 'legend' in line and '"' in line:
                    legend_text = line.split('"')[1]
                    legends.append(legend_text)
                if line.startswith('@') and 'yaxis' in line and 'label' in line:
                    # Extract and parse units from the yaxis label
                    units_str = line.split('"')[1]
                    # Split by commas and extract individual units
                    units = [u.strip().replace('(', '').replace(')', '').replace('^3', '³')  for u in units_str.split(',')]

                continue  # Skip header lines

            parts = line.split()
            times.append(float(parts[0]))
            data_list.append([float(x) for x in parts[1:]])

    # Convert lists to NumPy arrays for easier plotting
    times = np.array(times)
    data = np.array(data_list)

    #Check if number of columns is equal to the number of extracted legends
    if data.shape[1] != len(legends):
      raise ValueError(
            f"Number of data columns ({data.shape[1]}) does not match "
            f"the number of legends ({len(legends)})."
        )

    if len(units) != len(legends) and len(units) !=0 :
        raise ValueError("Number of units does not match the number of legends.")

    if not units: #if empty
        units = [""] * len(legends)  # Fill with empty strings if no units are found

    return times, data, legends, units


def plot_data(times, data, legends, units, output_dir, prefix="", plot_type="separate"):
    """Plots the data and saves the plots.

    Args:
        times (numpy.ndarray): Array of time values.
        data (numpy.ndarray): 2D array of data columns.
        legends (list): Legend labels.
        units (list): Units for each data column.
        output_dir (str): Directory to save plots in.
        prefix (str): Prefix for plot filenames.
        plot_type (str): 'separate' (default) for individual plots,
                         'combined' for a single plot, or 'both'.
    """
    os.makedirs(output_dir, exist_ok=True)

    if plot_type in ("separate", "both"):
        for i in range(data.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.plot(times, data[:, i])
            plt.xlabel("Time (ps)")
            plt.ylabel(f"{legends[i]} ({units[i]})")
            plt.title(f"{legends[i]} vs. Time")
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            output_filename = os.path.join(output_dir, f"{prefix}{legends[i].lower().replace(' ', '_')}_vs_time.png")
            plt.savefig(output_filename)
            plt.close()

    if plot_type in ("combined", "both"):
        plt.figure(figsize=(10, 8))  # Larger figure for combined plot
        for i in range(data.shape[1]):
            plt.plot(times, data[:, i], label=f"{legends[i]} ({units[i]})")
        plt.xlabel("Time (ps)")
        plt.ylabel("Values")  # More general y-label for combined plot
        plt.title("All Properties vs. Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        output_filename = os.path.join(output_dir, f"{prefix}combined_plot.png")
        plt.savefig(output_filename)
        plt.close()



def main():
    parser = argparse.ArgumentParser(description="Plot data from an XVG-like file.")
    parser.add_argument("-f", "--filename", type=str, required=True,
                        help="Path to the input XVG-like file.")
    parser.add_argument("-o", "--output_dir", type=str, default="plots",
                        help="Directory to save the plots (default: 'plots').")
    parser.add_argument("-p", "--prefix", type=str, default="",
                        help="Optional prefix for plot filenames.")
    parser.add_argument("-t", "--plot_type", type=str, default="separate",
                        choices=["separate", "combined", "both"],
                        help="Type of plot(s) to generate: 'separate', 'combined', or 'both' (default: 'separate').")

    args = parser.parse_args()

    try:
        times, data, legends, units = parse_xvg_data(args.filename)
    except FileNotFoundError:
        print(f"Error: File not found: {args.filename}")
        return
    except ValueError as e:
        print(f"Error parsing file: {e}")
        return

    if data.size == 0:
        print("Error: No data found in the file.")
        return

    plot_data(times, data, legends, units, args.output_dir, args.prefix, args.plot_type)
    print(f"Plots saved to the '{args.output_dir}' directory.")


if __name__ == "__main__":
    main()

# How to use
# See Obsidan notes
# 02 NPT Equilibration