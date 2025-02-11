import matplotlib.pyplot as plt
import numpy as np
import os

def plot_energy_xvg(filename):
    """
    Plots the Potential Energy from a GROMACS .xvg energy file.

    Args:
        filename (str, optional): The name of the .xvg file. Defaults to "energy_steps50000.xvg".
    """

    time = []
    potential_energy = []

    with open(filename, 'r') as f:
        for line in f:
            # Skip comment lines and lines starting with @
            if line.startswith(('#', '@')):
                continue

            try:
                # Split the line into time and potential energy values
                t, pe = map(float, line.split())
                time.append(t)
                potential_energy.append(pe)
            except ValueError:
                # Handle potential errors in the file format (e.g., empty lines)
                print(f"Skipping malformed line: {line.strip()}")
                continue

    # Convert lists to numpy arrays for easier plotting
    time = np.array(time)
    potential_energy = np.array(potential_energy)

    # Filter out the first 4000 ps
    time = time[5000:]
    potential_energy = potential_energy[5000:]

    # Create the plot
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    plt.plot(time, potential_energy,color='#332288', linewidth=1.5)

    # Add labels and title
    plt.xlabel('Time (ps)', fontsize=20)
    plt.ylabel('Potential Energy (kJ/mol)', fontsize=20)
    # plt.title('GROMACS Potential Energy vs. Time',  fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the plot
    plt.savefig(filename[:-3]+'png', dpi=300)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    files = os.listdir()
    for file in files:
        if file.endswith('.xvg'):
            filename = file
            print(f"Plotting {filename}")
            plot_energy_xvg(filename)  # Replace "energy_steps50000.xvg" with your filename if different
            print(f"Plot saved as: {filename[:-3]+'png'}")