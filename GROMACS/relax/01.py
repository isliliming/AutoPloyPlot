import matplotlib.pyplot as plt
import numpy as np

def plot_energy_xvg(filename="energy.xvg"):
    """
    Plots the Potential Energy from a GROMACS .xvg energy file.

    Args:
        filename (str, optional): The name of the .xvg file. Defaults to "energy.xvg".
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
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.plot(time, potential_energy, label='Potential Energy (kJ/mol)')

    # Add labels and title
    plt.xlabel('Time (ps)')
    plt.ylabel('Potential Energy (kJ/mol)')
    plt.title('GROMACS Potential Energy vs. Time')
    # set the x-axis limits
    # plt.xlim(4000,time[-1])
    plt.grid(True)  # Add a grid for better readability
    plt.legend()

    # Save the plot
    # plt.savefig("potential_energy.png")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_energy_xvg("energy.xvg")  # Replace "energy.xvg" with your filename if different