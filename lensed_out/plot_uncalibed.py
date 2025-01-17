from utils import load_spec_data
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load the spectrum data
    spectra = load_spec_data(group = "science")

    for name, item in spectra.items():
        plt.plot(item[0], item[1], label = name)

    plt.legend()
    plt.show()