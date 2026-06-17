"""
python3 scripts/plot_results.py

"""


import pandas as pd
import matplotlib.pyplot as plt


def main():
    # CSV file from evaluator_node
    csv_file = "trajectory_log.csv"

    # Read CSV
    data = pd.read_csv(csv_file)

    # Show available sources
    print("Available sources:")
    print(data["source"].unique())

    # Create figure
    plt.figure(figsize=(8, 6))

    sources_to_plot = ["amcl", "kf", "ekf", "pf"]

    for source in sources_to_plot:
        source_data = data[data["source"] == source]
        if source_data.empty:
            continue

        plt.plot(
            source_data["x"],
            source_data["y"],
            label=source
        )
    # Plot x-y trajectory for each source
#    for source in data["source"].unique():
#        source_data = data[data["source"] == source]

#        plt.plot(
#            source_data["x"],
#            source_data["y"],
#            label=source
#        )

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("Trajectory comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Save and show plot
    plt.savefig("trajectory_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

