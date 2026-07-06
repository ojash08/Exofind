import matplotlib.pyplot as plt


def plot_periodogram(results):

    periods = [r["period"] for r in results]
    scores = [r["score"] for r in results]

    plt.figure(figsize=(12,5))

    plt.plot(
        periods,
        scores,
        linewidth=2
    )

    plt.xlabel("Trial Period (days)")
    plt.ylabel("BLS Power")
    plt.title("ExoFind BLS Periodogram")

    plt.grid(alpha=0.3)

    plt.show()