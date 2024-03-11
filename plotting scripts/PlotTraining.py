from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def get_expo_smoothing(data, alpha):

    t = data[0]

    smoothed_data = [t]
    for x in data[1:]:
        t = t*alpha + x*(1-alpha)
        smoothed_data.append(t)

    return smoothed_data


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)
    
    print(df)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.lineplot(data=df.loc[:, "num steps"], ax=axes[0], markers=True, label="not smoothed", alpha=0.6)
    axes[0].set_title("Number of steps")
    axes[0].set_ylabel("Number of steps")

    num_steps_data = df.loc[:, "num steps"].to_numpy()
    num_steps_data_smoothed = get_expo_smoothing(num_steps_data, alpha=0.9)

    sns.lineplot(data=num_steps_data_smoothed, ax=axes[0], markers=True, label="smoothed")
     
    sns.lineplot(data=df.loc[:, "policy loss"], ax=axes[1], markers=True)
    axes[1].set_title("Policy loss")
    axes[1].set_ylabel("Loss")

    sns.lineplot(data=df.loc[:, "value loss"], ax=axes[2], markers=True)
    axes[2].set_title("Value loss")
    axes[2].set_ylabel("Loss")

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/Training.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")