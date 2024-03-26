import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, required=True, nargs='+')
    args = parser.parse_args()

    for label in args.experiments:
        value = pd.read_csv(label + "/total_episode_reward.csv")['Total Reward'].values
        plt.plot(value, label=label)
    
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Total reward per episode")
    plt.savefig("analysis_rewardperepisode.png")
    plt.grid()

    plt.close("all")

    for label in args.experiments:
        value = pd.read_csv(label + "/episode_reward.csv")['Reward every 1000'].values
        plt.plot(value, label=label)
    
    plt.legend()
    plt.xlabel("Step (x1000)")
    plt.ylabel("Average reward every 1000 steps")
    plt.savefig("analysis_rewardper1000.png")
    plt.grid()

    plt.close("all")

    for label in args.experiments:
        value = pd.read_csv(label + "/losses.csv")['q1_loss'].values
        plt.plot(value, label=label)
    
    plt.legend()
    plt.xlabel("Update")
    plt.ylabel("Q loss")
    plt.savefig("analysis_losses.png")
    plt.grid()

    plt.close("all")

    for label in args.experiments:
        value = pd.read_csv(label + "/losses.csv")['track_q1'].values
        plt.plot(np.arange(len(value))*1000, value, label=label)

    plt.legend()
    plt.xlabel("Update")
    plt.ylabel("Average q1")
    plt.savefig("avgq1.png")
    plt.grid()

    plt.close("all")