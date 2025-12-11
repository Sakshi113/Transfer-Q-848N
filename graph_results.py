import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# Function to add centered value labels
def add_labels(x, y, size=18):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center', fontsize=size)  # Aligning text at center

def plot_hh_rlhf():
    methods = ['Single[unaligned]', 'Single[indirect]', 'Agentic[unaligned]', 'Agentic[indirect]']
    mean_scores = [-3.67, -4.29, -4.29, -4.26]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, mean_scores)
    add_labels(methods, mean_scores)
    plt.ylim([-7, 0])
    plt.ylabel('Mean Scores HH-RLHF', fontsize=16)
    # plt.show()
    plt.savefig("mean_scores_hh_rlhf.png")


def plot_safenlp():
    methods = ['Single[unaligned]', 'Single[indirect]', 'Agentic[unaligned]', 'Agentic[indirect]']
    mean_scores = [-5.04, -5.60, -4.88, -5.24]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, mean_scores)
    add_labels(methods, mean_scores)
    plt.ylabel('Mean Scores SafeNLP')
    plt.ylim([-7, 0])
    # plt.show()
    plt.savefig("mean_scores_safenlp.png")

def plot_safenlp_santa():
    methods = ['Single[unaligned]', 'Single[indirect]', 'Agentic[unaligned]', 'Agentic[indirect]']
    mean_scores = [-1.19, -0.97, -1.30, -0.99]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, mean_scores)
    add_labels(methods, mean_scores)
    plt.ylabel('Mean Scores SafeNLP with Harmless Reward Model')
    plt.ylim([-2, 0])
    # plt.show()
    plt.savefig("mean_scores_safenlp_santa.png")

if __name__=="__main__":
    # plot_hh_rlhf()
    # plot_safenlp()
    plot_safenlp_santa()