# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    with open('birth_dev.tsv', encoding='utf-8') as f:
        true_places = [line.strip().split('\t')[1] for line in f]
        total = len(true_places)
        correct = len(list(filter(lambda x : x == 'London', true_places)))
    accuracy = correct * 100.0 / total
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
