import sys
import os
import json
import statistics
from scipy.stats import spearmanr

"""
Usage: python3 evaluate.py filepath set
filepath: Path to the predictions .jsonl file, e.g. predictions/random_predictions.jsonl 
set: The split to test on (dev, test, train) - Refers to the files in data/ directory.
Spearman and Accuracy scores will be printed on command line.
"""

from format_check import check_formatting

def get_standard_deviation(l):
    return statistics.stdev(l)

def get_average(l):
    return sum(l)/len(l)

def is_within_standard_deviation(prediction, labels):
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    # Is prediction within the range of the average +/- the standard deviation?
    if (avg - stdev) < prediction < (avg + stdev):
        return True

    # Is the distance between average and prediction less than one?
    if abs(avg - prediction) < 1:
        return True

    # If neither one applies, then this prediction will be counted as "wrong".
    return False

def spearman_evaluation_score(predictions_filepath: str, gold_data: dict):
    """
    Get the spearman score for a prediction filepath on the gold data.
    It calculates the correlation between the list of predictions and the list of human averages.
    Score is printed on command line.
    """
    gold_list = ["-"] * len(gold_data)
    pred_list = ["-"] * len(gold_data)

    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    for line in pred_lines:
        line = json.loads(line)
        gold_list[int(line["id"])] = get_average(gold_data[str(line["id"])]["choices"])
        pred_list[int(line["id"])] = line["prediction"]

    corr, value = spearmanr(pred_list, gold_list)
    print(f"----------\nSpearman Correlation: {corr}\nSpearman p-Value: {value}")



def accuracy_within_standard_deviation_score(predictions_filepath, gold_data):
    """
    Get the Acc. w/in SD score.
    It calculates the proportion of samples where the prediction is within either 1 or +/- standard deviation 
    of the average human judgment.
    Score is printed on command line.
    """
    with open(predictions_filepath, "r") as f:
        pred_lines = f.readlines()

    correct_guesses = 0
    wrong_guesses = 0

    for line in pred_lines:
        line = json.loads(line)
        labels = gold_data[str(line["id"])]["choices"]
        if is_within_standard_deviation(line["prediction"], labels):
            correct_guesses += 1
        else:
            print(line["id"])
            wrong_guesses += 1

    print(f"----------\nAccuracy: {correct_guesses / (correct_guesses + wrong_guesses)} ({correct_guesses}/{correct_guesses+wrong_guesses})")



if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) < 3:
        print("Please provide the path to the jsonl predictions file and the set to evaluate on (dev/test). \n" 
        "Example Usage: python3 evaluate.py predictions/random_predictions.jsonl dev")
        sys.exit()

    predictions_filepath = arguments[1]
    if not os.path.exists(predictions_filepath):
        print("Not a valid predictions filepath, file not found: ", predictions_filepath)
        sys.exit()

    testset = arguments[2]
    try:
        with open("data/" + testset + ".json", "r") as f:
            gold_data = json.load(f)
    except:
        print("No file data/" + testset + ".json found. Make sure the name of the set (train/dev/test) is specified in command line argument #2")
        sys.exit()

    if not check_formatting(predictions_filepath, gold_data):
        sys.exit()

    print(f"Everything looks OK. Evaluating file {predictions_filepath} on data/{testset}.json...")

    spearman_evaluation_score(predictions_filepath, gold_data)
    accuracy_within_standard_deviation_score(predictions_filepath, gold_data)

