import json
import sys


def check_formatting(filepath: str, gold_data: dict) -> bool:
    """
    Take a prediction filepath as input and print on the command line whether there
    are any peculiarities about the prediction formatting.

    :param filepath: str path to predictions file (should be jsonl)
    :param gold_data: dict samples of gold data
    :return: True = Can Parse, False = Cannot Parse
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    questionable_lines = []
    error_lines = []
    expected_ids = [id for id, label in gold_data.items()]
    line_ids = []
    missing_lines = []

    # TODO: check the other keys
    for line in lines:
        try:
            json_line = json.loads(line)
            pred = json_line["prediction"]
            pred = int(pred)
            if pred not in [1, 2, 3, 4, 5]:
                questionable_lines.append(line)

            index = str(json_line["id"])
            if index in expected_ids:
                expected_ids.remove(index)
            else:
                print("Error: The following id is unexpected or in the prediction file multiple times. ", index)
                error_lines.append(line)

        except:
            error_lines.append(line)


    if error_lines:
        print("Error: The following lines are malformatted or have wrong indices.")
        print(error_lines)
        print("The above lines are malformatted or have wrong indices. Please make sure each line is a valid json and check that each index appears exactly once.")
        return False

    if expected_ids:
        print("Error: The following ids are expected, but not in the predictions file. ", expected_ids)
        if len(expected_ids) > 100:
            print("Maybe a mismatch between the predictions file and the test set?")
        return False


    if questionable_lines:
        print("Warning: The following lines do not have expected values (1-5). Evaluation can still take place but please check your data.")
        print(questionable_lines)
        
    return True


    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give the file to check formatting of as an argument")

    with open("input/ref/solution.jsonl", 'r', encoding='utf-8') as f:
        data_dict = {}
        for line in f:
            import json
            item = json.loads(line)
            data_dict[item["id"]] = item["label"]
            
    check_formatting(sys.argv[1], data_dict)
