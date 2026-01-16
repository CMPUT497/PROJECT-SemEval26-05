# semeval26-05-scripts
Some scripts for Semeval 2026 Task 5. Equivalent to the scoring script on the CodaBench task. More baselines to be added later.

Link to submission website: https://www.codabench.org/competitions/10877/?secret_key=e3c13419-88c6-4c13-9989-8e694a2bc5c0

# How to evaluate predictions

First, remember to install the requirements.

To evaluate a prediction, please format it like the "predictions/[...].jsonl" files.
Each prediction must be in its own line. The "id" key corresponds to the keys of the samples in the gold data ("0", "1", etc).
The prediction key should be an integer between 1 and 5.

Once you have prepared your prediction data, put it in the predictions/ folder (replacing the existing file) and call the evaluation script like this:

```
python3 evaluate.py ./predictions/{prediction_file}.JSONL dev
```

Scores will be printed. If your predictions file contains bad formatting or is incomplete, it will print an error.

To submit to CodaBench, zip the predictions.jsonl up and upload it to the "My Submissions" tab on the task website.

Test set is yet unreleased, so you can only test on the dev set for now. The samples (including labels) are public here: https://github.com/Janosch-Gehring/ambistory