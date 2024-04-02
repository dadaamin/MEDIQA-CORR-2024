## Format dataset in json and clean sentence splits

```
python clean_dataset.py --input_path <path_to_dataset> --output_path <output_path>
```

## Train classifier for sentences with error

This can be skipped as a model checkpoint is already provided under 
```
python candidate_filtering/train_sentence_classification_ms.py --train_dataset <path_to_train_dataset> --val_dataset <path_to_val_dataset>
```

## Predict solution

```
python main.py --trainset <path_to_train_dataset> --ms_evalset <path_to_ms_val> --uw_evalset <path_to_uw_val> --testset <path_to_testset> --generate_cot
```