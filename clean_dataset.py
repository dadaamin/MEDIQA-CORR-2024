import json
import csv
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input_path")
parser.add_argument("--output_path")
args = parser.parse_args()

train_samples = []
with open(args.input_path) as f:
    for row in list(csv.reader(f))[1:]:
        if row[0].isnumeric():            
            row = row[1:]
        sample = {
            "text_id": row[0],
            "text": row[1],
            "sentences": row[2].split("\n")            
            }
        if len(row) > 3:
            sample["has_error"] = True if row[3] == "1" else False 
            sample["error_index"] = row[4]
            sample["wrong_sentence"] = row[5]
            sample["corrected_sentence"] = row[6]

        
        train_samples.append(sample)


cleaned_dataset = []
for example in train_samples:
    sentences = example["sentences"]
    sentences = [sent for sent in sentences if sent.strip()]
    new_sentences = []
    i = 0
    last_idx = 0
    
    while i < len(sentences):
        j = 1
        current_merge = sentences[i]
        if i < len(sentences) - 1:
            while i + j < len(sentences) and sentences[i + j].split()[0] != str(last_idx + 1):
                current_merge += " " + sentences[i + j]
                j += 1
        i += j
        last_idx += 1
        new_sentences.append(current_merge)
    example["sentences"] = new_sentences

    cleaned_dataset.append(example)

with open(args.output_path, "w") as f_w:
    json.dump(cleaned_dataset, f_w, ensure_ascii=False)