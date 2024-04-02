from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
import random
import os

import torch
from sentence_transformers import SentenceTransformer
from finch import FINCH
from openai import OpenAI
# Insert API key
client = OpenAI(api_key="")
def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
set_all_seeds(42)
from scipy.spatial.distance import cdist

from candidate_filtering.sentence_classification_model import SentenceClassificationModel

embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO").cuda()
embedding_dim = embedding_model[1].pooling_output_dimension
sentence_classifier = SentenceClassificationModel(embedding_dim).cuda()
sentence_classifier.load_state_dict(torch.load("candidate_filtering/checkpoint/sentence_classification.pt"))


system_prompt_ms = """"You are a highly skilled assistant, specifically trained to assist medical professionals. You will be given a report and a single sentence from the report that might contain a medical reasoning error or inaccuracy. Assess step-by-step if the sentence is sound or could be replaced by a more accurate sentence. Give a clear yes or no answer. If you think the sentence should be corrected or adjusted, please provide a corrected version. If you think the sentence is sound write \"NA\" instead of the correction. Provide a step by step reasoning for your prediction.
"""

system_prompt_uw = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will be given a report that might contain a single medical reasoning error or none. Your job is to assess step-by-step if all the sentences in the given reports are correct or if one sentence is incorrect. Give a clear yes or no answer. If you think a sentence should be corrected or adjusted, please provide a corrected version. Provide a step by step reasoning for your prediction."""

user_prompt_ms = """
Report: {report}

Sentence: {sentence}"""

response_ms= """
Reasoning: {reasoning}
----------------------------
Is correct: {correct}
Corrected sentence: {corrected}"""

user_prompt_uw = """
Report: {report}
"""

response_uw= """
Reasoning: {reasoning}
----------------------------
Is correct: {correct}
Wrong sentence: {wrong_sentence}
Corrected sentence: {corrected}"""



chain_of_thought_system_prompt_ms = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will be given a report without a medical reasoning error. Describe step-by-step how you would confirm that the report is correct. Do not mention that the solution was provided to you. Pretend you are responding directly to a request to check this report without knowing the answer beforehand."""

chain_of_thought_system_prompt_uw_correct = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will be given a report without a medical reasoning error. Describe step-by-step how you would confirm that the report is correct. Do not mention that the solution was provided to you. Pretend you are responding directly to a request to check this report without knowing the answer beforehand."""

chain_of_thought_system_prompt_uw_error = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will be given a report with a medical reasoning error and a corrected sentence. Describe step-by-step how you would find and correct this error. Do not mention that the solution was provided to you. Pretend you are responding directly to a request to check this report without knowing the answer beforehand."""


chain_of_thought_user_prompt_ms ="""
--------------BEGIN REPORT--------------
{report}
--------------END REPORT--------------
"""

chain_of_thought_user_prompt_error ="""
--------------BEGIN REPORT--------------
{report}
--------------END REPORT--------------
Wrong Sentence: {wrong_sentence}
Corrected Sentence: {corrected_sentence}
"""





def find_correct_sentence_embeddings(samples):
    correct_samples = [sample for sample in samples if not sample["has_error"]]
    correct_sentences_examples = []
    correct_sentence_embeddings = np.empty((len(correct_samples), 768))
    for i, sample in enumerate(tqdm(correct_samples, desc="Computing MS correct sentence embeddings")):
        sentence_embeddings = torch.Tensor(embedding_model.encode(sample["sentences"]))
        error_candidate_index = torch.argmax(sentence_classifier(sentence_embeddings)).item()
        correct_sentence_embeddings[i] = sentence_embeddings[error_candidate_index]
        correct_sentences_examples.append((sample["sentences"][error_candidate_index], sample["sentences"]))
    return correct_sentence_embeddings, correct_sentences_examples

def find_correct_embeddings_uw(samples):
    correct_samples = [sample for sample in samples if not sample["has_error"]]
    correct_examples = []
    correct_embeddings = np.empty((len(correct_samples), 768))
    
    
    for i, sample in enumerate(tqdm(correct_samples, desc="Computing UW correct embeddings")):
        sentences = [" ".join(sent.split(" ")[1:]) for sent in sample["sentences"]]
        sentence_embeddings = embedding_model.encode(sentences)
        correct_embeddings[i] = sentence_embeddings.mean()
        correct_examples.append(sample["sentences"])
    return correct_embeddings, correct_examples

def find_error_embeddings_uw(samples):
    error_samples = [sample for sample in samples if sample["has_error"]]
    error_examples = []
    error_embeddings = np.empty((len(error_samples), 768))
    
    for i, sample in enumerate(tqdm(error_samples, desc="Computing UW error embeddings")):
        sentences = [" ".join(sent.split(" ")[1:]) for sent in sample["sentences"]]
        sentence_embeddings = embedding_model.encode(sentences)
        error_embeddings[i] = sentence_embeddings.mean()
        error_examples.append((sample["sentences"][int(sample["error_index"])], sample["sentences"], sample["corrected_sentence"]))
    return error_embeddings, error_examples

def embed_error_sentences(samples):
    samples_with_error = [sample for sample in samples if sample["has_error"]]
    error_sentences_examples = []
    error_sentence_embeddings = np.empty((len(samples_with_error), 768))
    for i, sample in enumerate(tqdm(samples_with_error, desc="Computing MS error sentence embeddings")):
        sentence_embeddings = embedding_model.encode(sample["sentences"][int(sample["error_index"])])
        error_sentence_embeddings[i] = sentence_embeddings
        error_sentences_examples.append((sample["sentences"][int(sample["error_index"])], sample["sentences"], sample["corrected_sentence"]))
    return error_sentence_embeddings, error_sentences_examples


def predict_ms(sentences, sentence, few_shots):
    # Remove number at the beginning of the sentence
    sentence = " ".join(sentence.split(" ")[1: ])
    sentences = [" ".join(sent.split(" ")[1:]) for sent in sentences]
    
    report = " ".join(sentences)
    messages=[
        {"role": "system", "content": system_prompt_ms}
        ]
    for shot in few_shots:
        shot_report = [" ".join(s.split(" ")[1:]) for s in shot["text"]]
        shot_report = " ".join(shot_report)
        
        shot_sent = " ".join(shot["sentence"].split(" ")[1: ])
        
        if shot["is_correct"]:
            messages.extend([{"role": "user", "content": user_prompt_ms.format(report=shot_report, sentence=shot_sent)},
                            {"role": "assistant", "content": response_ms.format(reasoning=shot["CoT"], correct="Yes", corrected="NA")}])
        else:
            messages.extend([{"role": "user", "content": user_prompt_ms.format(report=shot_report, sentence=shot_sent)},
                            {"role": "assistant", "content": response_ms.format(reasoning=shot["CoT"], correct="No", corrected=shot["corrected_sentence"])}])
    messages.append({"role": "user", "content": user_prompt_ms.format(report=report, sentence=sentence)})
    completion = client.chat.completions.create(
    model="gpt-4-0613",
    messages=messages)

    
    return completion.choices[0].message.content

def predict_uw(report, few_shots):
    # Remove number at the beginning of the sentence
    
    messages=[
        {"role": "system", "content": system_prompt_uw}
        ]
    for shot in few_shots:
        
        
        if not "corrected_sentence" in shot:
            messages.extend([{"role": "user", "content": user_prompt_uw.format(report=shot["text"])},
                            {"role": "assistant", "content": response_uw.format(reasoning=shot["CoT"], correct="Yes", corrected="NA", wrong_sentence="NA")}])
        else:
            messages.extend([{"role": "user", "content": user_prompt_uw.format(report=shot["text"])},
                            {"role": "assistant", "content": response_uw.format(reasoning=shot["CoT"], correct="No", corrected=shot["corrected_sentence"], wrong_sentence=shot["wrong_sentence"])}])
    messages.append({"role": "user", "content": user_prompt_uw.format(report=report)})
    completion = client.chat.completions.create(
    model="gpt-4-0613",
    messages=messages)

    
    return completion.choices[0].message.content
   
def generate_chain_of_thought(sentences, wrong_sentence, corrected_sentence):
    # Remove number at the beginning of the sentence
    #sentence = " ".join(sentence.split(" ")[1: ])
    sentences = [" ".join(sent.split(" ")[1:]) for sent in sentences]
    
    report = " ".join(sentences)
    
    completion = client.chat.completions.create(
    model="gpt-4-0613",
    messages=[
        {"role": "system", "content": chain_of_thought_system_prompt_uw_error},
        {"role": "user", "content": chain_of_thought_user_prompt_error.format(report=report, wrong_sentence=wrong_sentence, corrected_sentence=corrected_sentence)}
        ])
    
    return completion.choices[0].message.content
    
        
def find_shot(shots, i):
    for shot in shots:
        if shot["cluster"] == i:
            return shot
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trainset", default="MEDIQA-CORR-2024-MS-TrainingData-final.json", type=str,  help="The path to the dataset")
    parser.add_argument("--ms_evalset", default="MEDIQA-CORR-2024-MS-ValidationSet-final.json", type=str,  help="The path to the dataset")
    parser.add_argument("--uw_evalset", default="MEDIQA-CORR-2024-UW-ValidationSet.json", type=str,  help="The path to the dataset")
    parser.add_argument("--testset", default="MEDIQA-CORR-Official-Test-Set.json", type=str,  help="The path to the dataset")
    parser.add_argument("--generate_cot", action="store_true")
    args = parser.parse_args()

    with open(args.trainset) as f:
        train_data = json.load(f)
    with open(args.ms_evalset) as f:
        ms_eval_data = json.load(f)
    
    ms_data = train_data + ms_eval_data
    correct_sentence_embeddings, correct_sentences_examples = find_correct_sentence_embeddings(ms_data)
    error_sentence_embeddings, error_sentences_examples = embed_error_sentences(ms_data)
    
    correct_c, correct_num_clust, _ = FINCH(correct_sentence_embeddings, distance='cosine')
    correct_cluster = correct_c[:, 1]
    correct_cluster_num = correct_num_clust[1]
    correct_centroids = [correct_sentence_embeddings[correct_cluster == i].mean(axis=0) for i in range(correct_cluster_num)]
    correct_representative = [cdist(correct_sentence_embeddings[correct_cluster == i],correct_centroids[i].reshape(1, -1)).argmin() for i in range(correct_cluster_num)]
    
    correct_sentences_examples = np.asarray(correct_sentences_examples, dtype=object)
    correct_sentences_examples_by_cluster = [correct_sentences_examples[correct_cluster == i] for i in range(correct_cluster_num)]
    correct_sentences_examples_representatives = [correct_sentences_examples_by_cluster[i][correct_representative[i]] for i in range(correct_cluster_num)]
    
    error_c, error_num_clust, _ = FINCH(error_sentence_embeddings, distance='cosine')
    error_cluster = error_c[:, 1]
    error_cluster_num = error_num_clust[1]
    error_centroids = [error_sentence_embeddings[error_cluster == i].mean(axis=0) for i in range(error_cluster_num)]
    error_representative = [cdist(error_sentence_embeddings[error_cluster == i],error_centroids[i].reshape(1, -1)).argmin() for i in range(error_cluster_num)]
    
    error_sentences_examples = np.asarray(error_sentences_examples, dtype=object)
    error_sentences_examples_by_cluster = [error_sentences_examples[error_cluster == i] for i in range(error_cluster_num)]
    error_sentences_examples_representatives = [error_sentences_examples_by_cluster[i][error_representative[i]] for i in range(error_cluster_num)]
    
    
    if args.generate_cot:
        for i in range(1, correct_cluster_num):
            cot = generate_chain_of_thought(correct_sentences_examples_representatives[i][0], 
                                    correct_sentences_examples_representatives[i][1],
                                    False)
            
            with open("ms_correct_cot.json", "a") as f_a:
                f_a.write(json.dumps({
                    "cluster": i,
                    "sentence": correct_sentences_examples_representatives[i][0],
                    "text": correct_sentences_examples_representatives[i][1],
                    "CoT": cot 
                }, ensure_ascii=False) + "\n")
                
        for i in tqdm(range(error_cluster_num)):
            cot = generate_chain_of_thought(error_sentences_examples_representatives[i][0], 
                                        error_sentences_examples_representatives[i][2],
                                    error_sentences_examples_representatives[i][1],
                                    True)
            
            with open("ms_error_cot.json", "a") as f_a:
                f_a.write(json.dumps({
                    "cluster": i,
                    "sentence": error_sentences_examples_representatives[i][0],
                    "corrected_sentence": error_sentences_examples_representatives[i][2],
                    "text": error_sentences_examples_representatives[i][1],
                    "CoT": cot 
                }, ensure_ascii=False) + "\n")
                
    
    #load cot
    with open("ms_correct_cot.json") as f:
        correct_cot = [json.loads(l) for l in f]
    with open("ms_error_cot.json") as f:
        error_cot = [json.loads(l) for l in f]

    with open(args.testset) as f:
        test_data = json.load(f)
    
    for sample in tqdm(test_data):
        if sample["text_id"].startswith("ms"):
            # Embed sentences
            sentence_embeddings = torch.Tensor(embedding_model.encode(sample["sentences"]))
            # Predict most probable sentence with error 
            error_candidate_index = torch.argmax(sentence_classifier(sentence_embeddings)).item()
            error_candidate = embedding_model.encode(sample["sentences"][error_candidate_index])
            # Find nearstest neighbor of correct sentences and error sentences
            correct_cluster = cdist(error_candidate.reshape(1, -1), np.array(correct_centroids)).argmin()
            error_cluster = cdist(error_candidate.reshape(1, -1), np.array(error_centroids)).argmin()
            
            correct_shot = find_shot(correct_cot, correct_cluster)
            correct_shot["is_correct"] = True
            error_shot = find_shot(error_cot, error_cluster)
            error_shot["is_correct"] = False
            
            shots = [correct_shot, error_shot]
            prediction = predict_ms(sample["sentences"], sample["sentences"][error_candidate_index], shots)
            
            
            with open("ms_predictions.json", "a") as f_a:
                f_a.write(json.dumps({"text_id": sample["text_id"], "sentence": sample["sentences"][error_candidate_index], "prediction": prediction}, ensure_ascii=False) + "\n")
        else:
            prediction = predict_uw(sample["text"], correct_cot + error_cot)
            with open("uw_predictions.json", "a") as f_a:
                f_a.write(json.dumps({"text_id": sample["text_id"], "prediction": prediction}, ensure_ascii=False) + "\n")
            

            
                        
                
        

            
            
            
    

    