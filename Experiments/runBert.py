import os
os.system('pip install bert-extractive-summarizer')
os.system('pip install rouge-score')
os.system('pip install rouge/requirements.txt')
os.system('pip install rouge')
os.system('pip install nltk')
os.system('pip install transformers --upgrade')
from rouge import Rouge
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction
import os
import glob
from tqdm import tqdm
from summarizer import Summarizer
import regex as re
import json 
rouge = Rouge()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], True)

model = Summarizer()
def prepare_paper( filename):
        paper_loc = filename
        with open("/scratch/nr2387/Data-4/Parsed_Papers/"+paper_loc, "r") as f:
            plaintext_paper = []
            for line in f.readlines():
                line = line.replace('@&#MAIN-TITLE@&#','').replace('@&#HIGHLIGHTS@&#','').replace('@&#KEYPHRASES@&#','').replace('@&#ABSTRACT@&#','').replace('@&#INTRODUCTION@&#','').replace('@&#DISCUSSION@&#','').replace('@&#REFERENCES@&#','')
                line = re.sub(' +', ' ', line)
                plaintext_paper.append(line)
        plaintext_paper = "".join(plaintext_paper)
        return plaintext_paper

def generateSummary():
    papersToParse =os.listdir("/scratch/nr2387/Data-4/Parsed_Papers")
    for idx, paper in enumerate(papersToParse):
        try:
            parsedTxt = prepare_paper(paper)
            result = model(parsedTxt, num_sentences=8)
            with open('/scratch/nr2387/Data-4/BertOutput-4/'+paper, "w+") as output:
                    output.write(result)
            if idx%500 == 0:
                print(str(idx) + "Papers parsed")
        except ValueError:
            with open('errorLog.txt', 'a') as errorLog:
                errorLog.write('Value Error Spacy for file '+paper)

def computeScore():
    gold_dir = "/scratch/nr2387/Data-4/Gold/"
    generated_dir = "/scratch/nr2387/Data-4/BertOutput-4/"
    output_dir = "scratch/nr2387/Data-4/Accuracy/"
    rouge = Rouge()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], True)
    count = 0
    total_1 = 0
    total_2 = 0
    gold_list = []
    generated_list = []
    resList = {}
    for filename in os.listdir(generated_dir):
        count += 1
        try:
            with open(os.path.join(gold_dir, filename), "r") as fp:
                gold = fp.read().strip().split()
            with open(os.path.join(generated_dir, filename), "r") as fp:
                generated = fp.read().strip().split()
            gold_str = " ".join(gold)
            generated_str = " ".join(generated)
            skip = False
            if len(gold) == 0:
                # print(count)
                skip = True
            if len(generated) == 0:
                skip = True
                # print(count)
            if skip:
                continue
            # corpus-bleu
            gold_list.append([gold])
            generated_list.append(generated)
            rouge_per = scorer.score(generated_str, gold_str)
           # bleu_per = sentence_bleu([gold], generated)
            bleu_per = sentence_bleu([gold], generated, weights=(1, 0.2, 0, 0), smoothing_function=SmoothingFunction(epsilon=1e-2).method1)

            results = {'filename' : filename, 'rouge': rouge_per, 'bleu': bleu_per}
            resList[filename] = results

            total_2 += rouge_per['rougeLsum'].fmeasure
        except FileNotFoundError:
            with open ("/scratch/nr2387/Data-4/Accuracy/Error4.txt", "a+") as fp:
              fp.write("Wrong file or file path"+ gold_dir+ filename)
           
    weights=(1, 0.2, 0, 0)
   # resList['scores']={'Rouge Score': total_2/count, 'Bleu Score':corpus_bleu(gold_list, generated_list)}
    resList['scores'] = {'Rouge Score': total_2/count, 'Bleu Score':corpus_bleu(gold_list, generated_list, weights=weights, smoothing_function=SmoothingFunction(epsilon=1e-2).method1)}
    with open("/scratch/nr2387/Data-4/NOS8NewBlue.json", "w+") as fp:
        json.dump(resList, fp)
              

if __name__ == "__main__":
    print('Started Generating Summary')
    generateSummary()
    print('Summaries generated')
    print('Started computing score')
    computeScore()
    print('Scores computed')
