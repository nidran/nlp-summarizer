import os
from rouge import Rouge
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

gold_dir = "/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/Generated_Data/Generated_Summaries/TextRank/Gold/"
generated_dir = "/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/Generated_Data/Generated_Summaries/TextRank/Text/"

output_dir = "/Users/arnabgupta/Documents/NYU/Coursework/Fall 21/2590 - Natural Language Processing/Project/scientific-paper-summarisation/Data/Generated_Data/Generated_Summaries/TextRank/Scores/"

rouge = Rouge()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], True)

count = 0
total_1 = 0
total_2 = 0

gold_list = []
generated_list = []

for filename in os.listdir(gold_dir):
    count += 1

    print(filename, '\t', count)
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

    
    # if count == 20000:
    #     break

    rouge_per = scorer.score(generated_str, gold_str)
    bleu_per = sentence_bleu([gold], generated)

    results = {'rouge': rouge_per, 'bleu': bleu_per}

    with open(os.path.join(output_dir, filename), "w") as fp:
        fp.write(str(results))


    total_2 += rouge_per['rougeLsum'].fmeasure

print("Rouge Score: {}".format(total_2/count))

print("Bleu score of corpus: {}".format(corpus_bleu(gold_list, generated_list)))
