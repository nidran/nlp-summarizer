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
        # gold = fp.readlines()
        gold = fp.read().strip().split()
    with open(os.path.join(generated_dir, filename), "r") as fp:
        # generated = fp.readlines()
        generated = fp.read().strip().split()

    gold_str = " ".join(gold)
    generated_str = " ".join(generated)

    skip = False
    if len(gold) == 0:
        # empty_gold.append(filename)
        print(count)
        skip = True
    if len(generated) == 0:
        # empty_generated.append(filename)
        skip = True
        print(count)

    if skip:
        continue


    # corpus-bleu
    gold_list.append([gold])
    generated_list.append(generated)


    # gen_list.append(generated_str)
    # gold_list.append(gold_str)

    # gen_list.append(generated)
    # gold_list.append(gold)


    # print("Gold: {}\n".format(gold_str))
    # print("Generated: {}\n".format(generated_str))
    # print("Rouge Score: {}".format(rouge.get_scores(generated_str, gold_str), avg=False))
    # print()

    
    if count == 20000:
        break

    # val_1 += rouge.get_scores(generated_str, gold_str)[0]['rouge-l']['f']    
    # val_2 += scorer.score(generated_str, gold_str)['rougeLsum'].fmeasure

    # val_1 = rouge.get_scores(generated_str, gold_str)
    rouge_per = scorer.score(generated_str, gold_str)
    bleu_per = sentence_bleu([gold], generated)

    results = {'rouge': rouge_per, 'bleu': bleu_per}

    with open(os.path.join(output_dir, filename), "w") as fp:
        fp.write(str(results))


    total_2 += rouge_per['rougeLsum'].fmeasure

    # t = []
    # for r in ref:
    #     # print(r)
    #     c = r.split()
    #     t.append(c)

    # ref = t
    # can = candidate.split()
    # # print(ref, can)
    # # print(can)
    # print(sentence_bleu(ref, can))
    # total_blue


# print(generated_list)
# print("Rouge Score: {}".format(total_1/count))
print("Rouge Score: {}".format(total_2/count))

# print(empty_gold)

# print("Rouge Scorer: {}".format(scorer.score(gold_str, gen_list)))
# print()
# print("Bleu Score: {}".format(sentence_bleu(gold_str.split(), generated_str.split())))
print("Bleu score of corpus: {}".format(corpus_bleu(gold_list, generated_list)))
# print()
    # if input() == "y":
    #     pass
    # else:
    #     break