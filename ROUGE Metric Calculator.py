import evaluate

references = [["I really did not like Game of Thrones Season 8.", "I disliked Game of Thrones Season 8.", "I hated Game of Thrones Season 8.", "Game of Thrones Season 8 was not to my liking."]]

prediction = ["I really did really did really did really did really did not like Game of Thrones Season 8."]
              
rouge = evaluate.load("rouge")

results = rouge.compute(predictions=prediction, references=references)

print(f"evaluate:{results}")

def ROUGE(predictions=None, references=None, microaveraging=False, casesensitive=True):
    import math
    import re
    from collections import Counter
    from statistics import geometric_mean
    from difflib import SequenceMatcher, Match
    import nltk
    from nltk.tokenize import word_tokenize

    if not casesensitive:
        predictions = [pred.lower() for pred in predictions]
        references = [[ref.lower() for ref in ref_list] for ref_list in references]

    #tokenization
    predictions = [re.sub(r'[^\w\s\']', ' ', pred) for pred in predictions]
    predictions = [re.sub(r'\s+', ' ', pred) for pred in predictions]

    references = [[re.sub(r'[^\w\s\']', ' ', ref) for ref in ref_list] for ref_list in references]
    references = [[re.sub(r'\s+', ' ', ref) for ref in ref_list] for ref_list in references]
    #TODO implement proper tokenization

    rouge1_recall = 0.0 
    rouge1_precision = 0.0
    rouge1_F1 = 0.0

    rouge2_recall = 0.0
    rouge2_precision = 0.0
    rouge2_F1 = 0.0

    rougeL_recall = 0.0
    rougeL_precision = 0.0
    rougeL_F1 = 0.0
    rougeLsum = 0.0

    rouge1_recall_scores = []
    rouge1_precision_scores = []
    rouge2_recall_scores = []
    rouge2_precision_scores = []
    rougeL_precision_scores = []
    rougeL_recall_scores = []

    if not microaveraging:
        #longest common sequence
        lcs = Match(a=0, b=0, size=0)
        for  i in range(len(predictions)):
            for reference in references[i]:
                if lcs.size < SequenceMatcher(None, reference.split(), predictions[i].split()).find_longest_match().size:
                    lcs = SequenceMatcher(None, reference.split(), predictions[i].split()).find_longest_match()
                    rougeL_recall = lcs.size/len(reference.split())
                    rougeL_precision = lcs.size/len(predictions[i].split())
            rougeL_precision_scores.append(rougeL_precision)
            rougeL_recall_scores.append(rougeL_recall)
            #TODO add functionality for non-continguous common substrings

        #tokenization
        #predictions = [word_tokenize(prediction) for prediction in predictions]
        #references = [word_tokenize(reference) for reference in references]

        #unigrams
        for i in range(len(predictions)):
            predicted_unigrams = predictions[i].split() 
            matched_unigrams_total = 0
            matched_unigrams_current = 0
            matched_unigrams = []
            for reference in references[i]:
                if (matched_unigrams_current < matched_unigrams_total) or (matched_unigrams_total == 0):
                    for unigram in predicted_unigrams:
                        if unigram in reference:
                            if reference.split().count(unigram) > matched_unigrams.count(unigram):
                                matched_unigrams_current += 1
                                matched_unigrams.append(unigram)
                    matched_unigrams_total = matched_unigrams_current
            rouge1_recall = matched_unigrams_total/len(reference.split())
            rouge1_precision = matched_unigrams_total/len(prediction[i].split())
            rouge1_recall_scores.append(rouge1_recall)
            rouge1_precision_scores.append(rouge1_precision)

        #bigrams
        for i in range(len(predictions)):
            try:
                predicted_bigrams = [tuple(predictions[i].split()[j:j+2]) for j in range(len(predictions[i].split()) - 1)]
                
                matched_bigrams_total = 0
                matched_bigrams_current = 0
                matched_bigrams = []
                for reference in references[i]:
                    reference_bigrams = [tuple(reference.split()[j:j+2]) for j in range(len(reference.split()) - 1)]
                    if (matched_bigrams_current < matched_bigrams_total) or (matched_bigrams_total == 0):
                        for bigram in predicted_bigrams:
                            if bigram in reference_bigrams:
                                if reference_bigrams.count(bigram) > matched_bigrams.count(bigram):
                                    matched_bigrams_current += 1
                                    matched_bigrams.append(bigram)
                        matched_bigrams_total = matched_bigrams_current
                        rouge2_recall = matched_bigrams_total/len(reference_bigrams)
                        rouge2_precision = matched_bigrams_total/len(predicted_bigrams)
            except ZeroDivisionError:
                pass
            rouge2_recall_scores.append(rouge2_recall)
            rouge2_precision_scores.append(rouge2_precision)



        rouge1_recall = sum(rouge1_recall_scores)/len(predictions)
        rouge1_precision = sum(rouge1_precision_scores)/len(predictions)
        rouge1_F1 = 2*((rouge1_recall*rouge1_precision)/(rouge1_recall + rouge1_precision))

        rouge2_recall = sum(rouge2_recall_scores)/len(predictions)
        rouge2_precision = sum(rouge2_precision_scores)/len(predictions)
        rouge2_F1 = 2*((rouge2_recall*rouge2_precision)/(rouge2_recall + rouge2_precision))

        rougeL_recall = sum(rougeL_recall_scores)/len(predictions)
        rougeL_precision = sum(rougeL_precision_scores)/len(predictions)
        rougeL_F1 = 2*((rougeL_recall*rougeL_precision)/(rougeL_recall + rougeL_precision))

    else: #microaveraging

        #longest common sequence
        lcs = Match(a=0, b=0, size=0)
        lcs_sizes = []
        lcs_reference_lengths = []
        lcs_prediction_lengths = []
        for  i in range(len(predictions)):
            lcs_prediction_lengths.append(len(predictions[i].split()))
            for reference in references[i]:
                if lcs.size < SequenceMatcher(None, reference.split(), predictions[i].split()).find_longest_match().size:
                    lcs = SequenceMatcher(None, reference.split(), predictions[i].split()).find_longest_match()
                    lcs_sizes.append(lcs.size)
                    lcs_reference_lengths.append(len(reference.split()))
                    rougeL_recall_scores.append(lcs.size/len(reference.split()))
                    rougeL_precision_scores.append(lcs.size/len(predictions[i].split()))
            rougeL_recall = sum(lcs_sizes) / sum(lcs_reference_lengths)
            rougeL_precision = sum(lcs_sizes) / sum(lcs_prediction_lengths)
            rougeL_F1 = 2*((rougeL_recall*rougeL_precision)/(rougeL_recall + rougeL_precision))
            #TODO add functionality for non-continguous common substrings

        #tokenization
        #predictions = [word_tokenize(prediction) for prediction in predictions]
        #references = [word_tokenize(reference) for reference in references]

        #unigrams
        total_elements = 0
        total_matches = 0
        total_predicted = 0
        for i in range(len(predictions)):
            predicted_unigrams = predictions[i].split() 
            total_predicted += len(predicted_unigrams)
            matched_unigrams_current = 0
            matched_unigrams_total = 0
            reference_length_current = 0
            matched_unigrams = []
            for reference in references[i]:
                if (matched_unigrams_current < matched_unigrams_total) or (matched_unigrams_total == 0):
                    for unigram in predicted_unigrams:
                        if unigram in reference:
                            if reference.split().count(unigram) > matched_unigrams.count(unigram):
                                matched_unigrams_current += 1
                                matched_unigrams.append(unigram)
                                reference_length_current = len(reference.split())
                    matched_unigrams_total = matched_unigrams_current
            matched_unigrams_total = matched_unigrams_current
            total_matches += matched_unigrams_total
            total_elements += reference_length_current

        rouge1_recall = total_matches / total_elements if total_elements > 0 else 0
        rouge1_precision = total_matches / total_predicted if total_predicted > 0 else 0
        rouge1_F1 = 2*((rouge1_recall*rouge1_precision)/(rouge1_recall + rouge1_precision))

        #bigrams
        for i in range(len(predictions)):
            try:
                predicted_bigrams = [tuple(predictions[i].split()[j:j+2]) for j in range(len(predictions[i].split()) - 1)]
                
                matched_bigrams_total = 0
                matched_bigrams_current = 0
                matched_bigrams = []
                total_elements = 0
                total_matches = 0
                total_predicted = 0

                total_predicted += len(predicted_bigrams)
                reference_length_current = 0
                matched_bigrams = []
                for reference in references[i]:
                    reference_bigrams = [tuple(reference.split()[j:j+2]) for j in range(len(reference.split()) - 1)]
                    if (matched_bigrams_current < matched_bigrams_total) or (matched_bigrams_total == 0):
                        for bigram in predicted_bigrams:
                            if bigram in reference_bigrams:
                                if reference_bigrams.count(bigram) > matched_bigrams.count(bigram):
                                    matched_bigrams_current += 1
                                    matched_bigrams.append(bigram)
                                    reference_length_current = len(reference_bigrams)
                        matched_bigrams_total = matched_bigrams_current
                        
                matched_bigrams_total = matched_bigrams_current
                total_matches += matched_bigrams_total
                total_elements += reference_length_current
                        
            except ZeroDivisionError:
                pass

        rouge2_recall = total_matches / total_elements if total_elements > 0 else 0
        rouge2_precision = total_matches / total_predicted if total_predicted > 0 else 0
        rouge2_F1 = 2*((rouge2_recall*rouge2_precision)/(rouge2_recall + rouge2_precision))


            
    return f"'ROUGE-1 recall': {rouge1_recall}, 'ROUGE-1 precision': {rouge1_precision}, 'ROUGE-1 F-1': {rouge1_F1}, 'ROUGE-2 recall': {rouge2_recall}, 'ROUGE-2 precision': {rouge2_precision},'ROUGE-2 F-1': {rouge2_F1} 'ROUGE-L recall': {rougeL_recall}, 'ROUGE-L precision': {rougeL_precision}, 'ROUGE-L F-1': {rougeL_F1}"

print(f"macro: {ROUGE(predictions=prediction, references=references, microaveraging=False)}")
print(f"micro: {ROUGE(predictions=prediction, references=references, microaveraging=True)}")