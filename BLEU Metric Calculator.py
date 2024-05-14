import translators as ts
import evaluate

input = "ايش بتسوي يا ولد؟"
references = [["What will you do now?", "What are you doing?", "What will you be doing, bro?", "What are you doing, boy?"]]

prediction = [ts.translate_text(input)]

bleu = evaluate.load("bleu")

results = bleu.compute(predictions=prediction, references=references)

print(f"prebuilt:{results}")

def BLEU(predictions=None, references=None, microaveraging=False, casesensitive=True):
    import math
    import re
    from statistics import geometric_mean
    from difflib import SequenceMatcher, Match

    if not casesensitive:
        predictions = [pred.lower() for pred in predictions]
        references = [[ref.lower() for ref in ref_list] for ref_list in references]

    #tokenization
    predictions = [re.sub(r'[^\w\s\']', ' ', pred) for pred in predictions]
    predictions = [re.sub(r'\s+', ' ', pred) for pred in predictions]

    references = [[re.sub(r'[^\w\s\']', ' ', ref) for ref in ref_list] for ref_list in references]
    references = [[re.sub(r'\s+', ' ', ref) for ref in ref_list] for ref_list in references]
    #TODO implement proper tokenization

    #print(predictions)

    BLEU_uni = 0.0
    BLEU_bi = 0.0
    BLEU_tri = 0.0
    BLEU_four = 0.0
    reference_length = 1000000
    bp = 0.0

    BLEU_uni_scores = []
    BLEU_bi_scores = []
    BLEU_tri_scores = []
    BLEU_four_scores = []
    bps = []

    if not microaveraging:
    #unigrams
        for i in range(len(predictions)):
            predicted_unigrams = predictions[i].split() 
            max_matched_unigrams = 0
            for reference in references[i]:
                matched_unigrams_current = 0
                matched_unigrams = []
                for unigram in predicted_unigrams:
                    if unigram in reference:
                        if reference.split().count(unigram) > matched_unigrams.count(unigram):
                            matched_unigrams_current += 1 
                            matched_unigrams.append(unigram)
                if matched_unigrams_current > max_matched_unigrams:
                    max_matched_unigrams = matched_unigrams_current

            BLEU_uni = max_matched_unigrams/len(predicted_unigrams)
            BLEU_uni_scores.append(BLEU_uni)


        #bigrams
        for i in range(len(predictions)):
            try:
                predicted_bigrams = [tuple(predictions[i].split()[j:j+2]) for j in range(len(predictions[i].split()) - 1)]
                max_matched_bigrams = 0
                for reference in references[i]:
                    matched_bigrams_current = 0
                    matched_bigrams = []
                    reference_bigrams = [tuple(reference.split()[j:j+2]) for j in range(len(reference.split()) - 1)]
                    for bigram in predicted_bigrams:
                        if bigram in reference_bigrams:
                            if reference_bigrams.count(bigram) > matched_bigrams.count(bigram):
                                matched_bigrams_current += 1
                                matched_bigrams.append(bigram)
                    if matched_bigrams_current > max_matched_bigrams:
                        max_matched_bigrams = matched_bigrams_current
                BLEU_bi = max_matched_bigrams/len(predicted_bigrams)
            except ZeroDivisionError:
                pass
            BLEU_bi_scores.append(BLEU_bi)

        #trigrams
        for i in range(len(predictions)):
            try:
                predicted_trigrams = [tuple(predictions[i].split()[j:j+3]) for j in range(len(predictions[i].split()) - 2)]
                max_matched_trigrams = 0
                for reference in references[i]:
                    matched_trigrams_current = 0
                    matched_trigrams = []
                    reference_trigrams = [tuple(reference.split()[j:j+3]) for j in range(len(reference.split()) - 2)]
                    for trigram in predicted_trigrams:
                        if trigram in reference_trigrams:
                            if reference_trigrams.count(trigram) > matched_trigrams.count(trigram):
                                matched_trigrams_current += 1
                                matched_trigrams.append(trigram)
                    if matched_trigrams_current > max_matched_trigrams:
                        max_matched_trigrams = matched_trigrams_current
                BLEU_tri = max_matched_trigrams/len(predicted_trigrams)
            except ZeroDivisionError:
                pass
            BLEU_tri_scores.append(BLEU_tri)

        #fourgrams
        for i in range(len(predictions)):
            try:
                predicted_fourgrams = [tuple(predictions[i].split()[j:j+4]) for j in range(len(predictions[i].split()) - 3)]
                max_matched_fourgrams = 0
                for reference in references[i]:
                    matched_fourgrams_current = 0
                    matched_fourgrams = []
                    reference_fourgrams = [tuple(reference.split()[j:j+4]) for j in range(len(reference.split()) - 3)]
                    for fourgram in predicted_fourgrams:
                        if fourgram in reference_fourgrams:
                            if reference_fourgrams.count(fourgram) > matched_fourgrams.count(fourgram):
                                matched_fourgrams_current += 1
                                matched_fourgrams.append(fourgram)
                    if matched_fourgrams_current > max_matched_fourgrams:
                        max_matched_fourgrams = matched_fourgrams_current
                BLEU_four = max_matched_fourgrams/len(predicted_fourgrams)
            except ZeroDivisionError:
                pass
            BLEU_four_scores.append(BLEU_four)



        #brevity penalty
        for i in range(len(predictions)):
            for reference in references[i]:
                if (len(reference.split()) - len(predictions[i])) < (reference_length - len(predictions[i])):
                    reference_length = len(reference.split())
            if reference_length > len(predictions[i].split()):
                bp = math.exp(1 - reference_length/len(predictions[i].split()))
            else:
                bp = 1.0
            bps.append(bp)
        
        BLEU_uni = sum(BLEU_uni_scores) / len(predictions)
        BLEU_bi = sum(BLEU_bi_scores) / len(predictions)
        BLEU_tri = sum(BLEU_tri_scores) / len(predictions)
        BLEU_four = sum(BLEU_four_scores) / len(predictions)
        bp = sum(bps) / len(predictions)

    else: #microaveraging
        #unigrams
        total_elements = 0
        total_matches = 0
        total_predicted = 0
        for i in range(len(predictions)):
            predicted_unigrams = predictions[i].split() 
            total_predicted += len(predicted_unigrams)
            max_matched_unigrams = 0
            reference_length_current = 0
            for reference in references[i]:
                matched_unigrams_current = 0
                matched_unigrams = []
                for unigram in predicted_unigrams:
                    if unigram in reference:
                        if reference.split().count(unigram) > matched_unigrams.count(unigram):
                            matched_unigrams_current += 1
                            matched_unigrams.append(unigram)
                            reference_length_current = len(reference.split())
                if matched_unigrams_current > max_matched_unigrams:
                    max_matched_unigrams = matched_unigrams_current
            total_matches += max_matched_unigrams
            total_elements += reference_length_current

        BLEU_uni = total_matches / total_predicted if total_predicted > 0 else 0


        #bigrams
        total_elements = 0
        total_matches = 0
        total_predicted = 0

        for i in range(len(predictions)):
            try:
                predicted_bigrams = [tuple(predictions[i].split()[j:j+2]) for j in range(len(predictions[i].split()) - 1)]
                total_predicted += len(predicted_bigrams)
                max_matched_bigrams = 0
                for reference in references[i]:
                    reference_bigrams = [tuple(reference.split()[j:j+2]) for j in range(len(reference.split()) - 1)]
                    matched_bigrams_current = 0
                    matched_bigrams = []
                    for bigram in predicted_bigrams:
                        if bigram in reference_bigrams:
                            if reference_bigrams.count(bigram) > matched_bigrams.count(bigram):
                                matched_bigrams_current += 1
                                matched_bigrams.append(bigram)
                    if matched_bigrams_current > max_matched_bigrams:
                        max_matched_bigrams = matched_bigrams_current
                        reference_length_current = len(reference_bigrams)
                total_matches += max_matched_bigrams
                total_elements += reference_length_current
            except ZeroDivisionError:
                pass

        BLEU_bi = total_matches / total_predicted if total_predicted > 0 else 0

        #trigrams
        total_elements = 0
        total_matches = 0
        total_predicted = 0

        for i in range(len(predictions)):
            try:
                predicted_trigrams = [tuple(predictions[i].split()[j:j+3]) for j in range(len(predictions[i].split()) - 2)]
                total_predicted += len(predicted_trigrams)
                max_matched_trigrams = 0
                for reference in references[i]:
                    reference_trigrams = [tuple(reference.split()[j:j+3]) for j in range(len(reference.split()) - 2)]
                    matched_trigrams_current = 0
                    matched_trigrams = []
                    for trigram in predicted_trigrams:
                        if trigram in reference_trigrams:
                            if reference_trigrams.count(trigram) > matched_trigrams.count(trigram):
                                matched_trigrams_current += 1
                                matched_trigrams.append(trigram)
                    if matched_trigrams_current > max_matched_trigrams:
                        max_matched_trigrams = matched_trigrams_current
                        reference_length_current = len(reference_trigrams)
                total_matches += max_matched_trigrams
                total_elements += reference_length_current
            except ZeroDivisionError:
                pass

        BLEU_tri = total_matches / total_predicted if total_predicted > 0 else 0

        #fourgrams
        total_elements = 0
        total_matches = 0
        total_predicted = 0

        for i in range(len(predictions)):
            try:
                predicted_fourgrams = [tuple(predictions[i].split()[j:j+4]) for j in range(len(predictions[i].split()) - 3)]
                total_predicted += len(predicted_fourgrams)
                max_matched_fourgrams = 0
                for reference in references[i]:
                    reference_fourgrams = [tuple(reference.split()[j:j+4]) for j in range(len(reference.split()) - 3)]
                    matched_fourgrams_current = 0
                    matched_fourgrams = []
                    for fourgram in predicted_fourgrams:
                        if fourgram in reference_fourgrams:
                            if reference_fourgrams.count(fourgram) > matched_fourgrams.count(fourgram):
                                matched_fourgrams_current += 1
                                matched_fourgrams.append(fourgram)
                    if matched_fourgrams_current > max_matched_fourgrams:
                        max_matched_fourgrams = matched_fourgrams_current
                        reference_length_current = len(reference_fourgrams)
                total_matches += max_matched_fourgrams
                total_elements += reference_length_current
            except ZeroDivisionError:
                pass

        BLEU_four = total_matches / total_predicted if total_predicted > 0 else 0

        #brevity penalty
        for i in range(len(predictions)):
            for reference in references[i]:
                if (len(reference.split()) - len(predictions[i])) < (reference_length - len(predictions[i])):
                    reference_length = len(reference.split())
            if reference_length > len(predictions[i].split()):
                bp = math.exp(1 - reference_length/len(predictions[i].split()))
            else:
                bp = 1.0
            bps.append(bp)

    #score
    if BLEU_uni != 0 and BLEU_bi != 0 and BLEU_tri != 0 and BLEU_four != 0:
        BLEU_score = geometric_mean([BLEU_uni, BLEU_bi, BLEU_tri, BLEU_four])
    else:
        BLEU_score = 0.0

    #return f"'bleu': {BLEU_score}, 'precisions': [{BLEU_uni}, {BLEU_bi}, {BLEU_tri}, {BLEU_four}], 'brevity_penalty': {bp}, 'length_ratio': {len(predictions[0].split())/reference_length}, 'translation_length': {len(predictions[0].split())}, 'reference_length': {reference_length}"

    return {
    'bleu': BLEU_score,
    'precisions': [BLEU_uni, BLEU_bi, BLEU_tri, BLEU_four],
    'brevity_penalty': bp,
    'length_ratio': len(predictions[0].split())/reference_length,
    'translation_length': len(predictions[0].split()),
    'reference_length': reference_length
    }


#print(BLEU(predictions=prediction, references=references))
print(BLEU(predictions=prediction, references=references, microaveraging=True))

###unit testing
import unittest

class TestBLEU(unittest.TestCase):
    def setUp(self):
        self.predictions = ["The quick brown fox", "jumps over the lazy dog"]
        self.references = [["The quick brown fox", "jumps over the lazy dog"], ["The quick brown fox", "jumps over the lazy dog"]]

    def test_BLEU(self):
        result = BLEU(self.predictions, self.references)
        self.assertEqual(result['bleu'], 1.0)

if __name__ == '__main__':
    unittest.main()

TestBLEU.setUp
TestBLEU.test_BLEU