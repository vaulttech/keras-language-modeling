from nltk.corpus import wordnet as wn

import spacy
import numpy as np
import pickle
import random
import sys

glove_vec_size = 300

def get_random_word(nlp):
    ret = None
    ret_vec = np.zeros(glove_vec_size)
    bad = np.zeros(glove_vec_size)
    while (all(ret_vec == bad)):
        ret = random.choice(nlp.vocab)
        ret_vec = ret.vector

    return ret

#def get_current_lemma(s, l):
#    lemmas_in_synset = s.lemmas()
#    for i in lemmas_in_synset:
#        print(i.name(), ' -- ', l)
#        if i.name() == l:
#            return i
#    return None

def get_triples(nlp, force_antonyms, remove_duplicates):
    ret = []
    duplicates = set()
    for s in wn.all_synsets():
        lemmas = s.lemmas()
        if len(lemmas) == 1:
            # This synset is composed by only one lemma
            continue

        for i in range(len(lemmas)):
            if force_antonyms:
                # Notice that in this case we iterate through all words in the
		# synset. This will allow us to get both
                # ('good', 'goodness', 'evil')
                # and
                # ('goodness', 'good', 'evilness')
                for j in range(1, len(lemmas)):
                    if i == j:
                        continue

                    for a in lemmas[i].antonyms():
                        triple = (lemmas[i].name(), lemmas[j].name(), a.name())
                        ret.append(triple)
            else:
                # Here, we want to avoid having both
                # ('good', 'goodness', <antonym>)
                # and
                # ('goodness', 'good', <antonym>)
                # , because in this case we are taking any random word in the
                # vocabulary as the antonym.
                for j in range(i+1, len(lemmas)):
                    antonym = get_random_word(nlp)
                    triple = (lemmas[i].name(), lemmas[j].name(), antonym.text)
                    if (remove_duplicates):
                        pair = (triple[0], triple[1])
                        if pair not in duplicates:
                            ret.append(triple)
                            duplicates.add(pair)
                    else:
                        ret.append(triple)

    return ret

#def get_triples(nlp, force_antonyms):
#    ret = []
#    already_visited = set()
#    for l in wn.all_lemma_names():
#        synsets = wn.synsets(l)
#        for s in synsets:
#            this_lemma = get_current_lemma(s, l)
#            if this_lemma is None:
#                print("Error: synset {} doesn't contain lemma {}".format(
#                    s, l))
#                sys.exit()
#
#            if len(s.lemmas()) == 1:
#                # This synset is composed by only one lemma
#                continue
#
#            for s_l in s.lemmas():
#                if (s_l.name() == l):
#                    # We don't want triples where the target word and the
#                    # synonym are the same
#                    continue
#
#                if force_antonyms:
#                    for a in s_l.antonyms():
#                        triple = (l, s_l.name(), a.name())
#                        ret.append(triple)
#                else:
#                    if (s_l in already_visited):
#                        # If we have ('good', 'goodness', <antonym>), then we
#                        # don't want ('goodness', 'good', <antonym>). Notice
#                        # that we only do this when we are taking any word as
#                        # antonyms. In the other case, the antonyms for the two
#                        # tuples will be different, and therefore relevant
#                        # ('evil' for the first, and 'evilness' for the second).
#                        continue
#
#                    # Get any random word. This will introduce some noise in the
#                    # dataset, but hopefully allow us to have much more data.
#                    antonym = get_random_word(nlp)
#                    triple = (l, s_l.name(), antonym.text)
#                    ret.append(triple)
#            already_visited.add(this_lemma)
#
#    return ret

def get_all_words(triples):
    all_words = []
    for i in triples:
        # Yes... we assume the triples are always correct
        all_words.append(i[0])
        all_words.append(i[1])
        all_words.append(i[2])

    deduplicated_words = set(all_words)
    return list(deduplicated_words)

def get_singleton_synsets():
    return [s for s in wn.all_synsets() if len(s.lemmas()) == 1 ]
    #ret = []
    #for s in wn.all_synsets():
    #    if len(s.lemmas()) == 1:
    #        ret.append(s)
    #return ret

def get_antonyms(remove_duplicates=False):
    antonyms = []
    lemmas_with_antonyms = []
    lemma_antonym_pairs = []
    for l in wn.all_lemma_names():
        synsets = wn.synsets(l)
        for s in synsets:
            for s_l in s.lemmas():
                if s_l.name() != l:
                    # Here I only care about the antonyms of the current lemma
                    continue

                found_antonym = False
                for a in s_l.antonyms():
                    antonyms.append(a)
                    found_antonym = True

                    if remove_duplicates:
                        lemma_antonym_pairs.append(frozenset((s_l, a)))
                    else:
                        lemma_antonym_pairs.append((s_l, a))


                if found_antonym:
                    lemmas_with_antonyms.append(s_l)

    if remove_duplicates:
        lemma_antonym_pairs = list(set(lemma_antonym_pairs))

    return antonyms, lemmas_with_antonyms, lemma_antonym_pairs


#def initialize_word_embeddings(words):
#    global nlp
#    if nlp is None:
#        nlp = spacy.load('en')
#
#    embeddings = np.zeros([len(words), glove_vec_size])
#    for idx, w in enumerate(words):
#        doc = nlp(w)
#        word_vec = doc.vector
#        embeddings[idx] = word_vec
#
#    return embeddings

def generate_dataset(force_antonyms=False, remove_duplicates=False):
    nlp = spacy.load('en')

    triples = get_triples(nlp, force_antonyms, remove_duplicates)
    train, test = triples[:-500], triples[-500:]

    pickle.dump(train, open('word_synonym_antonym.train', 'wb'))
    pickle.dump(test,  open('word_synonym_antonym.test',  'wb'))

    #all_words = get_all_words(train)
    #embeddings = initialize_word_embeddings(all_words)
    #np.save('word2vec_wordnet.embeddings', embeddings)

    # This will produce {1: 'word1', 2: 'word2', ...}
    #all_words = {i+1: j for (i, j) in enumerate(all_words)}
    #pickle.dump(all_words, open('word2vec_wordnet.vocabulary', 'wb'))

if __name__ == '__main__':
    generate_dataset(remove_duplicates=True)

