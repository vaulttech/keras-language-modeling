from nltk.corpus import wordnet as wn


def get_triples():
    ret = []
    for l in wn.all_lemma_names():
        synsets = wn.synsets(l)
        for s in synsets:
            if len(s.lemmas()) == 1:
                # This synset is composed by only one lemma
                continue

            for s_l in s.lemmas():
                if (s_l.name() == l):
                    # We don't want triples where the target word and the
                    # synonym are the same
                    continue

                for a in s_l.antonyms():
                    triple = (l, s_l.name(), a.name())
                    ret.append(triple)

    return ret

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

import spacy
import numpy as np
import pickle

glove_vec_size = 300
nlp = None

def initialize_word_embeddings(words):
    global nlp
    if nlp is None:
        nlp = spacy.load('en')

    embeddings = np.zeros([len(words), glove_vec_size])
    for idx, w in enumerate(words):
        doc = nlp(w)
        word_vec = doc.vector
        embeddings[idx] = word_vec

    return embeddings

def generate_word_embeddings():
    triples = get_triples()
    all_words = get_all_words(triples)

    embeddings = initialize_word_embeddings(all_words)
    np.save('word2vec_wordnet.embeddings', embeddings)

    # This will produce {0: 'word1', 1: 'word2', ...}
    all_words = {i+1: j for (i, j) in enumerate(all_words)}
    pickle.dump(all_words, open('word2vec_wordnet.vocabulary', 'wb'))

if __name__ == '__main__':
    generate_word_embeddings()

