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

