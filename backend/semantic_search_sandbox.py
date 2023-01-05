from rapidfuzz import process
from rapidfuzz import fuzz

import re
import json
from collections import Counter
import sys
import numpy as np
def query(vec,embs,n=3):
    # index = np.argmax(np.dot(embs,vec/np.linalg.norm(vec)))
    # scores = np.dot(embs,vec[0]/np.linalg.norm(vec[0]))
    scores = np.dot(embs,vec/np.linalg.norm(vec))
    # nonlocal scores
    # scores = -np.linalg.norm(embs-vec,axis=1)
    indices = np.argsort(scores)
    # for i in indices[-n:][::-1]:
    #     scores1.append(scores[i])
        # print(scores[i])
    return scores,indices[-n:]

default_text_weight = 0.4
default_image_weight = 0.6
# default_fuzzy_weight = 0.5
default_fuzzy_weight = 0.2

def normalize_embeddings(embs):
    return embs / np.linalg.norm(embs,axis=1, keepdims=True)

import numpy as np
def search(query_str, query_embedding, normalized_sentence_embeddings=None, texts=None, normalized_image_embeddings=None,n=3,fuzzy_weight=default_fuzzy_weight,text_weight=default_text_weight,image_weight=default_image_weight):
    print(query_str)
    scores_text,indices_text = query(query_embedding,normalized_sentence_embeddings,n)
    print(scores_text.shape)
    if normalized_image_embeddings is not None:
        scores_images,indices_images = query(query_embedding,normalized_image_embeddings,n)
    results_text = Counter({i:text_weight*scores_text[i] for i in indices_text})
    if normalized_image_embeddings is not None:
        results_images = Counter({i:image_weight*scores_images[i] for i in indices_images})
    # print(results1)
    if fuzzy_weight > 0:
        import time
        start_time = time.time()
        # results2 = process.extract(query_str, {i:x for i,x in enumerate(texts)}, limit=n)
        # print(query_str)
        # print(type(texts))
        # print(texts[0])
        results2 = process.extract(query_str, texts, scorer=fuzz.WRatio, limit=n)
        # results2 = process.extract("hahatest", ["test","tost"], scorer=fuzz.WRatio, limit=1)
        print("--- %s seconds ---" % (time.time() - start_time))
        results2 = Counter({x[2]:(fuzzy_weight*x[1]/100) for x in results2})
        # print(results2)
        for key,value in list(results_text.most_common()):
            results2[key] = fuzzy_weight*fuzz.WRatio(query_str,texts[key])/100
        if normalized_image_embeddings is not None:
            for key,value in list(results_images.most_common()):
                results2[key] = fuzzy_weight*fuzz.WRatio(query_str,texts[key])/100
        for key,value in list(results2.most_common()):
            results_text[key] = text_weight*scores_text[key]
            if normalized_image_embeddings is not None:
                results_images[key] = image_weight*scores_images[key]

        if normalized_image_embeddings is not None:
            results = results_text + results_images + results2
        else:
            results = results_text + results2
        return [key for key,value in results.most_common(n)]
    else:
        return [key for key,value in results_text.most_common(n)]
