import sys
import os
import gensim.downloader as api
import gensim.models

# Word similarity
sys.path.append(os.getcwd())


def evaluate_word_similarity(
    word_vectors: gensim.models.KeyedVectors, language: str
) -> dict:
    if language not in ["de", "en"]:
        raise ValueError("Language not supported")

    # Set the current working directory to the directory containing the current file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sim999 = word_vectors.evaluate_word_pairs(
        pairs=f"./evaluation_data/{language}/SimLex-999.csv",
    )

    wordsim = word_vectors.evaluate_word_pairs(
        pairs=f"./evaluation_data/{language}/wordsim_similarity_goldstandard.csv",
    )
    os.chdir("/")

    return {
        "sim999": {
            "pearson": sim999[0][0],
            "spearman": sim999[1][0],
            "hitrate": sim999[2],
        },
        "wordsim": {
            "pearson": wordsim[0][0],
            "spearman": wordsim[1][0],
            "hitrate": wordsim[2],
        },
    }
