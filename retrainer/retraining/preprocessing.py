import gensim
import nltk
import nltk.data
from nltk.corpus import stopwords
import os
import re
import multiprocessing as mp
import time

nltk.download("punkt")
nltk.download("stopwords")

CUR = os.path.dirname(__file__)
PUNCTUATION_TOKENS = [
    ".",
    "..",
    "...",
    ",",
    ";",
    ":",
    "(",
    ")",
    '"',
    "'",
    "[",
    "]",
    "{",
    "}",
    "?",
    "!",
    "-",
    "–",
    "+",
    "*",
    "--",
    "''",
    "``",
]
PUNCTUATION = "?.!/;:()&+"
THREADS = mp.cpu_count()
BATCH_SIZE = 32
sentence_detector = nltk.data.load("tokenizers/punkt/german.pickle")


class Word2VecPreprocessor:
    """
    Preprocesses the data for the Word2Vec model.
    """

    def __init__(self, filename):
        """Constructs the Word2VecRetraining Object.

        :filename (strin): The name of the file which will be preprocessed.
        """
        self.time = str(time.time())
        self.data = f"retrainer/app/static/files/custom_data/{filename}"
        self.outputFileName = f"{self.time}_{filename}"
        self.target = (
            f"retrainer/app/static/files/preprocessed_data/{self.time}_{filename}"
        )

    def replace_umlauts(self, text):
        """
        Replaces german umlauts and sharp s in given text.

        :param text: text as str
        :return: manipulated text as str
        """
        res = text
        res = (
            res.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("Ä", "Ae")
            .replace("Ö", "Oe")
            .replace("Ü", "Ue")
            .replace("ß", "ss")
        )
        return res

    def process_line(self, line):
        """
        Pre processes the given line.

        :param line: line as str
        :return: preprocessed sentence
        """
        sentences = sentence_detector.tokenize(line)
        for sentence in sentences:
            sentence = self.replace_umlauts(sentence)
            sentence = sentence.lower()
            words = nltk.word_tokenize(sentence)
            words = [x for x in words if x not in PUNCTUATION_TOKENS]
            words = [re.sub("[{}]".format(PUNCTUATION), "", x) for x in words]
            STOPWORDS = [
                self.replace_umlauts(token) for token in stopwords.words("german")
            ]
            words = [x for x in words if x not in STOPWORDS]
            if len(words) > 1:
                return "{}\n".format(" ".join(words))
            return words[0]

    def preprocess(self):
        """
        The actual preprocessing method

        :return: name of the proprocessed file
        """
        if not os.path.exists(os.path.dirname(self.target)):
            os.makedirs(os.path.dirname(self.target))
        with open(self.data, "r") as infile:
            pool = mp.Pool(THREADS)
            values = pool.imap(self.process_line, infile, chunksize=BATCH_SIZE)
            with open(self.target, "w") as outfile:
                for i, s in enumerate(values):
                    if i and i % 25000 == 0:
                        print(f"processed {i} sentences")
                        outfile.flush()
                    if s:
                        outfile.write(s)
        return self.outputFileName

    def bigram(self):
        """
        Create bigrams based on the preprocessed file
        """

        # Iterable Corpus Sentences
        class CorpusSentences:
            def __init__(self, filename):
                self.filename = filename

            def __iter__(self):
                for line in open(self.filename):
                    yield line.split()

        print("train bigram phrase detector")
        bigram = gensim.models.Phrases(CorpusSentences(self.target))
        print("transform corpus to bigram phrases")
        with open("{}.bigram".format(self.target), "w") as outfile:
            for tokens in bigram[CorpusSentences(self.target)]:
                outfile.write("{}\n".format(" ".join(tokens)))
