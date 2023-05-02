import os
from gensim.models import Word2Vec


def count_words(file_path):
    with open(file_path, "r") as f:
        contents = f.read()
        word_count = len(contents.split())
    return word_count


class Word2VecRetrainer:
    """
    Retrains the Word2Vec model.
    """

    def __init__(self, pretrained_path=None):
        """Constructs the Word2VecRetraining Object.

        Args:
            pretrained_path (string): The path where the pretrained model will be loaded from.
        """
        if pretrained_path is not None:
            print(f"Loading the pretrained model from {pretrained_path}...")
            if not os.path.isfile(pretrained_path):
                raise Exception(f"{pretrained_path} is not a file or was not found.")
            self.pretrained_model = Word2Vec.load(pretrained_path)

    def retrain(self, data, epochs=1000):
        """Retrains the Word2Vec model that was pretrained on the wikipedia corpus.

        Args:
            data (DataFrame): The data on which the model will be retrained.
                The columns "beschreibung_clean" and "problemloesung_clean" are used.
            epochs (int, optional): The number of epochs to use for the retraining. Defaults to 1000.

        Returns:
            Word2Vec: The retrained Word2Vec model.
        """

        print("Started updating of the vocabulary...")
        wc = count_words(file_path=data)
        if wc == 0:
            return "The input file cannot be empty.", 400
        self.pretrained_model.build_vocab(corpus_file=data, update=True)
        print("Starting to train...")
        self.pretrained_model.train(
            corpus_file=data,
            total_words=count_words(file_path=data),
            epochs=epochs,
            total_examples=self.pretrained_model.corpus_count,
            report_delay=1,
        )
        print("Finished training")
        return "Finished training", 200
        # self.pretrained_model is now the retrained model.

    def save_model(self, path="wiki_retrained.model"):
        """Saves the retrained model. This is to be used AFTER calling retrain().

        Args:
            path (str, optional): The path where the model should be saved. Defaults to "wiki_retrained.model".
        """
        print("Saving the model...")
        self.pretrained_model.save(path)
