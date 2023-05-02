import os
import sys
import shutil
import pickle
import mlflow
from retrainer.retraining.retraining import Word2VecRetrainer
from retrainer.retraining.preprocessing import Word2VecPreprocessor
from mlflow.sklearn import log_model
import requests
from retrainer.retraining.evaluation import evaluate_word_similarity
import mimetypes

sys.path.append(os.getcwd())


def retrain(
    model_name: str,
    file_name: str,
    base_model: str,
    model_uri: str = "",
    epochs: int = 1000,
):
    # Preprocessing
    file_mime_type, _ = mimetypes.guess_type(file_name)
    if not file_mime_type or not file_mime_type.startswith("text/"):
        return "Provided file is not a text file. File extension must be .txt", 400

    preprocessor = Word2VecPreprocessor(file_name)
    try:
        preprocessed_file_name = preprocessor.preprocess()
    except UnicodeDecodeError:
        return (
            "Error preprocessing your file. Make sure it is in the correct format.",
            400,
        )

    preprocessed_file_name = (
        f"retrainer/app/static/files/preprocessed_data/{preprocessed_file_name}"
    )

    # Downloading Base Model
    print(f"Downloading base model from Huggingface ...")
    link_model = f"https://huggingface.co/JayInovex/Word2Vec-Finetuning-Service-Base-Models/resolve/main/{base_model}/{base_model}.model"
    link_negative = f"https://huggingface.co/JayInovex/Word2Vec-Finetuning-Service-Base-Models/resolve/main/{base_model}/{base_model}.model.syn1neg.npy"
    link_vector = f"https://huggingface.co/JayInovex/Word2Vec-Finetuning-Service-Base-Models/resolve/main/{base_model}/{base_model}.model.wv.vectors.npy"

    if not model_uri:
        r = requests.get(link_model)
        p1 = "retrainer/models/pretrained.model"
        open(p1, "wb").write(r.content)
        print("pretrained.model downloaded!")
        r = requests.get(link_negative)
        p2 = "retrainer/models/pretrained.model.syn1neg.npy"
        open(p2, "wb").write(r.content)
        print("pretrained.model.syn1neg.npy downloaded!")
        r = requests.get(link_vector)
        p3 = "retrainer/models/pretrained.model.wv.vectors.npy"
        open(p3, "wb").write(r.content)
        print("pretrained.model.wv.vectors.npy downloaded!")

        # Retraining
        w2vr = Word2VecRetrainer("retrainer/models/pretrained.model")
        message, err = w2vr.retrain(data=preprocessed_file_name, epochs=epochs)

        os.remove(p1)
        os.remove(p2)
        os.remove(p3)

        if err == 400:
            return message, err
    else:
        print("Loading model from mlflow")
        mlflow_model_path = "retrainer/models/mlflow_pretrained"
        os.makedirs(exist_ok=True, name=mlflow_model_path)
        try:
            model = mlflow.sklearn.load_model(model_uri=model_uri.replace(" ", ""))
        except OSError:
            return (
                "Model URI does not have the right format. Please consult the documentation. (README.md)",
                400,
            )
        shutil.rmtree(mlflow_model_path)
        mlflow.sklearn.save_model(sk_model=model, path=mlflow_model_path)
        print(f"Downloaded model from mlflow")
        w2vr = Word2VecRetrainer()
        with open(mlflow_model_path + "/model.pkl", "rb") as f:
            model = pickle.load(f)
        w2vr.pretrained_model = model
        w2vr.retrain(data=preprocessed_file_name, epochs=epochs)
        shutil.rmtree(mlflow_model_path)

    # Save Model
    client = mlflow.tracking.MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.start_run()
    mlflow.set_tag("name", model_name)
    if model_uri in ["", " ", None]:
        mlflow.set_tag("base_model_used", base_model)
        lang = base_model[:2].lower()
        if lang not in ["en", "de"]:
            raise Exception(
                "Base Model does not have a name starting with 'EN' or 'DE'"
            )
        mlflow.set_tag("language", lang)
    else:
        mlflow.set_tag(
            "base_model_used",
            client.get_run(run_id=model_uri.split("/")[-3]).data.tags["name"],
        )
        lang = client.get_run(run_id=model_uri.split("/")[-3]).data.tags["language"]
        mlflow.set_tag("language", lang)

    print(lang)
    if lang not in ["en", "de"]:
        return "Language Tag must be either 'en' or 'de' ", 400

    evaluation_results = evaluate_word_similarity(w2vr.pretrained_model.wv, lang)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("sim999_pearson", evaluation_results["sim999"]["pearson"])
    mlflow.log_metric("sim999_spearman", evaluation_results["sim999"]["spearman"])
    mlflow.log_metric("sim999_hitrate", evaluation_results["sim999"]["hitrate"])

    mlflow.log_metric("wordsim_pearson", evaluation_results["wordsim"]["pearson"])
    mlflow.log_metric("wordsim_spearman", evaluation_results["wordsim"]["spearman"])
    mlflow.log_metric("wordsim_hitrate", evaluation_results["wordsim"]["hitrate"])

    log_model(
        w2vr.pretrained_model, artifact_path="model", registered_model_name=model_name
    )
    mlflow.end_run()

    # Delete Data
    os.remove(preprocessed_file_name)
    return "Success", 200
