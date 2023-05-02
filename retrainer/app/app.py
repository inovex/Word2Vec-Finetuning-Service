import os
import sys
from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, IntegerField, SelectField
from werkzeug.utils import secure_filename
from wtforms.validators import input_required, NumberRange
from werkzeug.datastructures import FileStorage
from retrainer.retraining.pipeline import retrain

sys.path.append(os.getcwd())


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[input_required()])
    model_name = StringField("Model Name", validators=[input_required()])
    epochs = IntegerField(
        "Number of epochs",
        render_kw={"placeholder": "Recommended: 1000"},
        validators=[
            input_required(),
            NumberRange(
                min=1,
                max=4000,
                message="Number of epochs " "must be between 1 " "and 4000",
            ),
        ],
    )
    base_model_choices = [
        ("DE_Wiki", "DE Wiki"),
        ("EN_Wiki", "EN Wiki"),
        ("DE_AT", "DE AT"),
        ("DE_CH", "DE CH"),
        ("DE_Positive", "DE Positive"),
        ("DE_Negative", "DE Negative"),
        ("DE_News", "DE News"),
        ("EN_AU", "EN AU"),
        ("EN_CA", "EN CA"),
        ("EN_GB", "EN GB"),
        ("EN_ZA", "EN_ZA"),
    ]
    base_model = SelectField("Field name", choices=base_model_choices)
    model_uri = StringField("Model URI")
    submit = SubmitField("Start Retraining", render_kw={"onclick": "load()"})


app = Flask(__name__)
app.config["SECRET_KEY"] = "password"
app.config["UPLOAD_FOLDER"] = "static/files"


@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data  # grabbing file
        data_path = "app/static/files/custom_data"
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
        file.save(
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                f"static/files/custom_data",
                secure_filename(file.filename),
            )
        )  # Save file

        model_uri = request.form["model_uri"] if "model_uri" in request.form else ""
        "EN_US",
        message, err_code = retrain(
            model_name=request.form["model_name"],
            epochs=int(request.form["epochs"]),
            file_name=file.filename,
            model_uri=model_uri.replace(" ", ""),
            base_model=request.form["base_model"],
        )
        if err_code != 200:
            return render_template("400.html", error=message), 400
        print("Done!")
        return render_template("finished.html")
    return render_template("index.html", form=form)


@app.route("/api", methods=["POST"])
def api():
    model_name = (
        request.form["model_name"] if "model_name" in request.form else "new_model"
    )
    model_uri = request.form["model_uri"] if "model_uri" in request.form else ""
    epoch = int(request.form["epoch"]) if "epoch" in request.form else 1000
    BASE_MODELS = [
        "DE_Wiki",
        "EN_Wiki",
        "DE_AT",
        "DE_CH",
        "DE_Positive",
        "DE_Negative",
        "DE_News",
        "EN_AU",
        "EN_CA",
        "EN_GB",
        "EN_ZA",
    ]
    base_model = (
        request.form["base_model"]
        if (
            ("base_model" in request.form)
            and (request.form["base_model"] in BASE_MODELS)
        )
        else "DE_Wiki"
    )
    if epoch > 4000 or epoch < 1:
        return "Epochs must be between 1 and 4000", 400
    file_name = f"data_{model_name}.txt"
    file = request.files["file"]
    if not file:
        return "No file was provided", 400
    FileStorage(file).save(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            f"static/files/custom_data",
            secure_filename(file_name),
        )
    )

    # Retraining
    message, err_code = retrain(
        model_name=model_name,
        epochs=epoch,
        file_name=file_name,
        base_model=base_model,
        model_uri=model_uri,
    )
    if err_code != 200:
        return message, err_code

    return (
        'Model retrained and logged to mlflow! View it on "http://0.0.0.0:5000/"',
        200,
    )


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=False, port=5004)
