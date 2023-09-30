import json
import pandas as pd

from loguru import logger


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


if __name__ == "__main__":

    ################################################################################
    # Internal files
    eps = 0.027
    outliers = load_json(f"./Results/internal/{eps}_[0, 1, 0.1]/outliers.json")

    # Convert the dictionary to a DataFrame
    rows = []
    hf_url = "https://huggingface.co/"
    for architecture, models in outliers.items():
        for model in models:
            url = hf_url + model
            rows.append([url, model, architecture])

    df = pd.DataFrame(rows, columns=["URL", "Model Name", "Model Architecture"])

    # Save the DataFrame to an Excel file
    df.to_excel("internal_outliers.xlsx", index=False)
    df.to_csv("internal_outliers.csv", index=False)
    ################################################################################


    ################################################################################
    # External files
    eps = "0.010"
    outliers = load_json(f"./Results/external/{eps}_[0, 1, 1]/outliers.json")

    # Convert the dictionary to a DataFrame
    rows = []
    for architecture, models in outliers.items():
        for model in models:
            model_name = model.split("/")[-1]
            hub_name = "/".join(model.split("/")[:-1])
            rows.append([hub_name, model_name, architecture])

    df = pd.DataFrame(rows, columns=["Model Hub", "Model Name", "Model Architecture"])
    df.to_excel("external_outliers.xlsx", index=False)
    df.to_csv("external_outliers.csv", index=False)
    ################################################################################
