import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv, set_key
from src.predict import *


def main():

    # setup
    load_dotenv()
    st_filepath = os.getenv('STATUSES_FILEPATH')
    cl_filepath = os.getenv('CLASSIFIED_FILEPATH')
    datefile = "2020-11-03.csv"

    print("Started script at " + datetime.now().strftime("%H:%M:%S"))
    # access tweets (postgres/ csv), convert to dataframe
    statuses = pd.read_csv(st_filepath + "tp-statuses-" + datefile)
    print("Loaded statuses at " + datetime.now().strftime("%H:%M:%S"))

    # if neccessary, to drop duplicates
    statuses = statuses.drop_duplicates()

    # use dataframe to predict
    print("Started predicting at " + datetime.now().strftime("%H:%M:%S"))
    predicted_statuses = predict(statuses)
    print("Finished predicting, starting write to csv at " + datetime.now().strftime("%H:%M:%S"))

    # output new csv
    predicted_statuses.to_csv(
        cl_filepath + "class-statuses-" + datefile, index=False, header=True)

    # predicted_statuses = pd.read_csv(cl_filepath + "class-statuses-" + datefile)
    hate_statuses = predicted_statuses[predicted_statuses['hate_score_consensus']==1]
    hate_statuses.to_csv(
        cl_filepath + "all-hate.csv", index=False, header=False, mode='a')

    print("Finished script at " + datetime.now().strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
