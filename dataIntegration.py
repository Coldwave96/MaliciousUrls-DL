import os
import shutil
import pandas as pd
import extractFromES

dataFromES = "RawData/malicious_es.txt"

maliciousData = "IntegratedData/malicious.txt"
benignData = "IntegratedData/benign.txt"


def es_integration():
    if os.path.exists(dataFromES):
        shutil.copy(dataFromES, maliciousData)
    else:
        print("Extracting data from ES...")
        extractFromES.data2file()
        shutil.copy(dataFromES, maliciousData)
    print("Malicious data from ES has been loaded.")


def csv_integration(path):
    csv_df = pd.read_csv(path)

    if os.path.exists(maliciousData):
        os.remove(maliciousData)

    if os.path.exists(benignData):
        os.remove(benignData)

    with open(maliciousData, 'a') as f:
        for index, row in csv_df.iterrows():
            if row['label'] == "bad" or row['label'] == "malicious":
                f.write(row['url'] + '\n')

    with open(benignData, 'a') as f:
        for index, row in csv_df.iterrows():
            if row['label'] == "good" or row['label'] == "benign":
                f.write(row['url'] + '\n')


if __name__ == '__main__':
    csv1 = "RawData/data.csv"
    csv2 = "RawData/urldata.csv"

    print("[*]Loading ES data...")
    es_integration()

    print("[*]Loading kaggle csv data...")
    csv_integration(csv1)
    csv_integration(csv2)
    print("Data from kaggle csv files have been loaded.")