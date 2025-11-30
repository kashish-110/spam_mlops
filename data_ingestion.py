import pandas as pd
import numpy as np
import csv
import logging
import os
logging.basicConfig(
    level = "INFO",
    format= "%(asctime)s - %(levelname)s - %(message)s",
    filename = r"C:\python_logs\data_ingestion_log.txt"
)
class DataIngestion():
    def read_csv(self,data_path):
        try:
            
            # directory to save raw data
            data_store = os.path.join(".\data", "raw")
            os.makedirs(data_store, exist_ok=True)

            df = pd.read_csv(data_path, encoding="latin1")
            logging.info(f"CSV file loaded successfully: {data_path}")

            print(df.head())
            logging.info("DataFrame created successfully")
            raw_file_path = os.path.join(data_store, "raw_data.csv")
            df.to_csv(raw_file_path, index=False)

            return df

        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}", exc_info=True)
            raise e

if __name__ == '__main__':
    obj = DataIngestion()
    data_path = r"C:\Users\Kashish Gupta\Downloads\archive (5)\spam.csv"
    obj.read_csv(data_path)
