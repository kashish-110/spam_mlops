import csv
import logging
import pandas as pd
import os
#from sklearn.preprocessing import LabelEncoder
logging.basicConfig(
    level = "INFO",
    filename=r"C:\python_logs\data_preprocessing_log.txt",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
class Preprocessing():
    def preprocessing_data(self,raw_data_path):
        try:
            df = pd.read_csv(raw_data_path, encoding="latin1")
            
            logging.info("csv file read successfully")
            print(df.columns)
            # del df['Unnamed: 2']
            # del df['Unnamed: 3']
            # del df['Unnamed: 4']
            df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
            df.dropna()
            df.info()

            #for spam-->1, for not spam(ham)-->0
            df['v1']=df['v1'].replace({'ham':0, 'spam': 1})
            print(df.head())
            print(df.columns)
            logging.info("csv file preprocessed")
            csv_save_dir = os.path.join(".\data","processed")
            os.makedirs(csv_save_dir,exist_ok=True)
            csv_save_file = os.path.join(csv_save_dir,"processed_csv.csv")
            df.to_csv(csv_save_file,index=False)
            logging.info("Processed data saved successfully")
        except Exception as e:
            logging.error(f"Error in opening file {e}")
        except Exception as e:
            logging.error(f"Error saving the file {e}")
            raise e

        return df

obj = Preprocessing()
obj.preprocessing_data(r"C:\Users\Kashish Gupta\python\spam_mlops\data\raw\raw_data.csv")
