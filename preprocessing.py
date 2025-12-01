import csv
import logging
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

logging.basicConfig(
    level="INFO",
    filename=r"C:\python_logs\data_preprocessing_log.txt",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Preprocessing():

    def transforming_data(self, text):
        ps = PorterStemmer()
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [ps.stem(word) for word in tokens]
        return " ".join(tokens)

    def preprocessing_data(self, raw_data_path, text_column='v2', target_column='v1'):
        try:
            df = pd.read_csv(raw_data_path, encoding="latin1")
            logging.info("CSV file read successfully")

            df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
            df = df.dropna()
            df = df.drop_duplicates(keep='first')

            df[target_column] = df[target_column].replace({'ham': 0, 'spam': 1})

            df[text_column] = df[text_column].apply(self.transforming_data)
            logging.info("Text transformation completed")

            # Split into train and test
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df[target_column]
            )

            # Save processed datasets
            csv_save_dir = os.path.join("data", "processed")
            os.makedirs(csv_save_dir, exist_ok=True)

            train_path = os.path.join(csv_save_dir, "train.csv")
            test_path = os.path.join(csv_save_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info("Train and test files saved successfully")

        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}", exc_info=True)
            raise e

        return train_df, test_df


if __name__ == '__main__':
    obj = Preprocessing()
    train_df, test_df = obj.preprocessing_data(
        r"C:\Users\Kashish Gupta\python\spam_mlops\data\raw\raw_data.csv"
    )
    print("TRAIN SHAPE:", train_df.shape)
    print("TEST SHAPE:", test_df.shape)
