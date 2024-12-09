from datasets import load_dataset

print("Starting data download.")
dataset = load_dataset("ealvaradob/phishing-dataset", "urls", trust_remote_code=True)

dataset['train'].to_csv('./data/urls.csv')

print("Data dowloaded and saved.")