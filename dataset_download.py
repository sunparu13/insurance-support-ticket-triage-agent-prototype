import kagglehub

kagglehub.login()  # Login


# Download dataset from Kaggle
path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets",
                                  output_dir="./data",
                                  force_download=True,)

print("Path to dataset files:", path)