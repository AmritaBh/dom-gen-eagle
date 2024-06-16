import pandas as pd
from typing import *

def process_chatgpt(path_to_csv):
    """
        Method to process "$text and $label for chatgpt dataset.
    """
    df = pd.read_csv(path_to_csv)
    if "title" not in df.columns:
        return
    
    df = df[["title", "text", "label"]]
    df = df.assign(text = lambda x: x.title + ". "+ x.text)
    df["label"] = df.label.apply(lambda x: int(x))
#     df[["text","label"]].to_csv(path_to_csv, index=False)
    return df

def create_combined_jsonl_files(source_list: List[str], data_dir: str, out_dir: str):
    
    num_labels = len(source_list)

    full_source_real_train_df = pd.DataFrame()
    full_source_fake_train_df = pd.DataFrame()
    full_source_real_valid_df = pd.DataFrame()
    full_source_fake_valid_df = pd.DataFrame()


    for i in range(len(source_list)):
        domain = source_list[i]
        domain_label = [float(i)] 

        real_train_path = data_dir + domain + '/real.train.jsonl'
        fake_train_path = data_dir + domain + '/fake.train.jsonl'
        real_valid_path = data_dir + domain + '/real.valid.jsonl'
        fake_valid_path = data_dir + domain + '/fake.valid.jsonl'

        df1 = pd.read_json(real_train_path, lines=True, orient="records")
        df2 = pd.read_json(fake_train_path, lines=True, orient="records")
        df1['domain_label'] = domain_label * len(df1)
        df2['domain_label'] = domain_label * len(df2)
        df1['domain_name'] = [domain] * len(df1)
        df2['domain_name'] = [domain] * len(df2)

        df3 = pd.read_json(real_valid_path, lines=True, orient="records")
        df4 = pd.read_json(fake_valid_path, lines=True, orient="records")
        df3['domain_label'] = domain_label * len(df3)
        df4['domain_label'] = domain_label * len(df4)
        df3['domain_name'] = [domain] * len(df3)
        df4['domain_name'] = [domain] * len(df4)

        full_source_real_train_df = pd.concat([full_source_real_train_df, df1], ignore_index=True)
        full_source_fake_train_df = pd.concat([full_source_fake_train_df, df2], ignore_index=True)
        full_source_real_valid_df = pd.concat([full_source_real_valid_df, df3], ignore_index=True)
        full_source_fake_valid_df = pd.concat([full_source_fake_valid_df, df4], ignore_index=True)
    #     full_source_train_df = pd.concat([full_source_train_df, df1, df2], ignore_index=True)
    #     full_source_valid_df = pd.concat([full_source_valid_df, df3, df4], ignore_index=True)

    full_source_real_train_df = full_source_real_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_source_fake_train_df = full_source_fake_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_source_real_valid_df = full_source_real_valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_source_fake_valid_df = full_source_fake_valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

    
    jsonl_data = full_source_real_train_df.to_json(orient='records', lines=True)
    with open(out_dir + 'combined_real.train.jsonl', "w") as text_file:
        text_file.write(jsonl_data)
        
    jsonl_data = full_source_real_valid_df.to_json(orient='records', lines=True)
    with open(out_dir + 'combined_real.valid.jsonl', "w") as text_file:
        text_file.write(jsonl_data)
        
    jsonl_data = full_source_fake_train_df.to_json(orient='records', lines=True)
    with open(out_dir + 'combined_fake.train.jsonl', "w") as text_file:
        text_file.write(jsonl_data)
        
    jsonl_data = full_source_fake_valid_df.to_json(orient='records', lines=True)
    with open(out_dir + 'combined_fake.valid.jsonl', "w") as text_file:
        text_file.write(jsonl_data)
    return