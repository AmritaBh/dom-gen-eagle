import pandas as pd

tgts = ['claude', 'gpt4']

for tgt in tgts:
    dir = f"/Data_for_Testing/Data/{tgt}/"
    fake_filepath = dir + 'fake.test.jsonl'
    real_filepath = dir + 'real.test.jsonl'

    real_df = pd.read_json(real_filepath, lines=True, orient='records')
    fake_df = pd.read_json(fake_filepath, lines=True, orient='records')

    if tgt=='claude':
        dom = 7
    if tgt=='gpt4':
        dom=8

    real_df['domain_label'] = dom
    fake_df['domain_label'] = dom

    save_dir = "/Data_for_Testing/syn_rep/Domain-Labels/"
    real_df.to_json(save_dir + f'domain_{tgt}_real.test.jsonl', lines=True, orient='records')
    fake_df.to_json(save_dir + f'domain_{tgt}_fake.test.jsonl', lines=True, orient='records')
