import argparse
from eagle_train import *
from create_combined_source_data import *
import os
import contextlib


tgt_map_k_5 = {'ctrl': ["fair_wmt19", "gpt2_xl", "gpt3", "grover_mega", "xlm"],
                'fair_wmt19': ["ctrl", "gpt2_xl", "gpt3", "grover_mega", "xlm"],
                'gpt2_xl': ["ctrl", "fair_wmt19", "gpt3", "grover_mega", "xlm"],
                'gpt3': ["ctrl", "fair_wmt19", "gpt2_xl", "grover_mega", "xlm"],
                'grover_mega': ["ctrl", "fair_wmt19", "gpt3", "gpt2_xl", "xlm"],
                'xlm': ["ctrl", "fair_wmt19", "gpt3", "grover_mega", "gpt2_xl"]}

#target_domains = ['ctrl', 'fair_wmt19', 'gpt2_xl', 'gpt3', 'grover_mega', 'xlm']
target_domains = ['gpt2_xl']

loss_type = 'simclr'  

for tgt in target_domains:
    
    data_dir = "./domain_gen_syn_rep_data/TuringBench/"
    processed_file_out_dir = "./processed_data/"
    
    ## create the train & validation files with all k sources
    create_combined_jsonl_files(source_list=tgt_map_k_5[tgt], data_dir=data_dir, out_dir=processed_file_out_dir)

    ## now set params for training model

    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-sequence-length', type=int, default=256)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--loss_type', type=str, default=loss_type)
    parser.add_argument('--seed', type=int, default=None)
   
    parser.add_argument('--src_data-dir', type=str, default='')
    parser.add_argument('--src_real-dataset', type=str, default='combined_real')
    parser.add_argument('--src_fake-dataset', type=str, default='combined_fake')

    parser.add_argument('--num_sources', type=int, default=5) # number of sources to use, i.e. k value

    
    parser.add_argument('--model_save_path', default=os.getcwd()+'/models/')
    
    if loss_type=='simclr':
        parser.add_argument('--model_save_name', default=f'combined_k_5_for_{tgt}.pt')
    ## structure:  combined_k_{number of source domains}_for_{target}.pt

    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--load-decay', type=float, default=0)
    parser.add_argument('--lambda_w', type=float, default=0.5)

    src_data_dir = "./processed_data/"

    model_save_path = "./domain_gen_models/"
    
    args = parser.parse_args(args=['--max-epochs=3', '--model_save_path='+model_save_path,\
         '--src_data-dir='+src_data_dir])
        
        
    filename = f'output_log_k_5_{tgt}.log'
    filepath = os.getcwd() + '/output_logs/'
    with open(filepath+filename, 'w') as f:
        with contextlib.redirect_stderr(f):
            main(args)
