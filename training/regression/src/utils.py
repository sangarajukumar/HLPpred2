import yaml
from argparse import Namespace
from Bio import SeqIO
import re
import numpy as np

def set_nested_attr(obj, key, value):
    if isinstance(value, dict):
        if not hasattr(obj, key):
            setattr(obj, key, Namespace())
        
        for subkey in value:
            set_nested_attr(getattr(obj, key), subkey, value[subkey])
    else:
        setattr(obj, key, value)

def get_config(config_file):
    f_cfg = yaml.safe_load(open(config_file, 'r'))
    config = Namespace()
    for key, value in f_cfg.items():
        set_nested_attr(config, key, value)
    return config

def extract_headers_phc50(fasta_file):
    """
    Extract headers and pHC50 values from regression FASTA
    
    Header format: >Positive_1_HC50_50.10_pHC50_4.3002
    """
    headers = []
    phc50_values = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        header_full = record.description
        
        header = header_full.split('_HC50_')[0]
        headers.append(header)
        
        try:
            phc50_str = header_full.split('_pHC50_')[1]
            phc50 = float(phc50_str)
            phc50_values.append(phc50)
        except:
            raise ValueError(f"Could not extract pHC50 from header: {header_full}")
    
    phc50_values = np.array(phc50_values, dtype=np.float32)
    
    return headers, phc50_values
