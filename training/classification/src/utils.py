import yaml
from argparse import Namespace
from Bio import SeqIO

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

def extract_headers_labels(fasta_file):
    headers= []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        tem_header = record.description.replace('.', '_')
        headers.append(tem_header)
        if 'Severe' in tem_header:
            labels.append(1)
        else:
            labels.append(0)
    return headers, labels
