import hashlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--hash_name', type=str, required=True)
args = parser.parse_args()
with open(args.model_path, 'rb') as f:
    d = hashlib.sha256(f.read()).hexdigest()
with open(f'{args.hash_name}_hash.txt', '+w') as f:
    f.write(d)