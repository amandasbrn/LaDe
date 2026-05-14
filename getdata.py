import argparse
import os
import pandas as pd

HUGGINGFACE_DATASETS = {
    'pickup': 'LaDe-P',
}

PICKUP_SPLITS = {
    'sh': 'data/pickup_sh-00000-of-00001-79fabe8088e723a2.parquet',
    'hz': 'data/pickup_hz-00000-of-00001-2641abebfe50648a.parquet',
    'cq': 'data/pickup_cq-00000-of-00001-a172031e5392f9d3.parquet',
    'jl': 'data/pickup_jl-00000-of-00001-9b430a56a935f284.parquet',
    'yt': 'data/pickup_yt-00000-of-00001-6d21a4dccd28ee03.parquet',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and save LaDe pickup data in /data/raw/<type>/<file>.csv format.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='pickup',
        choices=['pickup'],
        help='Dataset type to fetch. Currently only pickup is supported.'
    )
    parser.add_argument(
        '--city',
        type=str,
        default='sh',
        choices=['sh', 'hz', 'cq', 'jl', 'yt'],
        help='City code for the dataset file.',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='data/raw',
        help='Root output folder for the saved CSV file.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.lower()
    city = args.city.lower()

    if dataset != 'pickup':
        raise ValueError('Only pickup is supported by this script at the moment.')

    hf_dataset = HUGGINGFACE_DATASETS[dataset]
    if city not in PICKUP_SPLITS:
        raise ValueError(f'Unknown city code: {city}. Use one of {list(PICKUP_SPLITS.keys())}.')

    hf_path = f'hf://datasets/Cainiao-AI/{hf_dataset}/{PICKUP_SPLITS[city]}'
    print(f'Reading data from {hf_path}')
    df = pd.read_parquet(hf_path)

    output_dir = os.path.join(args.output_root, dataset)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f'{dataset}_{city}.csv'
    output_path = os.path.join(output_dir, output_filename)

    print(f'Saving CSV to {output_path}')
    df.to_csv(output_path, index=False)
    print('Done.')


if __name__ == '__main__':
    main()
