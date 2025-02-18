import argparse
import glob
import gzip
import json
import os
import random
import time
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm


def extract(lang: str, sample_rate: float, gzipped_file: str) -> list[str]:
    def keep():
        return random.random() < sample_rate

    # gzipped_file: /home/yus21/clueweb22/outlink/en/en00/en0000/en0000-46.json.gz
    adj_list = []
    with gzip.open(gzipped_file, "rt") as f:
        for line in f:
            line = line.strip()
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = obj["ClueWeb22-ID"]
            outlinks = obj["outlinks"]
            outlinks = [
                x[5]
                for x in outlinks
                if x[5] is not None
                and x[5] != doc_id
                and x[5].startswith(f"clueweb22-{lang}0")
                and keep()
            ]
            if len(outlinks) > 0:
                adj_list.append(f"{doc_id} {' '.join(outlinks)}")
    return adj_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clueweb_path", type=str)
    parser.add_argument("--adjacency_lists_path", type=str)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=128)
    args = parser.parse_args()

    random.seed(args.seed)

    gzipped_files = os.path.join(  # /home/yus21/clueweb22/outlink/en/en00/en0000/en0000-46.json.gz
        args.clueweb_path,  # /home/yus21/clueweb22/
        "outlink",  # outlink
        args.lang,  # en
        "*",  # en00
        "*",  # en0000
        "*.json.gz",  # en0000-46.json.gz
    )
    all_files = glob.glob(gzipped_files)
    all_files.sort()
    print(f"Number of files: {len(all_files)}")

    extract_partial = partial(
        extract,
        args.lang,
        args.sample_rate,
    )
    if not os.path.exists(args.adjacency_lists_path):
        os.makedirs(args.adjacency_lists_path, exist_ok=True)

    saved = 0
    with Pool(args.num_processes) as p:
        cur_adj_list: list[str] = []
        for i, adj_lines in enumerate(
            tqdm(
                p.imap(extract_partial, all_files),
                total=len(all_files),
            )
        ):
            cur_adj_list.extend(adj_lines)
            if (i + 1) % args.save_every == 0 or i == len(all_files) - 1:
                with open(os.path.join(args.adjacency_lists_path, f"{saved:05d}.txt"), "w") as f:
                    for line in cur_adj_list:
                        f.write(line + "\n")
                if args.compress:
                    time.sleep(1)
                    with os.popen(
                        f"zstd {os.path.join(args.adjacency_lists_path, f'{saved:05d}.txt')} --rm"
                    ) as p:
                        pass
                cur_adj_list = []
                saved += 1


if __name__ == "__main__":
    main()
