import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--file_path",
    type=str
)

parser.add_argument(
    "-i_l",
    "--initial_line",
    default=0,
    type=int
)

parser.add_argument(
    "-b_l",
    "--blank_line",
    default="False",
    type=str
)

args = parser.parse_args()

count = 0
with open(args.file_path,'r') as csvin, open('newtsv.tsv', 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in tqdm(csvin):
        if args.initial_line <= count:
                if args.blank_line == "True":
                    tsvout.writerow([row[0].strip()])
                else:
                    tsvout.writerow(row)
        count += 1
