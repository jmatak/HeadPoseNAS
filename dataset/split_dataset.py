CSV = "./pose_300w.csv"
CSV_T = "./pose_300w_train.csv"
CSV_V = "./pose_300w_val.csv"

split = 0.7
lines = open(CSV,"r").readlines()
lines = [l.strip() for l in lines]
n = len(lines)
n = int(n * split)

train = lines[:n]
validation = lines[n:]

open(CSV_T, "w+").write("\n".join(train))
open(CSV_V, "w+").write("\n".join(validation))