import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
LABELS_CSV   = "xr.csv"
FULL_LIST    = "batch_4686.txt"
BATCH_FILES  = ["batch_500.txt", "batch_1000.txt", "batch_1500.txt",
                "batch_2000.txt", "batch_3000.txt", "batch_4686.txt"]
TEST_SIZE    = 0.20
SEED         = 42
OUT_DIR      = "."   # writes outputs next to this script

# ---------------- LOAD LABELS ----------------
xr = pd.read_csv(LABELS_CSV)
xr["SEQN"] = xr["SEQN"].astype(str)
grade_r = dict(zip(xr.SEQN, xr.XRPKLR))   # right-knee KL grade
grade_l = dict(zip(xr.SEQN, xr.XRPKLL))   # left-knee  KL grade

name_pat = re.compile(r"^(\d+)K1_(L|R)\.tiff$", re.IGNORECASE)


def parse(name):
    m = name_pat.match(name.strip())
    if not m:
        return None
    seqn, side = m.group(1), m.group(2).upper()
    grade = grade_r.get(seqn) if side == "R" else grade_l.get(seqn)
    return seqn, side, grade


def load_list(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            name = line.strip()
            if not name:
                continue
            p = parse(name)
            if p is None:
                raise ValueError(f"Unparseable filename: {name}")
            rows.append((name, *p))
    return pd.DataFrame(rows, columns=["image", "seqn", "side", "grade"])


# ---------------- BUILD FULL TABLE ----------------
df = load_list(FULL_LIST)

# Subject-level stratum = highest KL grade across that subject's knees.
# This keeps the rare high grades proportionally represented in both splits.
subj = (df.groupby("seqn")["grade"].max()
          .rename("subj_grade").reset_index())

# ---------------- STRATIFIED SUBJECT SPLIT ----------------
train_subj, test_subj = train_test_split(
    subj["seqn"],
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=subj["subj_grade"],
)
test_subj = set(test_subj)
train_subj = set(train_subj)

df["split"] = df["seqn"].apply(lambda s: "test" if s in test_subj else "train")

# ---------------- SANITY CHECKS ----------------
assert test_subj.isdisjoint(train_subj), "subject overlap!"
overlap_imgs = set(df[df.split == "test"].seqn) & set(df[df.split == "train"].seqn)
assert not overlap_imgs, "subject leaked across splits!"

# ---------------- WRITE FROZEN SPLIT ----------------
df.sort_values("image").to_csv(f"{OUT_DIR}/frozen_split_manifest.csv", index=False)

test_df  = df[df.split == "test"]
train_df = df[df.split == "train"]

with open(f"{OUT_DIR}/frozen_test.txt", "w") as fh:
    fh.write("\n".join(sorted(test_df.image)) + "\n")
with open(f"{OUT_DIR}/train_pool.txt", "w") as fh:
    fh.write("\n".join(sorted(train_df.image)) + "\n")


def dist(frame):
    return frame.grade.value_counts().sort_index().to_dict()


print("=" * 60)
print("FROZEN SUBJECT-LEVEL SPLIT   (seed=%d, test=%.0f%%)"
      % (SEED, TEST_SIZE * 100))
print("=" * 60)
print(f"Subjects : {len(train_subj)} train  |  {len(test_subj)} test")
print(f"Images   : {len(train_df)} train  |  {len(test_df)} test")
print(f"\nTRAIN pool grade dist : {dist(train_df)}")
print(f"TEST  set  grade dist : {dist(test_df)}")
print(f"\nGrade 4 -> {dist(train_df).get(4,0)} trainable, "
      f"{dist(test_df).get(4,0)} frozen in test")

# ---------------- FILTER EACH BATCH (remove test subjects) ----------------
print("\n" + "=" * 60)
print("LEAKAGE-FREE TRAINING BATCHES (test subjects removed)")
print("=" * 60)
print(f"{'batch':<14}{'orig':>6}{'removed':>9}{'final':>7}   grade dist (final)")
for bf in BATCH_FILES:
    bdf = load_list(bf)
    keep = bdf[~bdf.seqn.isin(test_subj)].copy()
    removed = len(bdf) - len(keep)
    out_name = bf.replace(".txt", "_train.txt")
    with open(f"{OUT_DIR}/{out_name}", "w") as fh:
        fh.write("\n".join(sorted(keep.image)) + "\n")
    print(f"{bf:<14}{len(bdf):>6}{removed:>9}{len(keep):>7}   {dist(keep)}")

print("\nWrote: frozen_test.txt, train_pool.txt, frozen_split_manifest.csv,")
print("       and *_train.txt for each batch.")
