import pandas as pd
import requests
from random import shuffle

# get json from wikidata raw url
# for example http://www.wikidata.org/entity/Q42493 turns into
# https://www.wikidata.org/wiki/Special:EntityData/Q42493.json
# then we get it via requests
def get_wikidata(original_url):
    url = original_url.replace("www.wikidata.org/wiki/", "www.wikidata.org/wiki/Special:EntityData/") + ".json"
    r = requests.get(url)
    rjson = r.json()
    try:
        return {list(rjson["entities"].values())[0]["labels"]["en"]["value"]}
    except:
        return pd.NA


df1 = pd.read_csv("gold_affiliation_annotations_2021.csv")
df2 = pd.read_csv("gold_affiliation_annotations_2022.csv")

# valid only
df2 = df2[df2.valid_affiliation_string == "yes"].drop(
    ["valid_affiliation_string", "openalex_institution_ids", "annotator"], axis=1
)

# sometimes incorrect, but there isn't a ror link and there is no wikidata link
incorrect_and_no_label = (df2.correct == "no") & df2.ror_labels_or_wikidata.apply(
    lambda s: pd.isna(s) or ("ror.org" not in s and "wikidata.org" not in s)
)
df2 = df2[~incorrect_and_no_label]

# custom processing for where there ror link
def process_ror_str(s):
    if not pd.isna(s):
        if "ror" in s:
            if "," in s:
                return set([i.strip() for i in s.split(",")])
            else:
                return set([s.strip()])
    return s


df2.loc[:, "ror_labels_or_wikidata"] = df2.ror_labels_or_wikidata.apply(process_ror_str)

# if ror_ids exists and it's correct and there is no ror_labels_or_wikidata -> move over ror_ids
correct_no_label = (df2.correct == "yes") & pd.isna(df2.ror_labels_or_wikidata) & ~pd.isna(df2.ror_ids)
df2.loc[correct_no_label, "ror_labels_or_wikidata"] = df2.loc[correct_no_label, "ror_ids"].apply(lambda s: set(eval(s)))

# get wikidata labels
has_wikidata = df2.ror_labels_or_wikidata.apply(lambda s: not pd.isna(s) and "wikidata.org" in s)
df2.loc[has_wikidata, "ror_labels_or_wikidata"] = df2.loc[has_wikidata, "ror_labels_or_wikidata"].apply(get_wikidata)

# for those still missing, just move over the raw_affiliation_string
still_missing = df2.ror_labels_or_wikidata.isna()
df2.loc[still_missing, "ror_labels_or_wikidata"] = df2.loc[still_missing, "raw_affiliation_string"]

# which ones aren't sets? some are HTML and some are strings
# drop HTML and convert strings to a set
def process_not_set(s):
    if "http" in s:
        return pd.NA
    else:
        return set([s])


not_set = df2.ror_labels_or_wikidata.apply(lambda s: not isinstance(s, set))
df2.loc[not_set, "ror_labels_or_wikidata"] = df2.loc[not_set, "ror_labels_or_wikidata"].apply(process_not_set)
df2.dropna(subset=["ror_labels_or_wikidata"], inplace=True)

# some of the sets have both ror and not ror
# if so, we should drop the rors only
def find_multi_source(s):
    has_ror = any(["ror.org" in i for i in s])
    if has_ror:
        return set([i for i in s if "ror.org" in i])
    else:
        return s


df2.loc[:, "ror_labels_or_wikidata"] = df2.ror_labels_or_wikidata.apply(find_multi_source)

# combine and save
df1_sub = df1[["original_affiliation", "mag_correct", "labels", "split"]]
df2_sub = df2[["raw_affiliation_string", "correct", "ror_labels_or_wikidata"]]
# randomly assign df2_sub.split one of:  'train' 'val' 'test'
n = df2_sub.shape[0]
train_val_test = ["train"] * int(n * 0.6) + ["val"] * int(n * 0.2)
train_val_test += ["test"] * (n - len(train_val_test))

shuffle(train_val_test)
df2_sub.loc[:, "split"] = train_val_test

df1_sub.columns = ["original_affiliation", "original_labels_correct", "labels", "split"]
df2_sub.columns = ["original_affiliation", "original_labels_correct", "labels", "split"]

# how correct are df1 and df2?
mag_correct = (df1_sub.original_labels_correct == "yes").mean()
openalex_correct = (df2_sub.original_labels_correct == "yes").mean()
print("MAG correct:", mag_correct)  # 0.77
print("OpenAlex correct:", openalex_correct)  # 0.83

df = pd.concat([df1_sub, df2_sub])

df.to_csv("gold_affiliation_annotations.csv", index=False)


rorless = df2_sub.labels.apply(lambda s: any([i for i in s if "ror.org" not in i]))
