import pandas as pd
from s2aff.consts import PATHS
from s2aff.file_cache import cached_path


def load_gold_affiliation_annotations(gold_path=PATHS["gold_affiliation_annotations"], keep_non_ror_gold=False):
    """Load the gold annotation file.

    Args:
        gold_path (str, optional): Location of gold. Defaults to PATHS["gold_affiliation_annotations"].
        keep_non_ror_gold (bool, optional): Whether to keep annotations for rows without ROR entries.
            Defaults to False.

    Returns:
        gold_df: DataFrame with gold annotations.
    """
    df = pd.read_csv(cached_path(gold_path))
    df.loc[:, "labels"] = df["labels"].apply(eval)
    if not keep_non_ror_gold:
        keep_flag = df.labels.apply(lambda x: any(["ror.org" in i for i in x]))
        df = df[keep_flag]
    return df
