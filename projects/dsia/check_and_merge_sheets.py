import os
import pandas as pd
import numpy as np
from pathlib import Path


def check_and_merge_year(year: int):
    print(f"===  {year}  ===")

    p_key = "p_izo"
    num_keys = 22

    if 2018 <= year < 2020:
        num_keys = 21
    elif year == 2017:
        num_keys = 20
        p_key = "P_IZO"
    elif year < 2017:
        raise ValueError(f"Year {year} not supported!")

    home = Path(os.environ["HOME"])
    data_root = home / "data" / "projects" / "idea" / "data" / "dsia" / "data"
    file_name = "M03_{}.xlsx"

    excel_file = pd.ExcelFile(data_root / str(year) / file_name.format(year))
    
    # first as the shape etc
    sheet1 = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])

    to_skip = [f"v03{str(year)[-2:]}{t}" for t in ["b", "o", "t", "_43", "43"]]
    keys = sheet1.columns[:num_keys]
    keys_wo_p_izo = keys.drop(p_key)
    df_keys = sheet1[keys]
    assert len(df_keys) == len(df_keys[p_key].unique()), f"{p_key} is not unique!"
    exp_len = len(df_keys)
    df_keys_sorted = df_keys.sort_values(p_key).reset_index(drop=True)
    df = df_keys_sorted.copy()
    cols_so_far = []

    for sn in excel_file.sheet_names:
        print(f"Processing sheet {sn}")
        other = pd.read_excel(excel_file, sheet_name=sn)

        # check the shape
        if sn in to_skip:
            assert len(other) != exp_len
            print(f"Sheet {sn} has shape {other.shape} and should be excluded (not len {exp_len})")
            continue

        assert len(other) == exp_len
        print(f"Sheet {sn} has shape {other.shape} and the expected number of rows")

        # check the key cols are identical
        if not df_keys_sorted.equals(other[keys].sort_values(p_key).reset_index(drop=True)):
            num_diffs = 0
            other_sorted = other[keys].sort_values(p_key).reset_index(drop=True)
            for i, row in df_keys_sorted.iterrows():
                # assert row.equals(other_sorted.loc[i]), f"diff at {i}: row = {row}, other = {other_sorted.loc[i]}"
                if not row.equals(other_sorted.loc[i]):
                    print(f"Diff!: {row} vs {other_sorted.loc[i]}")
                    num_diffs += 1
            assert num_diffs <= 1, f"num_diffs = {num_diffs}"

        # assert df_keys_sorted.equals(other[keys].sort_values(p_key).reset_index(drop=True)), \
        #     f"Key columns are not identical for {sn}"

        other = other.drop(columns=keys_wo_p_izo)
        if "zujzriz" in cols_so_far and "zujzriz" in other.columns:
            other = other.drop(columns="zujzriz")
        other_cols = other.drop(columns=[p_key]).columns
        intersect = [c for c in other_cols if c in cols_so_far]
        assert len(intersect) == 0, f"There are overlapping columns: {intersect}!"
        cols_so_far += list(other_cols)
        df = pd.merge(df, other, on=p_key, how="outer")
        assert len(df) == exp_len, "Not the expected length!"
        print(f"{sn} merged, curr shape = {df.shape}")

    print("All sheets processed, storing")
    df.to_csv(data_root / ".." / "clean" / f"M03_{year}.csv")
    df.to_parquet(data_root / ".." / "clean" / f"M03_{year}.parquet")


year = 2017
check_and_merge_year(year)

