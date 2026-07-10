import pandas as pd

test_df = pd.read_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/test.csv")
train_df = pd.read_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/train.csv")

test_df.to_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/test_init.csv")
train_df.to_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/train_init.csv")

test_df = test_df[["date", "Wholesale broiler composite", "Retail-wholesale spread for broiler composite", "OT"]]
train_df = train_df[["date", "Wholesale broiler composite", "Retail-wholesale spread for broiler composite", "OT"]]

test_df.to_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/test.csv", index=False)
train_df.to_csv("/data/triho/Time_Series_Research/Baselines/AlphaCast/data/Time-MDD/Agriculture/train.csv", index=False)
