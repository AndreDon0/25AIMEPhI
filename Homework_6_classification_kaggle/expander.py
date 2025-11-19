from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import pandas as pd

class Expander:
    def __init__(self, multi_columns: dict, one_columns: dict):
        self.multis = {}
        self.ones = {}
        for column in multi_columns:
            self.multis[column] = MultiLabelBinarizer()
        for column in one_columns:
            self.ones[column] = OneHotEncoder(handle_unknown='infrequent_if_exist')
    
    def fit(self, df: pd.DataFrame):
        for column, multi in self.multis.items():
            multi.fit(df[column])
        for column, one in self.ones.items():
            one.fit(df[[column]])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for column, multi in self.multis.items():
            transformed = multi.transform(df[column])
            col_names = [f"{column}__{cls}" for cls in multi.classes_]
            df_transformed = pd.DataFrame(transformed, columns=col_names, index=df.index)
            df = pd.concat([df, df_transformed], axis=1)
            df = df.drop(columns=[column])

        for column, one in self.ones.items():
            transformed = one.transform(df[[column]]).toarray()
            col_names = [f"{column}__{cat}" for cat in one.categories_[0]]
            df_transformed = pd.DataFrame(transformed, columns=col_names, index=df.index)
            df = pd.concat([df, df_transformed], axis=1)
            df = df.drop(columns=[column])

        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit()
        return self.transform(df)