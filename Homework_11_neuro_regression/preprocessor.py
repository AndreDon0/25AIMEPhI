from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


class GameRatingPreprocessor:
    """Class-based preprocessing pipeline for BGG game rating regression."""

    TARGET_COL = "Rating Average"
    DROP_COLS = ("Name",)
    MULTILABEL_COLS = ("Mechanics", "Domains")
    ZERO_AS_MISSING_COLS = (
        "Year Published",
        "Min Players",
        "Max Players",
        "Play Time",
        "Min Age",
        "Complexity Average",
    )
    BASE_NUMERIC_COLS = (
        "Year Published",
        "Min Players",
        "Max Players",
        "Play Time",
        "Min Age",
        "Users Rated",
        "BGG Rank",
        "Complexity Average",
        "Owned Users",
    )
    MIN_CATEGORY_COUNT = 50

    def __init__(self, imputation_strategy: str = "mean") -> None:
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
        self.scaler = StandardScaler()
        self.mlb_mechanics = MultiLabelBinarizer()
        self.mlb_domains = MultiLabelBinarizer()

        self.year_min_: float | None = None
        self.year_max_: float | None = None
        self.complexity_min_: float | None = None
        self.complexity_max_: float | None = None
        self.min_age_min_: float | None = None
        self.min_age_max_: float | None = None

        self.kept_mlb_cols_: list[str] | None = None
        self.base_cols_: list[str] | None = None
        self.feature_columns_: list[str] | None = None
        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> "GameRatingPreprocessor":
        df = data.copy()
        df = self._convert_numeric_columns(df, include_target=True)
        df = self._clean_missing_values(df)

        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Training data must contain target column `{self.TARGET_COL}`.")
        target = df[self.TARGET_COL].copy()
        df = df.drop(columns=[self.TARGET_COL])

        df = self._add_multilabel_counts(df)
        df = self._expand_multilabel_columns(df, fit=True)

        self.base_cols_ = [c for c in self.BASE_NUMERIC_COLS if c in df.columns]
        self.imputer.fit(df[self.base_cols_])
        df[self.base_cols_] = self.imputer.transform(df[self.base_cols_])

        self.year_min_ = float(df["Year Published"].min())
        self.year_max_ = float(df["Year Published"].max())
        self.complexity_min_ = float(df["Complexity Average"].min())
        self.complexity_max_ = float(df["Complexity Average"].max())
        self.min_age_min_ = float(df["Min Age"].min())
        self.min_age_max_ = float(df["Min Age"].max())

        df = self._feature_engineering(df)

        mlb_cols = [c for c in df.columns if c.startswith("Mech_") or c.startswith("Dom_")]
        col_sums = df[mlb_cols].sum()
        self.kept_mlb_cols_ = col_sums[col_sums >= self.MIN_CATEGORY_COUNT].index.tolist()
        drop_cols = [c for c in mlb_cols if c not in self.kept_mlb_cols_]
        df = df.drop(columns=drop_cols)

        self.feature_columns_ = df.columns.tolist()
        self.scaler.fit(df[self.feature_columns_])

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        self._ensure_fitted()

        df = data.copy()
        df = self._convert_numeric_columns(df, include_target=include_target)
        df = self._clean_missing_values(df)

        target = None
        if include_target and self.TARGET_COL in df.columns:
            target = df[self.TARGET_COL].copy()
            df = df.drop(columns=[self.TARGET_COL])

        df = self._add_multilabel_counts(df)
        df = self._expand_multilabel_columns(df, fit=False)

        df[self.base_cols_] = self.imputer.transform(df[self.base_cols_])

        df = self._feature_engineering(df)

        df = df.reindex(columns=self.feature_columns_, fill_value=0)

        scaled_values = self.scaler.transform(df[self.feature_columns_])
        result = pd.DataFrame(scaled_values, columns=self.feature_columns_, index=df.index)

        if include_target and target is not None:
            result[self.TARGET_COL] = target

        return result

    def fit_transform(self, data: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        return self.fit(data).transform(data, include_target=include_target)

    def _convert_numeric_columns(self, data: pd.DataFrame, include_target: bool) -> pd.DataFrame:
        converted = data.drop(columns=list(self.DROP_COLS), errors="ignore")
        if "Complexity Average" in converted.columns:
            converted["Complexity Average"] = self._parse_decimal_column(converted["Complexity Average"])
        if include_target and self.TARGET_COL in converted.columns:
            converted[self.TARGET_COL] = self._parse_decimal_column(converted[self.TARGET_COL])
        return converted

    @staticmethod
    def _parse_decimal_column(series: pd.Series) -> pd.Series:
        as_text = series.astype(str).str.replace(",", ".", regex=False)
        as_text = as_text.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        return pd.to_numeric(as_text, errors="coerce")

    def _clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.copy()
        for col in self.ZERO_AS_MISSING_COLS:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].replace(0, np.nan)
        return cleaned

    def _add_multilabel_counts(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col_name in self.MULTILABEL_COLS:
            series = df.get(col_name, pd.Series("", index=df.index))
            split = self._split_multilabel(series)
            df[f"{col_name} Count"] = split.apply(len)
        return df

    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        engineered = data.copy()
        eps = 1e-6

        year_min = self.year_min_ if self.year_min_ is not None else engineered["Year Published"].min()
        year_max = self.year_max_ if self.year_max_ is not None else engineered["Year Published"].max()
        complexity_min = self.complexity_min_ if self.complexity_min_ is not None else engineered["Complexity Average"].min()
        complexity_max = self.complexity_max_ if self.complexity_max_ is not None else engineered["Complexity Average"].max()
        min_age_min = self.min_age_min_ if self.min_age_min_ is not None else engineered["Min Age"].min()
        min_age_max = self.min_age_max_ if self.min_age_max_ is not None else engineered["Min Age"].max()

        engineered["Log Year"] = np.log(np.clip(engineered["Year Published"] - year_min + 1, 1, None))
        engineered["Log Min Players"] = np.log(engineered["Min Players"] + 1)
        engineered["Log Max Players"] = np.log(engineered["Max Players"] + 1)
        engineered["Log Play Time"] = np.log(engineered["Play Time"] + 1)
        engineered["Log Min Age"] = np.log(engineered["Min Age"] + 1)
        engineered["Log Users Rated"] = np.log(engineered["Users Rated"] + 1)
        engineered["Log Owned Users"] = np.log(engineered["Owned Users"] + 1)
        engineered["Log BGG Rank"] = np.log(engineered["BGG Rank"] + 1)
        engineered["Log Complexity Average"] = np.log(engineered["Complexity Average"] + 0.1)

        engineered["Avg Players"] = (engineered["Min Players"] + engineered["Max Players"]) / 2
        engineered["Player Range"] = engineered["Max Players"] - engineered["Min Players"]
        engineered["Player Flexibility"] = engineered["Player Range"] / (engineered["Max Players"] + 1)
        engineered["Ownership Rate"] = engineered["Owned Users"] / (engineered["Users Rated"] + 1)
        engineered["Play Time x Min Age"] = engineered["Play Time"] * engineered["Min Age"]
        engineered["Complexity x Players"] = engineered["Complexity Average"] * engineered["Avg Players"]

        engineered["Complexity Squared"] = engineered["Complexity Average"] ** 2
        engineered["Year Squared"] = (engineered["Year Published"] - year_min) ** 2
        engineered["Popularity Score"] = (engineered["Log Owned Users"] + engineered["Log Users Rated"]) / 2
        engineered["Rank Score"] = 1 / (engineered["BGG Rank"] / 10000 + 1)

        engineered["Years Since Release"] = year_max - engineered["Year Published"]
        engineered["Is Recent"] = (engineered["Years Since Release"] <= 5).astype(int)
        engineered["Is Classic"] = (engineered["Years Since Release"] > 15).astype(int)

        engineered["Min Age Category"] = pd.cut(
            engineered["Min Age"], bins=[0, 6, 12, 16, 100], labels=[1, 2, 3, 4],
            ordered=True, include_lowest=True,
        ).cat.codes.clip(lower=0)
        engineered["Play Time Category"] = pd.cut(
            engineered["Play Time"], bins=[0, 30, 60, 120, 10000], labels=[1, 2, 3, 4],
            ordered=True, include_lowest=True,
        ).cat.codes.clip(lower=0)
        engineered["Complexity Tier"] = pd.cut(
            engineered["Complexity Average"], bins=[0, 2, 3, 4, 5], labels=[1, 2, 3, 4],
            ordered=True, include_lowest=True,
        ).cat.codes.clip(lower=0)

        complexity_span = complexity_max - complexity_min
        age_span = min_age_max - min_age_min
        engineered["Complexity Normalized"] = (
            engineered["Complexity Average"] - complexity_min
        ) / (complexity_span + eps)
        engineered["Age Normalized"] = (
            engineered["Min Age"] - min_age_min
        ) / (age_span + eps)
        engineered["Complex Game"] = (engineered["Complexity Average"] >= 3).astype(int)
        engineered["Long Game"] = (engineered["Play Time"] >= 120).astype(int)
        engineered["Multiplayer Focused"] = (engineered["Avg Players"] >= 3).astype(int)
        engineered["Casual Index"] = (1 - engineered["Complexity Normalized"]) * (1 - engineered["Age Normalized"])
        engineered["Social Index"] = engineered["Avg Players"] / (engineered["Max Players"] + 1)

        return engineered

    def _expand_multilabel_columns(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        expanded = data.copy()
        mechanics = self._split_multilabel(expanded.get("Mechanics", pd.Series("", index=expanded.index)))
        domains = self._split_multilabel(expanded.get("Domains", pd.Series("", index=expanded.index)))

        if fit:
            mech_encoded = self.mlb_mechanics.fit_transform(mechanics)
            dom_encoded = self.mlb_domains.fit_transform(domains)
        else:
            mech_encoded = self.mlb_mechanics.transform(mechanics)
            dom_encoded = self.mlb_domains.transform(domains)

        mech_df = pd.DataFrame(
            mech_encoded,
            columns=[f"Mech_{name}" for name in self.mlb_mechanics.classes_],
            index=expanded.index,
        )
        dom_df = pd.DataFrame(
            dom_encoded,
            columns=[f"Dom_{name}" for name in self.mlb_domains.classes_],
            index=expanded.index,
        )

        expanded = expanded.drop(columns=list(self.MULTILABEL_COLS), errors="ignore")
        expanded = pd.concat([expanded, mech_df, dom_df], axis=1)
        return expanded

    @staticmethod
    def _split_multilabel(series: pd.Series) -> pd.Series:
        values = series.fillna("").astype(str)
        return values.apply(lambda x: [item.strip() for item in x.split(",") if item.strip()])

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self.feature_columns_ is None:
            raise RuntimeError("Preprocessor is not fitted yet. Call `fit` before `transform`.")
