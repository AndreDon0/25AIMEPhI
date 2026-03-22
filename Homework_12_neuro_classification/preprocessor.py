from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


class OilFieldClassificationPreprocessor:
    """Class-based preprocessing pipeline for oil-field classification."""

    TARGET_COL = "Onshore/Offshore"
    DROP_COLS = ("Field name", "Reservoir unit", "Latitude", "Longitude")
    FILL_UNKNOWN_COLS = ("Country", "Region", "Basin name")
    MULTI_COLUMNS = ("Country", "Tectonic regime", "Operator company", "Structural setting")
    ONE_COLUMNS = (
        "Region",
        "Basin name",
        "Hydrocarbon type",
        "Reservoir status",
        "Reservoir period",
        "Lithology",
    )
    TARGET_MAPPING = {"ONSHORE": 1, "OFFSHORE": 0, "ONSHORE-OFFSHORE": 2}

    def __init__(
        self,
        random_state: int = 42,
        smote_k_neighbors: int = 2,
    ) -> None:
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors

        self.scaler = StandardScaler()
        self.mlb_: dict[str, MultiLabelBinarizer] = {
            col: MultiLabelBinarizer() for col in self.MULTI_COLUMNS
        }
        self.one_categories_: dict[str, list[str]] = {}
        self.feature_columns_: list[str] | None = None
        self._is_fitted = False

    def __call__(
        self,
        data: pd.DataFrame,
        *,
        fit: bool = False,
        include_target: bool = False,
        apply_sampling: bool = True,
    ) -> pd.DataFrame:
        """Operator-overloaded entry point: prep(df, fit=True/False)."""
        if fit:
            return self.fit_transform(
                data,
                include_target=include_target,
                apply_sampling=apply_sampling,
            )
        return self.transform(data, include_target=include_target)

    def __matmul__(self, data: pd.DataFrame) -> pd.DataFrame:
        """`prep @ df` means fit + transform for training data."""
        include_target = self.TARGET_COL in data.columns
        return self.fit_transform(data, include_target=include_target, apply_sampling=True)

    def fit(self, data: pd.DataFrame) -> "OilFieldClassificationPreprocessor":
        df = self._prepare_base(data, include_target=True)
        features = df.drop(columns=[self.TARGET_COL], errors="ignore")
        expanded = self._expand_features(features, fit=True)

        self.feature_columns_ = expanded.columns.tolist()
        self.scaler.fit(expanded[self.feature_columns_])
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
        self._ensure_fitted()

        df = self._prepare_base(data, include_target=include_target)
        target = None
        if include_target and self.TARGET_COL in df.columns:
            target = df[self.TARGET_COL].copy()
            df = df.drop(columns=[self.TARGET_COL])

        expanded = self._expand_features(df, fit=False)
        expanded = expanded.reindex(columns=self.feature_columns_, fill_value=0.0)

        scaled_values = self.scaler.transform(expanded[self.feature_columns_])
        result = pd.DataFrame(scaled_values, columns=self.feature_columns_, index=expanded.index)

        if include_target and target is not None:
            result[self.TARGET_COL] = target.values
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        *,
        include_target: bool = True,
        apply_sampling: bool = True,
    ) -> pd.DataFrame:
        transformed = self.fit(data).transform(data, include_target=include_target)
        if include_target and apply_sampling:
            transformed = self._sample_training(transformed)
        return transformed

    def prepare_training_data(
        self,
        data: pd.DataFrame,
        *,
        use_validation: bool = True,
        val_size: float = 0.2,
        random_state: int | None = None,
        apply_sampling: bool = True,
        stratify: bool = True,
    ) -> dict[str, pd.DataFrame | None]:
        """
        Build train/val datasets or train-on-full dataset.

        Returns:
            {"train": train_df, "val": val_df_or_none}
        """
        if self.TARGET_COL not in data.columns:
            raise ValueError(f"Input data must contain target column `{self.TARGET_COL}`.")

        seed = self.random_state if random_state is None else random_state

        if not use_validation:
            train_df = self.fit_transform(
                data,
                include_target=True,
                apply_sampling=apply_sampling,
            )
            return {"train": train_df, "val": None}

        stratify_target: pd.Series | None = None
        if stratify:
            stratify_target = self._encode_target(data[self.TARGET_COL])

        train_raw, val_raw = train_test_split(
            data,
            test_size=val_size,
            random_state=seed,
            stratify=stratify_target,
        )

        self.fit(train_raw)
        train_df = self.transform(train_raw, include_target=True)
        if apply_sampling:
            train_df = self._sample_training(train_df)
        val_df = self.transform(val_raw, include_target=True)
        return {"train": train_df, "val": val_df}

    def _prepare_base(self, data: pd.DataFrame, include_target: bool) -> pd.DataFrame:
        df = data.copy()
        df = df.drop(columns=list(self.DROP_COLS), errors="ignore")
        df = self._fill_missing(df)
        df = self._split_multilabel_columns(df)

        if include_target and self.TARGET_COL in df.columns:
            df[self.TARGET_COL] = self._encode_target(df[self.TARGET_COL])
        return df

    def _fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col in self.FILL_UNKNOWN_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        for col in self.MULTI_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("")
        for col in self.ONE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    def _split_multilabel_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        separators = {
            "Country": " /",
            "Tectonic regime": "/",
            "Operator company": " /",
            "Structural setting": "/",
        }
        for col, sep in separators.items():
            if col not in df.columns:
                continue
            values = df[col].astype(str)
            values = values.replace({"nan": "", "None": ""})
            df[col] = values.apply(lambda x: [v.strip() for v in x.split(sep) if v.strip()])
        return df

    def _expand_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = data.copy()

        passthrough_cols = [
            c for c in df.columns if c not in self.MULTI_COLUMNS and c not in self.ONE_COLUMNS
        ]
        expanded_parts = [df[passthrough_cols].copy()]

        for col in self.MULTI_COLUMNS:
            if col not in df.columns:
                values = pd.Series([[] for _ in range(len(df))], index=df.index)
            else:
                values = df[col]

            if fit:
                arr = self.mlb_[col].fit_transform(values)
            else:
                arr = self.mlb_[col].transform(values)

            part = pd.DataFrame(
                arr,
                columns=[f"{col}__{name}" for name in self.mlb_[col].classes_],
                index=df.index,
            )
            expanded_parts.append(part)

        for col in self.ONE_COLUMNS:
            if col not in df.columns:
                values = pd.Series(["Unknown"] * len(df), index=df.index)
            else:
                values = df[col].astype(str).fillna("Unknown")

            if fit:
                categories = sorted(values.dropna().unique().tolist())
                if not categories:
                    categories = ["Unknown"]
                self.one_categories_[col] = categories
            categories = self.one_categories_.get(col, ["Unknown"])

            categorical = pd.Categorical(values, categories=categories)
            part = pd.get_dummies(categorical, prefix=col, dtype=float)
            part.index = df.index
            expanded_parts.append(part)

        expanded = pd.concat(expanded_parts, axis=1)
        expanded = expanded.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return expanded

    def _sample_training(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.TARGET_COL not in data.columns:
            return data

        x_data = data.drop(columns=[self.TARGET_COL], errors="ignore")
        y_data = data[self.TARGET_COL]
        if y_data.nunique(dropna=True) < 2:
            return data

        smote = SMOTE(
            sampling_strategy="auto",
            random_state=self.random_state,
            k_neighbors=self.smote_k_neighbors,
        )
        try:
            x_resampled, y_resampled = smote.fit_resample(x_data, y_data)
        except ValueError:
            # Keep original data if a class has too few samples for current k_neighbors.
            return data

        y_series = pd.Series(y_resampled, name=self.TARGET_COL)
        sampled = pd.concat(
            [pd.DataFrame(x_resampled, columns=x_data.columns), y_series],
            axis=1,
        )
        sampled.reset_index(drop=True, inplace=True)
        return sampled

    def _encode_target(self, target: pd.Series) -> pd.Series:
        mapped = target.map(self.TARGET_MAPPING)
        numeric = pd.to_numeric(target, errors="coerce")
        return mapped.where(mapped.notna(), numeric)

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self.feature_columns_ is None:
            raise RuntimeError("Preprocessor is not fitted yet. Call `fit` before `transform`.")


_default_preprocessor = OilFieldClassificationPreprocessor()


def pipeline(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    - fit=True: fit + transform + optional sampling, with target if present.
    - fit=False: transform only, no fitting.
    """
    include_target = fit and _default_preprocessor.TARGET_COL in df.columns
    return _default_preprocessor(
        df,
        fit=fit,
        include_target=include_target,
        apply_sampling=True,
    )


def prepare_training_data(
    df: pd.DataFrame,
    *,
    use_validation: bool = True,
    val_size: float = 0.2,
    random_state: int = 42,
    apply_sampling: bool = True,
    stratify: bool = True,
) -> dict[str, pd.DataFrame | None]:
    """Module-level helper for split/full-dataset training workflows."""
    processor = OilFieldClassificationPreprocessor(random_state=random_state)
    return processor.prepare_training_data(
        df,
        use_validation=use_validation,
        val_size=val_size,
        random_state=random_state,
        apply_sampling=apply_sampling,
        stratify=stratify,
    )
