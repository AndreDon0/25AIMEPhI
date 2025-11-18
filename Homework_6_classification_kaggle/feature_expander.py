from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np

class FeatureExpander:
    """
    Optimized feature expansion class for multi-label and categorical encoding.
    Handles rare categories and ensures train-test consistency.
    """
    
    def __init__(self, 
                 multi_label_cols=None,
                 one_hot_cols=None,
                 min_frequency=1,
                 handle_unknown='ignore'):
        """
        Parameters:
        -----------
        multi_label_cols : list
            Columns for MultiLabelBinarizer encoding
        one_hot_cols : list
            Columns for OneHotEncoder encoding
        min_frequency : int
            Minimum number of samples for a class to be kept (default=1)
        handle_unknown : str
            How to handle unknown categories in test set ('ignore' or 'infrequent_if_exist')
        """
        self.multi_label_cols = multi_label_cols or [
            'Country', 'Tectonic regime', 'Operator company', 'Structural setting'
        ]
        self.one_hot_cols = one_hot_cols or [
            'Region', 'Basin name', 'Hydrocarbon type', 
            'Reservoir status', 'Reservoir period', 'Lithology'
        ]
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        
        # Initialize encoders
        self.multis = {col: MultiLabelBinarizer() for col in self.multi_label_cols}
        self.ones = {
            col: OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=False,  # More efficient than .toarray()
                min_frequency=min_frequency if min_frequency > 1 else None,
                dtype=np.uint8  # Use smaller dtype for binary features
            ) 
            for col in self.one_hot_cols
        }
        
        self.is_fitted = False
        self.valid_classes = {}  # Store valid classes for multi-label columns
        
    def _filter_rare_classes(self, data: pd.DataFrame, column: str) -> set:
        """
        Identify classes that appear at least min_frequency times.
        """
        if self.min_frequency <= 1:
            # Get all unique classes from the multi-label column
            all_classes = set()
            for labels in data[column]:
                if isinstance(labels, (list, tuple, set)):
                    all_classes.update(labels)
            return all_classes
        
        # Count occurrences of each class
        class_counts = {}
        for labels in data[column]:
            if isinstance(labels, (list, tuple, set)):
                for label in labels:
                    class_counts[label] = class_counts.get(label, 0) + 1
        
        # Filter classes that meet minimum frequency
        valid_classes = {cls for cls, count in class_counts.items() 
                        if count >= self.min_frequency}
        
        return valid_classes
    
    def _filter_labels(self, labels, valid_classes):
        """
        Filter labels to keep only valid classes.
        Returns empty list (not [None]) if no valid labels remain.
        """
        if isinstance(labels, (list, tuple, set)):
            filtered = [label for label in labels if label in valid_classes]
            return filtered
        return labels

    
    def fit(self, data: pd.DataFrame) -> 'FeatureExpander':
        """
        Fit encoders on training data.
        """
        # Fit multi-label encoders with rare class filtering
        for column in self.multi_label_cols:
            if column in data.columns:
                # Filter rare classes
                self.valid_classes[column] = self._filter_rare_classes(data, column)
                
                # Filter data to only valid classes
                filtered_data = data[column].apply(
                    lambda x: self._filter_labels(x, self.valid_classes[column])
                )
                
                self.multis[column].fit(filtered_data)
        
        # Fit one-hot encoders
        for column in self.one_hot_cols:
            if column in data.columns:
                self.ones[column].fit(data[[column]])
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Transform data using fitted encoders.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform
        inplace : bool
            Whether to modify data inplace (default=False for safety)
        """
        if not self.is_fitted:
            raise ValueError("FeatureExpander must be fit before transform. Call fit() first.")
        
        if not inplace:
            data = data.copy()
        
        # Transform multi-label columns
        for column in self.multi_label_cols:
            if column in data.columns:
                # Filter to only valid classes from training
                filtered_data = data[column].apply(
                    lambda x: self._filter_labels(x, self.valid_classes[column])
                )
                
                transformed = self.multis[column].transform(filtered_data)
                col_names = [f"{column}__{cls}" for cls in self.multis[column].classes_]
                
                # Use uint8 for binary features (memory optimization)
                df_transformed = pd.DataFrame(
                    transformed.astype(np.uint8), 
                    columns=col_names, 
                    index=data.index
                )
                
                data = pd.concat([data, df_transformed], axis=1)
                data.drop(columns=[column], inplace=True)
        
        # Transform one-hot columns
        for column in self.one_hot_cols:
            if column in data.columns:
                transformed = self.ones[column].transform(data[[column]])
                
                # Get feature names
                col_names = self.ones[column].get_feature_names_out([column])
                
                df_transformed = pd.DataFrame(
                    transformed.astype(np.uint8),
                    columns=col_names,
                    index=data.index
                )
                
                data = pd.concat([data, df_transformed], axis=1)
                data.drop(columns=[column], inplace=True)
        
        return data
    
    def fit_transform(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Fit encoders and transform data in one step.
        """
        self.fit(data)
        return self.transform(data, inplace=inplace)
    
    def get_feature_names(self) -> dict:
        """
        Get all feature names created by the encoders.
        """
        feature_names = {}
        
        for column, multi in self.multis.items():
            if hasattr(multi, 'classes_'):
                feature_names[column] = [f"{column}__{cls}" for cls in multi.classes_]
        
        for column, one in self.ones.items():
            if hasattr(one, 'categories_'):
                feature_names[column] = [f"{column}__{cat}" for cat in one.categories_[0]]
        
        return feature_names
