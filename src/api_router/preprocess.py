# Pipeline for numerical columns with missing values
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import joblib
# Define Preprocessor class
class Preprocessor:
    def __init__(self, num_rest_attribs, num_missvalue_attribs, cat_nominal_attribs, cat_binary_attribs):
        self.num_rest_attribs = num_rest_attribs  # Numerical columns without missing values
        self.num_missvalue_attribs = num_missvalue_attribs  # Numerical columns with missing values
        self.cat_nominal_attribs = cat_nominal_attribs  # Categorical columns (OneHotEncoder)
        self.cat_binary_attribs = cat_binary_attribs  # Binary categorical columns (OrdinalEncoder)
        self.prep_pipeline = None

    def fit_transform(self, X):
        # Define pipeline for numerical columns with missing values
        num_imputer_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median", missing_values=0)),  # Replace 0 with median
            ('scaler', StandardScaler())  # Standardize numerical data
        ])

        # Define pipeline for numerical columns without missing values
        num_scaler_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        # Define pipeline for nominal categorical columns
        cat_nominal_pipeline = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown="ignore"))
        ])

        # Define pipeline for binary categorical columns
        cat_binary_pipeline = Pipeline([
            ('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        # Combine all transformers into a ColumnTransformer
        combine_transformers = ColumnTransformer(
            transformers=[
                ("num_missvalue", num_imputer_pipeline, self.num_missvalue_attribs),
                ("num_rest", num_scaler_pipeline, self.num_rest_attribs),
                ("cat_nominal", cat_nominal_pipeline, self.cat_nominal_attribs),
                ("cat_binary", cat_binary_pipeline, self.cat_binary_attribs),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        # Build the full pipeline
        self.prep_pipeline = Pipeline(steps=[
            ('preprocessor', combine_transformers)
        ])

        # Fit and transform the data
        transformed_data = self.prep_pipeline.fit_transform(X)
        return transformed_data

    def transform(self, X):
        """Transform the input data using the fitted pipeline."""
        # Transform data
        transformed_data = self.prep_pipeline.transform(X)
        return transformed_data

    def get_feature_names(self):
        """Retrieve feature names after transformation."""
        return list(self.prep_pipeline.named_steps['preprocessor'].get_feature_names_out())