
from sklearn.pipeline import Pipeline

import preprocessors as pp

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']


price_pipe = Pipeline(
    [
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=CATEGORICAL_VARS))
    ]
)
