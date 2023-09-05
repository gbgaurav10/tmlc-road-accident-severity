
import os, sys 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from accident_severity.logging import logger
import warnings
warnings.filterwarnings("ignore")
from accident_severity.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.preprocessor = None 
        self.transformed_df = None 


    def get_data_transformation(self):
        try:

            # Load the dataset
            df = pd.read_csv(self.config.data_path)

            # Renaming the columns
            col_map={
                'Time': 'time',
                'Day_of_week': 'day_of_week',
                'Age_band_of_driver': 'driver_age',
                'Sex_of_driver': 'driver_sex',
                'Educational_level': 'educational_level',
                'Vehicle_driver_relation': 'vehicle_driver_relation',
                'Driving_experience': 'driving_experience',
                'Type_of_vehicle': 'vehicle_type',
                'Owner_of_vehicle': 'vehicle_owner',
                'Service_year_of_vehicle': 'service_year',
                'Defect_of_vehicle': 'vehicle_defect',
                'Area_accident_occured': 'accident_area',
                'Lanes_or_Medians': 'lanes',
                'Road_allignment': 'road_allignment',
                'Types_of_Junction': 'junction_type',
                'Road_surface_type': 'surface_type',
                'Road_surface_conditions': 'road_surface_conditions',
                'Light_conditions': 'light_condition',
                'Weather_conditions': 'weather_condition',
                'Type_of_collision': 'collision_type',
                'Number_of_vehicles_involved': 'vehicles_involved',
                'Number_of_casualties': 'casualties',
                'Vehicle_movement': 'vehicle_movement',
                'Casualty_class': 'casualty_class',
                'Sex_of_casualty': 'casualty_sex' ,
                'Age_band_of_casualty': 'casualty_age',
                'Casualty_severity': 'casualty_severity',
                'Work_of_casuality': 'casualty_work',
                'Fitness_of_casuality': 'casualty_fitness',
                'Pedestrian_movement': 'pedestrian_movement',
                'Cause_of_accident': 'accident_cause',
                'Accident_severity': 'accident_severity'
            }
            df.rename(columns=col_map, inplace=True)

            # drop some columns
            df.drop(columns=["time"], axis=1, inplace=True) 

            # Define the target_variable
            X = df.drop(columns=["accident_severity"], axis=1)
            y = df["accident_severity"]

            # Map the target variable mannually
            y.replace({'Slight Injury': 0,
               'Serious Injury': 1,
               'Fatal injury': 2}, inplace=True)
            
            # Define numerical and categorical features

            numerical_features = X.select_dtypes(exclude="object").columns
            categorical_features = X.select_dtypes(include="object").columns

            # Define the pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("robustscaler", RobustScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(handle_unknown="error"))
                ]
            )

            logger.info(f"Numerical columns: {numerical_features}")
            logger.info(f"Categorical columns: {categorical_features}")

            # Define the Preprocessor

            preprocessor = ColumnTransformer(
                transformers=[
                    ("mum_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            self.preprocessor = preprocessor # Store the preprocessor for later usage

            # Transform the whole data using the preprocessor
            X_transformed = preprocessor.fit_transform(X)

            # Get the updated column names after ordinal encoding
            column_names = numerical_features.to_list() + categorical_features.to_list()

            # Combine X_transformed and y back into one Dataframe
            self.transformed_df = pd.DataFrame(X_transformed, columns=column_names)
            self.transformed_df["accident_severity"] = y 

            logger.info(f"Data preprocessing completed")

        except Exception as e:
            raise e
        
    
    def train_test_split(self):
        if self.preprocessor is None:
            raise ValueError(f"Preprocessor is not available. Please call get_data_transformation")
        
        # Split the data into train and test sets
        train, test = train_test_split(self.transformed_df)


        # Save the encoded train and test sets in form of csv files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info(f"Spitted the data into train and test set")
        logger.info(f"Shape of train data: ", train.shape)
        logger.info(f"Shape of test data: ", test.shape)
