import sagemaker
import boto3
import pandas as pd
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter, HyperparameterTuningJobAnalytics
from sagemaker.amtviz import visualize_tuning_job

import sys

session = sagemaker.Session()
region = session.boto_region_name
role = 'arn:aws:iam::065939300880:role/service-role/AmazonSageMaker-ExecutionRole-20250927T183929'
s3_bucket = 's3://grod-general-purpose/Brain_Tumor_Dataset/'
client = boto3.client('sagemaker')
tuning_jobs = client.list_hyper_parameter_tuning_jobs(MaxResults=10)
# for job in tuning_jobs['HyperParameterTuningJobSummaries']:
#     job_name = job['HyperParameterTuningJobName']
#     print(job)
#     break

# Evaluation Metrics:
# SageMaker will look for output in the log that starts with pre: and is followed
# by one or more whitespace and then a number that we want to extract, which is why
# we use the round parenthesis. Every time SageMaker finds a value like that, it turns
# it into a CloudWatch metric with the name valid-precision.
metric_definitions = [
    {'Name': 'valid-precision',  'Regex': r'pre:\s+(-?[0-9\.]+)'},
    {'Name': 'valid-recall',     'Regex': r'rec:\s+(-?[0-9\.]+)'},
    {'Name': 'valid-f1',         'Regex': r'f1:\s+(-?[0-9\.]+)'},
    {'Name': 'validation-accuracy', 'Regex': r'Validation Accuracy:\s+([0-9\.]+)'}]

hyper_parameters = {
        'epochs': 20,
        'batch_size': 5,
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'dropout_probability': 0.3
    }

# Sagemaker Hyper Parameter Optimization(HPO)
estimator = PyTorch(
    entry_point='train.py',
    # Current Directory
    source_dir='.',
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version='2.1.0',
    py_version='py310',
    metric_definitions=metric_definitions,
    hyperparameters=hyper_parameters
)

# Define hyperparameter ranges
hpt_ranges = {
    'epochs': IntegerParameter(10, 100),
    'batch_size': IntegerParameter(5, 100),
    'learning_rate': ContinuousParameter(0.00001, 0.01),
    'weight_decay': ContinuousParameter(0.0001, 0.1),
    'dropout_probability': ContinuousParameter(0.1, 0.5)
}

tuner_parameters = {
    'estimator': estimator,
    'base_tuning_job_name': 'b-tune',
    'metric_definitions': metric_definitions,
    'objective_metric_name': 'validation-accuracy',
    'objective_type': 'Maximize',
    'hyperparameter_ranges': hpt_ranges,
    'strategy': 'Bayesian',
    'max_jobs': 50,
    'max_parallel_jobs': 2
}

bayesian_tuner = HyperparameterTuner(**tuner_parameters)
inputs = {
    "training": TrainingInput(s3_data=s3_bucket, content_type="image/jpeg")
}
# bayesian_tuner.fit(inputs=inputs, wait=False)

print(f'LATEST: {bayesian_tuner.latest_tuning_job}')# .tuning_job_name
job_name = 'b-tune-251006-1725'
print(f'Job Name: {job_name}')

# job_desc = bayesian_tuner.describe() # Get the job description
visualize_tuning_job(job_name, advanced=True, trials_only=True)
tuner_analysis = HyperparameterTuningJobAnalytics(hyperparameter_tuning_job_name=job_name)
results = tuner_analysis.dataframe()

# Results
print(results['FinalObjectiveValue'])
# Latest Tuning Job
tuner = HyperparameterTuner.attach(job_name)
# Print Latest Tuning Job Description
job_desc = tuner.describe()
import pprint
pprint.pprint(job_desc)