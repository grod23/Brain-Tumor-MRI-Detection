import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter, HyperparameterTuningJobAnalytics

session = sagemaker.Session()
region = session.boto_region_name
role = 'arn:aws:iam::065939300880:role/service-role/AmazonSageMaker-ExecutionRole-20250927T183929'
s3_bucket = 's3://grod-general-purpose'

# Evaluation Metrics:
# SageMaker will look for output in the log that starts with pre: and is followed
# by one or more whitespace and then a number that we want to extract, which is why
# we use the round parenthesis. Every time SageMaker finds a value like that, it turns
# it into a CloudWatch metric with the name valid-precision.
metric_definitions = [
    {'Name': 'valid-precision',  'Regex': r'pre:\s+(-?[0-9\.]+)'},
    {'Name': 'valid-recall',     'Regex': r'rec:\s+(-?[0-9\.]+)'},
    {'Name': 'valid-f1',         'Regex': r'f1:\s+(-?[0-9\.]+)'}]

hyper_parameters = {
        'epochs': 20,
        'batches': 5,
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'dropout_probability': 0.3
    }

# Sagemaker Hyper Parameter Optimization(HPO)
estimator = PyTorch(
    entry_point='train.py',
    source_dir='BrainTumorDetectionMRI',
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version='1.12.1',
    py_version='py38',
    metric_definitions=metric_definitions,
    hyperparameters=hyper_parameters
)

# No input needed, 'train.py' script handles data loading.
estimator.fit(wait=True)


# Define hyperparameter ranges
hpt_ranges = {
    'epochs': IntegerParameter(10, 100),
    'batches': IntegerParameter(5, 100),
    'learning_rate': ContinuousParameter(0.00001, 0.01),
    'weight_decay': ContinuousParameter(0.0001, 0.1),
    'dropout_probability': ContinuousParameter(0.1, 0.5)
}

tuner_parameters = {
    'estimator': estimator,
    'base_tuning_job_name': 'bayesian-tuning',
    'metric_definitions': metric_definitions,
    'objective_metric_name': 'valid-recall',
    'objective_type': 'maximize',
    'hyperparameter_ranges': hpt_ranges,
    'strategy': 'Bayesian',
    'max_jobs': 50,
    'max_parallel_jobs': 2
}

bayesian_tuner = HyperparameterTuner(**tuner_parameters)
bayesian_tuner.fit({'train.py': s3_bucket}, wait=False)

