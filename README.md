# huggingface-sentiment-analysis-to-mlflow

This repo contains a python script that can be used to log the huggingface sentiment-analysis task as a model in MLflow. It can then be registered and available for use by the rest of the MLflow users.

## 1. Running this script to load the model into MLflow
Ensure that MLFLOW_TRACKING_URI is set correctly in your environment. For example, if using InfinStor's MLflow service, MLFLOW_TRACKING_URI must be set to infinstor://infinstor.com/

The conda environment that you are working in must have pytorch, mlflow and transformers installed.

```
$ python ./log_model.py
```

## 2. Serving the model locally:
Note the id of the run that was generated from the run of log_model.py in step (1) above. Suppose it is 0-351e1a1e91334d9ca5cb704b0792d9b3, then the following command serves that model

```
$ mlflow models serve -m runs:/0-351e1a1e91334d9ca5cb704b0792d9b3/model
```

## 3. Input:
Input is a pandas dataframe with a column titled 'text' that has the text for which we are estimating sentiment. This is supplied to the inference process that was started in step (2) above in the pandas-split format. The following is an example of such input text:

```
{"columns":["text"],"data":[["This is meh weather"], ["This is great wuther"]]}
```

## 4. Example run of inference:

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["This is meh weather"], ["This is great wuther"]]}' http://127.0.0.1:5000/invocations
```

## 5. Example output
```
[{"text": "This is meh weather", "label": "NEGATIVE", "score": 0.753343403339386}, {"text": "This is great wuther", "label": "POSITIVE", "score": 0.9984261989593506}]
```

As evident, output is a pandas datafrom with three columns - 'text', 'label' and 'score', where 'text' is the same text that was passed in as input, 'label' is the classification (POSITIVE or NEGATIVE) and 'score' is the confidence score.
