# DBSCAN
DBSCAN clustering algorithm realisation

## Installation (Windows)

Download the repo and open it in cmd.

Create python virtual environment using venv:
```
$ python -m venv dbscan_venv
```
Activate created environment:
```
$ dbscan_venv\scripts\activate
```
Install python libs from requirements.txt:
```
$ pip install -r requirements.txt
```
Run geneticMinimizer:
```
$(dbscan_venv) geneticMinimier.py
```

## Usage
To get the clusters labels:
```python
clusters_dict = DBSCAN(0.2, 4, euclid, dataset[0], False)
clusters_labels = label_clusters(dataset, clusters_dict)
print(clusters_labels)
```
