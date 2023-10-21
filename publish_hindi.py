from huggingface_hub import login
from datasets import load_dataset
import datasets

datasets.disable_caching()
login(token="<token goes here.>")

dataset = load_dataset("hindi.py")

dataset.push_to_hub("<yourusername>/<datasetname>")
