import json

data = json.load(open("data/tvqa/tvqa_train_processed.json"))

print(data[0]["vid_name"])