import os
import json
import torch
import numpy as np
from tqdm import tqdm
import yaml

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default="./configs/dogs.yaml", help='provide the path to the expriment setting')
parser.add_argument('--debugme', default=False, help='debug mode', action="store_true")

args = parser.parse_args()

if args.debugme:
	import debugpy

	strport = 4444
	debugpy.listen(strport)
	print(
		f"waiting for debugger on {strport}. Add the following to your launch.json and start the VSCode debugger with it:"
	)
	print(
		f'{{\n    "name": "Python: Attach",\n    "type": "python",\n    "request": "attach",\n    "connect": {{\n      "host": "localhost",\n      "port": {strport}\n    }}\n }}'
	)

## load yaml file
with open(args.config, 'r') as f:
	cfg = yaml.load(f, Loader=yaml.FullLoader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""Some data processing
"""

f= open(os.path.join(cfg['data_path'], "ours", f"questions{cfg['number_of_questions']}.json"),'r')
data = json.load(f)

conceptlist = list(data.keys())


"""Initialize dataloader
"""
if cfg['global_sub']=="dog":
	from src.datamodules.dogs import  Dogs
	test_dataset = Dogs(cfg['data_path'], train=False, download=False)
else:
	raise NotImplementedError

read_file = open(os.path.join(cfg['data_path'], "ours", f"conceptdb{cfg['number_of_questions']}.json"), "r")
conceptload = json.load(read_file)


"""Initialize model
"""
models = []
if cfg['model']=="blip":
	from src.models.models import BLIPModel
	models.append(BLIPModel(device)) 
else:
	raise NotImplementedError


qlist = []
for _, desc in conceptload.items():
	qlist += desc

print("Length of qlist is: ", len(qlist))

total = [0]*len(models)
correct = [0]*len(models)
for en in tqdm(range(len(test_dataset))):
	(input, target) = test_dataset.__getitem__(en)
	desc= conceptload[target]
	assert(len(desc)==cfg['number_of_questions'])
	
	for idx, model in enumerate(models):
		predlabel = []
		for q in qlist:
			y_pred = model.inference((input,q,["Yes","No"]))
			if y_pred=="Yes":
				predlabel.append(True)
			elif y_pred=="No":
				predlabel.append(False)

		if np.mean(predlabel)>=0.5:
			correct[idx]+=1
		total[idx]+=1
	
for idx, model in enumerate(models):
	print(f"Model {idx} Accuracy is: ", correct[idx]*100/total[idx])
