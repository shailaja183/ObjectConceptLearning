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
parser.add_argument('--nq', default=None, type=int, required=False, help='number_of_questions')
parser.add_argument('--output_dir', default="./outputs/", type=str, required=False, help='output folder')

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

if args.nq:
	cfg['number_of_questions'] = args.nq

output_path = os.path.join(args.output_dir, f"{cfg['global_sub']}_questions_{cfg['number_of_questions']}_results.txt")

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
elif cfg['global_sub']=="object":
	from src.datamodules.tiny_imagenet import TinyImageNet
	test_dataset = TinyImageNet('./data/tiny-imagenet', split='val', download=False)
elif cfg['global_sub']=="cifar10":
	from src.datamodules.cifar10 import Cifar10
	test_dataset = Cifar10(train=False, download=True)
elif cfg['global_sub']=="cifar100":
	from src.datamodules.cifar100 import Cifar100
	test_dataset = Cifar100(train=False, download=True, classes = list(data.keys()))
elif cfg['global_sub']=="food101":
	from src.datamodules.food101 import Food101
	test_dataset = Food101(split="test", download=True)
elif cfg['global_sub']=="dtd":
	from src.datamodules.dtd import DTD
	test_dataset = DTD(split="test", download=True)
elif cfg['global_sub']=="oxfordpets":
	from src.datamodules.oxfordpets import OxfordPets
	test_dataset = OxfordPets(split="test", download=True)
else:
	raise NotImplementedError

# read_file = open(os.path.join(cfg['data_path'], "ours", f"conceptdb{cfg['number_of_questions']}.json"), "r")
# conceptload = json.load(read_file)


"""Initialize model
"""
models = []
if cfg['model']=="blip":
	from src.models.models import VILTModel
	models.append(VILTModel(device))
else:
	raise NotImplementedError


total = [0]*len(models)
correct = [0]*len(models)
for en in tqdm(range(len(test_dataset))):
	(input, target) = test_dataset.__getitem__(en)
	qlist = data[target]
	assert(len(qlist)==cfg['number_of_questions'])

	for idx, model in enumerate(models):
		predlabel = []
		for q in qlist:
			y_pred = model.inference((input,q,["Yes","No"]))
			if y_pred=="Yes":
				predlabel.append(True)
			elif y_pred=="No":
				predlabel.append(False)
			# else:
			# 	raise

		if np.mean(predlabel)>=0.5:
			correct[idx]+=1
		total[idx]+=1

with open(output_path, "a") as h:
	for idx, model in enumerate(models):
		print(f"Model:{model.__class__.__name__} Accuracy is: ", correct[idx]*100/total[idx])
		h.write(f"Model:{model.__class__.__name__} Accuracy is: {correct[idx]*100/total[idx]}\n")
