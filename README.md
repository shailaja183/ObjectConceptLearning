# ObjectConceptLearning
Code for Visual Concept Learning using LLM+VQA 

# 1. Dataset Used:

CUB- <https://drive.google.com/drive/folders/1urrMMYQnYGg_u9R3gJVh5_jou2oIwf3Y?usp=sharing>\
CIFAR10 and CIFAR100- <https://www.cs.toronto.edu/~kriz/cifar.html>\
Food101- <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>\
DTD- <https://www.robots.ox.ac.uk/~vgg/data/dtd/>

# 2. LLM generated Concept Descriptions:

- In Conceptdb directory, files are stored in the conceptdb{m}_{dataset_name} format:\
Contains mapping of each object category in a particular dataset to LLM-generated concept descriptions 

- In Questiondb directory, files are stored in questions{m}_{dataset_name} format:\
Contains mapping of each object category in a particular dataset to a set of meta-questions generated form LLM-generated concept descriptions (will be used for evaluation)

- m is an integer 1/3/5 denoting the number of concept descriptions obtained from LLM corresponding to each object category

- Only provided here for the reference (if user would like to view LLM generated concept descriptions); the script works in end-to-end fashion from dataset download, concept descriptions using LLMs, call VQA model and aggregate to predict the final answer

# 3. Experiments

- Zero-shot CUB evaluation and analysis:

1. Download the CUB dataset from <https://drive.google.com/drive/folders/1urrMMYQnYGg_u9R3gJVh5_jou2oIwf3Y?usp=sharing> (Size exceeds GitHub Limitation)

2. Zero-shot_CUB.ipynb - Takes pre-trained version of BLIP, GIT and ViLT models and uses test images from the respective dataset along with corresponding Questiondb file to determine the presence/absence of a particular object category

3. To run the code end-to-end, you need to provide your GPT-3 user api key under "apikey" variable

- Other VQA Dataset evaluation (Experiments directory):

1. pip install -r requirements.txt

2. Run the following commands to individually run the script for the respective dataset, results are stored in the outputs/ directory
To run CIFAR10: python main.py --config="./configs/cifar10.yaml"\
To run CIFAR100: python main.py --config="./configs/cifar100.yaml"\
To run Food101: python main.py --config="./configs/food101.yaml"\
To run DTD: python main.py --config="./configs/dtd.yaml"

3. Run the following commands to run all four datasets\
sh run_all.sh






