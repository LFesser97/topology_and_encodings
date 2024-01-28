from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
import torch_geometric.transforms as T

# import custom encodings
from torchvision.transforms import Compose
from custom_encodings import ShortestPathGenerator, OneHotEdgeAttr, LocalCurvatureProfile, AltLocalCurvatureProfile

from experiments.graph_classification import Experiment


import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, borf

import pickle
import wget
import zipfile
import os


mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
collab = list(TUDataset(root="data", name="COLLAB"))
reddit = list(TUDataset(root="data", name="REDDIT-BINARY"))

"""
# load peptides dataset from url to the current directory using os and wget
peptides_url = "https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1"
peptides_zip_filepath = os.getcwd()

# Download the zip folder
wget.download(peptides_url, peptides_zip_filepath)

# Unzip the folder
pepties_zip = os.path.join(peptides_zip_filepath, "peptidesfunc.zip")

with zipfile.ZipFile(pepties_zip, 'r') as zip_ref:
    zip_ref.extractall(peptides_zip_filepath)

# load the peptides dataset train.pt
peptides = torch.load(os.path.join(peptides_zip_filepath, "peptidesfunc", "test.pt"))

# load pascal dataset from url to the current directory using os and wget
pascal_url = "https://www.dropbox.com/s/8x722ai272wqwl4/pascalvocsp.zip?dl=1"
pascal_zip_filepath = os.getcwd()

# Download the zip folder
wget.download(pascal_url, pascal_zip_filepath)

# Unzip the folder
pascal_zip = os.path.join(pascal_zip_filepath, "pascalvocsp.zip")

with zipfile.ZipFile(pascal_zip, 'r') as zip_ref:
    zip_ref.extractall(pascal_zip_filepath)

# load the pascal dataset train.pt
with open(os.path.join(pascal_zip_filepath, "voc_superpixels_edge_wt_region_boundary", "test.pickle"), 'rb') as f:
    pascal = pickle.load(f)

# load coco dataset from url to the current directory using os and wget
coco_url = 'https://www.dropbox.com/s/r6ihg1f4pmyjjy0/cocosp.zip?dl=1'
coco_zip_filepath = os.getcwd()

# Download the zip folder
wget.download(coco_url, coco_zip_filepath)

# Unzip the folder
coco_zip = os.path.join(coco_zip_filepath, "cocosp.zip")

with zipfile.ZipFile(coco_zip, 'r') as zip_ref:
    zip_ref.extractall(coco_zip_filepath)

# load the coco dataset train.pt
with open(os.path.join(coco_zip_filepath, "coco_superpixels_edge_wt_region_boundary", "test.pickle"), 'rb') as f:
    coco = pickle.load(f)
"""

# load encoded datasets
# imdb_encoded = torch.load("data/imdb_encoded.pt")
# print("IMDB ENCODED LOADED")

# proteins_encoded = torch.load("data/proteins_encoded.pt")
# print("IMDB ENCODED LOADED")

# datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins, "imdb": imdb}
# datasets = {"mutag": mutag, "enzymes": enzymes, "proteins": proteins}
datasets = {"collab": collab, "reddit": reddit}


for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            n = graph.num_nodes
            graph.x = torch.ones((n,1))

def average_spectral_gap(dataset):
    # computes the average spectral gap out of all graphs in a dataset
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)

def log_to_file(message, filename="results/graph_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

def _convert_lrgb(dataset: torch.Tensor) -> torch.Tensor:
    x = dataset[0]
    edge_attr = dataset[1]
    edge_index = dataset[2]
    y = dataset[3]

    return Data(x = x, edge_index = edge_index, y = y, edge_attr = edge_attr)

default_args = AttrDict({
    "dropout": 0.5,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": True,
    "num_trials": 50,
    "eval_every": 1,
    "rewiring": "borf",
    "num_iterations": 3,
    "patience": 100,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": None,
    "last_layer_fa": False,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : False,
    "encoding" : None
})

hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2}),
    "peptides": AttrDict({"output_dim": 10}),
    "pascal": AttrDict({"output_dim": 20}),
    "coco": AttrDict({"output_dim": 80})
}

results = []
args = default_args
args += get_args_from_input()
if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    args += hyperparams[key]
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    print(f"TESTING: {key} ({args.rewiring} - layer {args.layer_type})")

    if key in ["peptides", "pascal", "coco"]:
        dataset = [_convert_lrgb(datasets[key][i]) for i in range(len(datasets[key]))]

        # only use every 10th element if the key is coco
        if key == "coco":
            dataset = dataset[::10]

    else:
        dataset = datasets[key]
    
    
    # encode the dataset using the given encoding, if args.encoding is not None
    if args.encoding in ["LAPE", "RWPE", "LCP", "LDP", "SUB"]:

        if os.path.exists(f"data/{key}_{args.encoding}.pt"):
            print('ENCODING ALREADY COMPLETED...')
            dataset = torch.load(f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "LCP":
            print('ENCODING STARTED...')
            lcp = LocalCurvatureProfile()
            for i in range(len(dataset)):
                dataset[i] = lcp.compute_orc(dataset[i])
                print(f"Graph {i} of {len(dataset)} encoded with {args.encoding}")
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")

        else:
            print('ENCODING STARTED...')
            org_dataset_len = len(dataset)
            drop_datasets = []
            current_graph = 0

            for i in range(org_dataset_len):
                if args.encoding == "LAPE":
                    num_nodes = dataset[i].num_nodes
                    eigvecs = np.min([num_nodes, 8]) - 2
                    transform = T.AddLaplacianEigenvectorPE(k=eigvecs)

                elif args.encoding == "RWPE":
                    transform = T.AddRandomWalkPE(walk_length=16)

                elif args.encoding == "LDP":
                    transform = T.LocalDegreeProfile()

                elif args.encoding == "SUB":
                    transform = T.RootedRWSubgraph(walk_length=10)

                try:
                    dataset[i] = transform(dataset[i])
                    print(f"Graph {current_graph} of {org_dataset_len} encoded with {args.encoding}")
                    current_graph += 1

                except:
                    print(f"Graph {current_graph} of {org_dataset_len} dropped due to encoding error")
                    drop_datasets.append(i)
                    current_graph += 1

            for i in sorted(drop_datasets, reverse=True):
                dataset.pop(i)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")


    """
    print('REWIRING STARTED...')
    start = time.time()
    with tqdm.tqdm(total=len(dataset)) as pbar:
        if args.rewiring == "fosr":
            for i in range(len(dataset)):
                edge_index, edge_type, _ = fosr.edge_rewire(dataset[i].edge_index.numpy(), num_iterations=args.num_iterations)
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
                pbar.update(1)
        elif args.rewiring == "sdrf_orc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=False, is_undirected=True, curvature='orc')
                pbar.update(1)
        elif args.rewiring == "sdrf_bfc":
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(dataset[i], loops=args.num_iterations, remove_edges=args["sdrf_remove_edges"], 
                        is_undirected=True, curvature='bfc')
                pbar.update(1)
        elif args.rewiring == "borf":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf3(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "barf_3":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf4(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "barf_4":
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf5(dataset[i], 
                        loops=args.num_iterations, 
                        remove_edges=False, 
                        is_undirected=True,
                        batch_add=args.borf_batch_add,
                        batch_remove=args.borf_batch_remove,
                        dataset_name=key,
                        graph_index=i)
                pbar.update(1)
        elif args.rewiring == "digl":
            for i in range(len(dataset)):
                dataset[i].edge_index = digl.rewire(dataset[i], alpha=0.1, eps=0.05)
                m = dataset[i].edge_index.shape[1]
                dataset[i].edge_type = torch.tensor(np.zeros(m, dtype=np.int64))
                pbar.update(1)
    end = time.time()
    rewiring_duration = end - start

    print('REWIRING COMPLETED...')
    """

    # create a dictionary of the graphs in the dataset with the key being the graph index
    graph_dict = {}
    for i in range(len(dataset)):
       graph_dict[i] = []
    print('GRAPH DICTIONARY CREATED...') 

    
    #spectral_gap = average_spectral_gap(dataset)
    print('TRAINING STARTED...')
    start = time.time()
    for trial in range(args.num_trials):
        train_acc, validation_acc, test_acc, energy, dictionary = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
        for name in dictionary:
            graph_dict[name].append(dictionary[name])
    end = time.time()
    run_duration = end - start

    # pickle the graph dictionary in a new file depending on the number of layers 
    if args.num_layers == 4:
        with open(f"results/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle", "wb") as f:
            pickle.dump(graph_dict, f)
            print(f"Graph dictionary for {key} pickled")

    else:
        with open(f"results/{args.num_layers}_layers/{key}_{args.layer_type}_{args.encoding}_graph_dict.pickle", "wb") as f:
            pickle.dump(graph_dict, f)
            print(f"Graph dictionary for {key} pickled")

    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    train_ci = 2 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    log_to_file(f"RESULTS FOR {key} ({args.rewiring}), {args.num_iterations} ITERATIONS:\n")
    log_to_file(f"average acc: {test_mean}\n")
    log_to_file(f"plus/minus:  {test_ci}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "sdrf_remove_edges" : args.sdrf_remove_edges, 
        "alpha": args.alpha,
        "eps": args.eps,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci,
        "last_layer_fa": args.last_layer_fa,
        "run_duration" : run_duration,
    })

    # Log every time a dataset is completed
    df = pd.DataFrame(results)
    with open(f'results/graph_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)