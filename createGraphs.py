import bilstmTrain
import torch
import matplotlib.pyplot as plt
import sys

#flavors = ['a', 'b', 'c', 'd']
#tagging = ['ner', 'pos']
flavors = ['d']
tagging = ['ner']

def createGraphs():
    acc_data = torch.load('accuracy_graphs_data')
    for tag in tagging:
        plt.figure()
        for flavor in flavors:
            accuracy = acc_data[tag+flavor]
            plt.plot([i for i in range(len(accuracy))], accuracy, label = flavor)
        plt.xlabel("number of samples seen / 500")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(tag + "_accuracy_graph.png")

if __name__ == "__main__": 
    import random
    random.seed(0)

    RUN_PARAMS = bilstmTrain.FAVORITE_RUN_PARAMS

    model_file = sys.argv[1]

    for tag in tagging:
        for flavor in flavors:
            print("Running for " + str(tag) + " " + str(flavor))
            train_file = tag + "/train"
            dev_file = tag + "/dev"

            RUN_PARAMS.update({ 'FLAVOR': flavor, 
                            'TRAIN_FILE': train_file,
                            'DEV_FILE' : dev_file,
                            'TAGGING_TYPE' : tag,
                            'TEST_FILE': None, #test_file,
                            'TEST_O_FILE': None, #test_o_file,
                            'MODEL_FILE': model_file,
                            'SAVE_TO_FILE': True, 
                            'RUN_DEV' : True})
            
            run = bilstmTrain.Run(RUN_PARAMS)
            run.train()
    
    createGraphs()
