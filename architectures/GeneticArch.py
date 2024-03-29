import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

class arch(torch.nn.Module):

    def __init__(self):
        super(arch, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=1) # 25 x 25                                  
        #self.conv2 = torch.nn.Conv2d(32, 8, kernel_size=5, stride=1, padding=2) # 25 x 25                                  
        #self.conv3 = torch.nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2) # 25 x 25                                   
        #self.conv4 = torch.nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=1) # 23 x 23                                   
        #self.bn = torch.nn.BatchNorm2d(num_features=8)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1) # 8 x 12 x 12                                   
        #self.dropout = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(8 * 12 * 12, 1)

    def forward(self, x):
        #Computes the activation of the first convolution                                                                  
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 8 * 12 * 12)
        #Computes the activation of the first fully connected layer                                                        
        x = F.relu(self.fc1(x))
        return(x)

def init_weights(m):
   if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.00)

def random_population(num_parents):
    parents = []
    for _ in range(num_parents):
        parent = arch()
        parent.cuda()
        for param in parent.parameters():
            param.requires_grad = False
        init_weights(parent)
        parents.append(parent)
    return parents

def mutate(parent):
    child = copy.deepcopy(parent)
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child.parameters():
        if(len(param.shape)==4): #weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()

        elif(len(param.shape)==2): #weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1]+= mutation_power * np.random.randn()

        elif(len(param.shape)==1): #biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0]+=mutation_power * np.random.randn()
    return child

def mutate2(parent, rate):
    child = copy.deepcopy(parent)
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    param_size = count_parameters(parent)
    genes_to_mutate = []
    for m in np.arange(int(param_size * 0.1)):
       genes_to_mutate.append(np.random.randint(param_size))
    genes_to_mutate = sorted(genes_to_mutate)
    index = 0
    for p in child.parameters():
       for gene in genes_to_mutate[index:]:
         if gene < np.prod(p.size()):
           loc = np.unravel_index(gene, p.size())
           p[loc]+= mutation_power * np.random.randn()
           index +=1
    return child

def return_children(parent, num_child, rate):
    children = []
    for i in range(num_child):
        children.append(mutate2(parent, rate))
    return children

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
   num_parents = 1
   num_child = 4
   mutation_rate = 0.1
   parents = random_population(num_parents)
   param_size = count_parameters(parents[0])
   num_mutations = int(param_size * 0.1)
   
   for i, parent in enumerate(parents):
      print("parent {} state_dict:".format(i))
      for param_tensor in parent.state_dict():
         print(param_tensor, "\t", parent.state_dict()[param_tensor].size())
      children = return_children(parent, num_child, mutation_rate)
      for j, child in enumerate(children):
          print("child {} state_dict:".format(j))
          for param_tensor in child.state_dict():
             print(param_tensor, "\t", child.state_dict()[param_tensor].size())

if __name__ == "__main__":
   main()    
