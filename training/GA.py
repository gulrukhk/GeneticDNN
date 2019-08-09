import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

class GA():
   def __init__(self, arch, mutation_rate=0.01, mutation_power=0.02):
     #super(GA, self).__init__()
     self.mutation_rate= mutation_rate 
     self.mutation_power = mutation_power
     self.weight_init= torch.nn.init.xavier_uniform_
     self.arch = arch
     self.size = self.count_parameters(arch())
     self.num_parents = 1
     self.num_children = 4
          
   def count_parameters(self, model):
     return sum(p.numel() for p in model.parameters())

   def init_weights(self, m):
     if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
      self.weight_init(m.weight)
      m.bias.data.fill_(0.00)
   
   def random_population(self):
     parents = []
     for _ in range(self.num_parents):
        parent = self.arch()
        parent.cuda()
        for param in parent.parameters():
            param.requires_grad = False
        self.init_weights(parent)
        parents.append(parent)
     return parents

   def mutate(self, parent):
     child = copy.deepcopy(parent)
     param_size = self.count_parameters(parent)
     mutation_size = int(param_size * self.mutation_rate)
     genes_to_mutate = np.random.randint(param_size, size=mutation_size)
     #genes_to_mutate = sorted(genes_to_mutate)
     index = 0
     for p in child.parameters():
       for gene in genes_to_mutate[index:]:
         if gene < np.prod(p.size()):
           loc = np.unravel_index(gene, p.size())
           p[loc]+= self.mutation_power * np.random.randn()
           index +=1
     return child
 
   def return_children(self, parent):
     children = []
     for i in range(self.num_children):
       children.append(self.mutate(parent))
     return children


  
