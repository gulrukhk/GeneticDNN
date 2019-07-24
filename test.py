import torch
import numpy as np
from training.GA import GA
from architectures.GeneticArch import arch
import setGPU

def main():
   ga = GA(model=arch())
   parents = ga.random_population()
   param_size = ga.count_parameters(parents[0])
   num_mutations = int(param_size * 0.1)

   for i, parent in enumerate(parents):
      print("parent {} state_dict:".format(i))
      for param_tensor in parent.state_dict():
         print(param_tensor, "\t", parent.state_dict()[param_tensor].size())
      children = ga.return_children(parent)
      for j, child in enumerate(children):
          print("child {} state_dict:".format(j))
          for param_tensor in child.state_dict():
             print(param_tensor, "\t", child.state_dict()[param_tensor].size())

if __name__ == "__main__":
   main()

