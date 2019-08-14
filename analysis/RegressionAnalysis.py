from __future__ import print_function
seed = 0 # seed for reproducibility
import numpy as np
np.random.seed(seed)
import glob
import time
import torch
#from memory_profiler import profile
import sys
sys.path.insert(0,'../')
# added for reproducibility
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from torchsummary import summary
from architectures.GeneticArch import arch 
from training.GA import GA
from loaders.HDF5torch import HDF5generator # generator for data   
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import setGPU
except:
    pass
import ROOT
import argparse
import utils.common as ut

def get_parser():
  parser = argparse.ArgumentParser(description='GA Params' )
  parser.add_argument('--batch_size', action='store', type=int, default=128, help='batch size per update')
  parser.add_argument('--train_ratio', action='store', default=0.9, help='test train ratio')
  parser.add_argument('--shuffle', action='store', default=False, help='shuffle data')
  parser.add_argument('--datapath', action='store', type=str, default="/bigdata/shared/LCD/NewV1/", help='base path for training data')
  parser.add_argument('--outpath', action='store', type=str, default="results/GA_regression/", help='path for results')
  parser.add_argument('--modelpath', action='store', type=str, default="../results/models/GA_mp_1_model.ckpt", help='path to save model')
  parser.add_argument('--particles', nargs='+', default=['Ele'], help='particles used in training (Types should be written without quotes delimited by spaces)')
  parser.add_argument('--dformat', action='store', type=str, default='channels_first')
  parser.add_argument('--dim', action='store', type=int, default=2, help='Number of data dimensions to use')
  return parser

def analysis():
   parser = get_parser()
   params = parser.parse_args()
   batch_size = params.batch_size
   train_ratio = params.train_ratio
   shuffle = params.shuffle
   datapath = params.datapath
   modelpath = params.modelpath
   outpath = params.outpath
   particles = params.particles
   dim = params.dim
   dformat = params.dformat
   ut.safe_mkdir(outpath)
   input_shape = ut.get_input_shape(dformat, dim)
   model = arch().cuda()
   summary(model, input_size=input_shape)
   model.load_state_dict(torch.load(modelpath))
   print('weights loaded from {}'.format(modelpath))
   train_files, test_files = ut.divide_train_test(datapath, particles, train_ratio)
   test_loader = HDF5generator(test_files, batch_size=batch_size, shuffle=shuffle, num_events=20000)
   for i, data in enumerate(test_loader):
      if i==0:
       model.eval()
       with torch.no_grad():
         energy = data['target'].detach().cpu().numpy()
         outputs = model(data['ECAL']).detach().cpu().numpy()
      else:
         with torch.no_grad():
           outputs = np.concatenate((outputs, model(data['ECAL']).detach().cpu().numpy()))
           energy = np.concatenate((energy, data['target'].detach().cpu().numpy()))
   plot_graph(100 * energy, 100 * outputs, "Primary Energy Regression", 'True Energy [GeV]', 'Predicted Energy [GeV]', outpath + 'regression.pdf')
   error = (energy-outputs)/energy
   plot_graph(100 * energy, error, "Primary Energy Regression", 'True Energy [GeV]', 'Relative error', outpath + 'relative_error.pdf')
   plot_prof(100 * energy, 100 * outputs, "Primary Energy Regression", 'True Energy [GeV]', 'Predicted Energy [GeV]', outpath + 'regression_prof.pdf')
   plot_prof(100 * energy, error, "Primary Energy Regression", 'True Energy [GeV]', 'Relative error', outpath + 'relative_error_prof.pdf')

         
def plot_graph(xarray, yarray, label, xlabel, ylabel, filename):
   c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
   c.SetGrid()
   graph = ROOT.TGraph()
   ut.fill_graph(graph, xarray, yarray) 
   graph.SetTitle("{};{}; {};".format(label, xlabel, ylabel))
   graph.SetMarkerColor(4)
   graph.Draw('ap')
   c.Update()
   c.Print(filename)

def plot_prof(xarray, yarray, label, xlabel, ylabel, filename):
   c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
   c.SetGrid()
   maxprofx = np.amax(xarray)
   maxprofy = np.amax(yarray)
   prof= ROOT.TProfile("prof", "{};{}; {};".format(label, xlabel, ylabel), 100, 0, 1.1 * maxprofx, 0, 1.1 * maxprofy)
   ut.fill_profile(prof, xarray, yarray)
   prof.SetMarkerColor(4)
   prof.Draw()
   c.Update()
   c.Print(filename)


if __name__ == "__main__":
   analysis()
