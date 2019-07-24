import ROOT
import numpy as np
from utils.common import safe_mkdir
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
    lossfile = '../results/loss_history_simple.pkl'
    losspath = 'results/loss_plots_simple_dropout/'
    safe_mkdir(losspath)
    with open(lossfile, 'rb') as f:
         x = pickle.load(f)
    train_loss = x['train'][5:]
    test_loss = x['test'][5:]
    color =2
    PlotLossRoot(train_loss, 'Primary Energy Regression ( Train loss)', losspath + 'train_loss.pdf', color=color)
    PlotAccuracyRoot(train_loss, 'Primary Energy Regression ( Train Accuracy)', losspath + 'train_accuracy.pdf', color=color)
    PlotLossRoot(test_loss, 'Primary Energy Regression ( Test loss)', losspath + 'test_loss.pdf', color=color+2)
    PlotAccuracyRoot(test_loss, 'Primary Energy Regression ( Test Accuracy)', losspath + 'test_accuracy.pdf', color=color+2)
    
def PlotLossRoot(loss_list, label, filename, color=2):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    loss_array = np.zeros(len(loss_list))
    x_array = np.zeros(len(loss_list))
    for i, l in enumerate(loss_list):
      loss_array[i] = l
      x_array[i] = i
    graph = ROOT.TGraph(len(loss_list), x_array, loss_array)
    graph.SetTitle(label + "Epochs; Loss;")
    graph.SetLineColor(color)
    graph.GetXaxis
    graph.Draw()
    c.Update()
    c.Print(filename)

def PlotAccuracyRoot(loss_list, label, filename, color=2):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    acc_array = np.zeros(len(loss_list))
    x_array = np.zeros(len(loss_list))
    for i, l in enumerate(loss_list):
      acc_array[i] = 1-l
      x_array[i] = i
    graph = ROOT.TGraph(len(loss_list), x_array, acc_array)
    graph.SetTitle(label + ";Epochs; Accuracy;")
    graph.SetLineColor(color)
    graph.GetXaxis
    graph.Draw()
    c.Update()
    c.Print(filename)

if __name__ == "__main__":
   main()   
           
