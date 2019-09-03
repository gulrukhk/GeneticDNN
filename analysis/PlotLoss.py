import ROOT
import numpy as np
from utils.common import safe_mkdir
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse

def main():
    parser = get_parser()
    params = parser.parse_args()

    losspath = params.losspath
    lossfiles = params.lossfiles 
    #lossfiles.append('loss_history_simple_1layer_rmsprop.pkl')
    outpath = params.outpath
    safe_mkdir(outpath)
    start = params.start
    stop = params.stop
    loss_ids = params.labels
    losses =[]
    #loss_ids =['rmsprop']
    for lfile in lossfiles:
      with open(losspath + lfile, 'rb') as f:
         losses.append(pickle.load(f))
    for key in losses[0]:
      if key!='timing' and len(losses[0][key]) > 1:
        loss = []
        time =[]
        epochs = [] 
        for index, item in enumerate(losses):
          loss.append(item[key][start:stop])
          time_list =[]
          if index==0:
            d = 8.1
          else:
            d = 0.0
          if 'timing' in item:
            for j, t in enumerate(item['timing'][start:stop]):
              if j==0:
                time_list.append(t + d)
              else:
                time_list.append(t + d + time_list[j-1])
            time.append(time_list) 
          epochs.append(np.arange(len(item[key])))
        PlotLossRoot(loss,  loss_ids, epochs, 'Primary Energy Regression ({})'.format(key), 'epochs', outpath + key + '_loss.pdf')
        PlotAccuracyRoot(loss, loss_ids, epochs, 'Primary Energy Regression({})'.format(key), 'epochs', outpath + key + '_accuracy.pdf')
        PlotLossRoot(loss,  loss_ids, time, 'Primary Energy Regression ({})'.format(key), 'time [sec]', outpath + key + '_loss_time.pdf')
        PlotAccuracyRoot(loss, loss_ids, time, 'Primary Energy Regression({})'.format(key), 'time [sec]', outpath + key + '_accuracy_time.pdf')

def get_parser():
    parser = argparse.ArgumentParser(description='Loss plots' )
    parser.add_argument('--losspath', action='store', type=str, default='../results/history/', help='dir for loss history')
    parser.add_argument('--outpath', action='store', type=str, default='results/loss_plots_GA_data1_1000events/', help='directory for results')
    parser.add_argument('--lossfiles', nargs='+', default=['GA_loss_history.pkl'], help='loss history (list of pkl files delimited by spaces)')
    parser.add_argument('--labels', nargs='+', default=['GA'], help='label for training')
    parser.add_argument('--start', type=int, default=0, help='can be used to remove initial epochs')
    parser.add_argument('--stop', type=int, default=1000, help='can be used to remove later epochs')
    return parser
    
def PlotLossRoot(loss_list, loss_ids, xaxis, label, xlabel, filename, color=2, style=1):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    legend = ROOT.TLegend(.7, .7, .9, .9)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("{} ;{}; loss;".format(label, xlabel))
    loss_list = equalize(loss_list)
    xaxis = xaxis[:len(loss_list[0])]
    graphs=[]
    for index, item in enumerate(loss_list):
      loss_array = np.zeros(len(item))
      x_array = np.zeros(len(item))
      for i, l in enumerate(item):
        loss_array[i] = l
        x_array[i] = xaxis[index][i]
      min = np.amin(loss_array)
      max = np.amax(loss_array)
      graphs.append(ROOT.TGraph(len(loss_array), x_array, loss_array))
      graph = graphs[index] 
      graph.SetLineColor(color)
      graph.SetLineStyle(style)
      graph.GetYaxis().SetRangeUser(0, 1.1 * max)
      mg.Add(graph)
      c.Update()
      legend.AddEntry(graph, loss_ids[index] + " (min {:.4f})".format(min),"l")
      color+=2
      style+=1
    mg.Draw('AL')
    legend.Draw()
    c.Update()
    c.Print(filename)

def PlotAccuracyRoot(loss_list, loss_ids, xaxis, label, xlabel, filename, color=2, style=1):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    legend = ROOT.TLegend(.7, .1, .9, .3)
    mg = ROOT.TMultiGraph()
    mg.SetTitle("{};{}; accuracy(1-loss);".format(label, xlabel))
    xaxis = xaxis[:len(loss_list[0])]
    graphs=[]
    for index, item in enumerate(loss_list):
      acc_array = np.zeros(len(item))
      x_array = np.zeros(len(item))
      for i, l in enumerate(item):
        acc_array[i] = 1-l
        x_array[i] = xaxis[index][i]
      max = np.amax(acc_array)
      graphs.append( ROOT.TGraph(len(acc_array), x_array, acc_array))
      graph = graphs[index]
      graph.SetLineColor(color)
      graph.SetLineStyle(style)
      mg.Add(graph)
      legend.AddEntry(graph, loss_ids[index] + " (max {:.4f})".format(max),"l")
      color+=2
      style+=1
    mg.Draw('AL')
    legend.Draw()
    c.Update()
    c.Print(filename)

def equalize(list_of_lists):
    for i, l in enumerate(list_of_lists):
      if i ==0:
        length = len(l)
      else:
        if len(l) < length:
         length = len(l)
         l = l[:length]
    return list_of_lists


if __name__ == "__main__":
   main()   
           
