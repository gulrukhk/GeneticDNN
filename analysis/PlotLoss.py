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
    outpath = params.outpath
    safe_mkdir(outpath)
    start = params.start
    stop = params.stop
    loss_ids = params.labels
    mono = params.mono
    add_minmax = params.add_minmax
    
    losses =[] # list for losses

    # load all loss history in a nested list
    for lfile in lossfiles:
      with open(losspath + lfile, 'rb') as f:
         losses.append(pickle.load(f))
    
    # the first loss file keys will be used
    for key in losses[0]:
      
      # if only key is not related to timing and there is some data corressponding to the key
      if (not 'timing' in key) and len(losses[0][key]) > 1:
        loss = []
        epochs = []
        time = [] 
        time_batch = []
        for index, item in enumerate(losses):
          if '_batch' in key:
             loss.append(item[key]) # take all iteration history
          else:
             loss.append(item[key][start:stop]) # take certain number of epochs
          
          # accumulate timing for epochs
          time_list =[]
          if 'timing' in item:
            for j, t in enumerate(item['timing'][start:stop]):
              if j==0:
                time_list.append(t)
              else:
                time_list.append(t + time_list[j-1])
            time.append(time_list)
          time_list =[]
  
          # accumulate timing for iterations/batches
          if 'timing_batch' in item:
            for j, t in enumerate(item['timing_batch']):
              if j==0:
                time_list.append(t)
              else:
                time_list.append(t + time_list[j-1])
            time_batch.append(time_list)
          epochs.append(np.arange(len(item[key])))

        # for batchwise history
        if '_batch' in key:
          PlotLossRoot(loss,  loss_ids, epochs, 'Primary Energy Regression ({})'.format(key), 'iterations', outpath + key + '_loss.pdf', mono=mono, add_min=add_minmax)
          PlotAccuracyRoot(loss, loss_ids, epochs, 'Primary Energy Regression({})'.format(key), 'iterations', outpath + key + '_accuracy.pdf', mono=mono, add_max=add_minmax)
          PlotLossRoot(loss,  loss_ids, time_batch, 'Primary Energy Regression ({})'.format(key), 'time [sec]', outpath + key + '_loss_time.pdf', mono=mono, add_min=add_minmax)
          PlotAccuracyRoot(loss, loss_ids, time_batch, 'Primary Energy Regression({})'.format(key), 'time [sec]', outpath + key + '_accuracy_time.pdf', mono=mono, add_max=add_minmax)
        # for epochwise history
        else:
          PlotLossRoot(loss,  loss_ids, epochs, 'Primary Energy Regression ({})'.format(key), 'epochs', outpath + key + '_loss.pdf', mono=mono, add_min=add_minmax)
          PlotAccuracyRoot(loss, loss_ids, epochs, 'Primary Energy Regression({})'.format(key), 'epochs', outpath + key + '_accuracy.pdf', mono=mono, add_max=add_minmax)
          PlotLossRoot(loss,  loss_ids, time, 'Primary Energy Regression ({})'.format(key), 'time [sec]', outpath + key + '_loss_time.pdf', mono=mono, add_min=add_minmax)
          PlotAccuracyRoot(loss, loss_ids, time, 'Primary Energy Regression({})'.format(key), 'time [sec]', outpath + key + '_accuracy_time.pdf', mono=mono, add_max=add_minmax)

# command line arguments
def get_parser():
    parser = argparse.ArgumentParser(description='Loss plots' )
    parser.add_argument('--losspath', action='store', type=str, default='../results/history/', help='dir for loss history')
    parser.add_argument('--outpath', action='store', type=str, default='results/loss_GA_batch_test/', help='directory for results')
    parser.add_argument('--lossfiles', nargs='+', default=['GA_training_loss_history.pkl', 'GA_training_batch_256_loss_history.pkl'], help='loss history (list of pkl files delimited by spaces)')
    parser.add_argument('--labels', nargs='+', default=['GA '], help='label for training')
    parser.add_argument('--start', type=int, default=0, help='can be used to remove initial epochs')
    parser.add_argument('--stop', type=int, default=50, help='can be used to remove later epochs')
    parser.add_argument('--mono', type=int, default=0, help='if style should also be changed')
    parser.add_argument('--add_minmax', type=int, default=0, help='if include min or max in labels')
    return parser
    
# Plot loss from an array
def PlotLossRoot(loss_list, loss_ids, xaxis, label, xlabel, filename, color=2, style=1, mono=0, add_min=0):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    legend = ROOT.TLegend(.5, .7, .9, .9)
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
      if mono: graph.SetLineStyle(style)
      graph.GetYaxis().SetRangeUser(0, 1.1 * max)
      mg.Add(graph)
      c.Update()
      legend.AddEntry(graph, loss_ids[index],"l")
      if add_min: legend.AddEntry(graph, " (min {:.4f})".format(min), "l")
      color+=1
      if color == 3 or color == 5 or color == 10:
         color+=1
      style+=1
    mg.Draw('AL')
    legend.Draw()
    c.Update()
    c.Print(filename)

# Plot accuracy from an array
def PlotAccuracyRoot(loss_list, loss_ids, xaxis, label, xlabel, filename, color=2, style=1, mono=0, add_max=0):
    c = ROOT.TCanvas('c' ,"" ,200 ,10 ,700 ,500)
    c.SetGrid()
    legend = ROOT.TLegend(.5, .1, .9, .3)
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
      if mono: graph.SetLineStyle(style)
      mg.Add(graph)
      legend.AddEntry(graph, loss_ids[index],"l")
      if add_max: legend.AddEntry(graph, " (min {:.4f})".format(max), "l")
      color+=1
      if color == 3 or color == 5 or color ==10:
        color+=1
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
           
