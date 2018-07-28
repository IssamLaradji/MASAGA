import matplotlib
matplotlib.use('Agg')

import math
import torch
from matplotlib import pyplot as plt
import numpy as np
from torch.utils import data
import argparse
import main
from itertools import product
from addons import pretty_plot
import utils as ut
import pandas as pd
import borgy_utils as bu 
import os 
import experiments
import train
from addons import vis


s2s = {"uniform":"U", "lipschitz":"NU"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--exp') 
    parser.add_argument('-m','--mode') 
    parser.add_argument('-b','--borgy', type=int, default=0) 
    parser.add_argument('-r','--reset', default=0, type=int) 
    parser.add_argument('-c','--cut', default=None, type=int) 

    parser.add_argument('-dList','--dList', default=None, type=str, nargs="+") 
    parser.add_argument('-mList','--mList', default=None, type=str, nargs="+") 
    parser.add_argument('-eList','--eList', default=None, type=int, nargs="+") 
    parser.add_argument('-lList','--lList', default=None, nargs="+") 
    parser.add_argument('-sList','--sList', default=None, type=str, nargs="+") 

    args = parser.parse_args()


    dList, mList, eList, lList, sList = experiments.get_experiment(args.exp, args)
    mList.sort()
    sList.sort()
    
    create_plot = False
    results = {}
    for d, m, e, l, s in product(dList, mList, eList, lList, sList):
        history = ut.load_history(d, m, l, e, s, args.reset)


        if args.mode == "train":
            if len(history["loss"])==0  or args.reset:
                if args.borgy:
                    command = bu.get_command(d,m,e,l,s, args.reset)                    
                    bu.run_command(command, force=True)

                else:
                    train.train(dataset_name=d, model_name=m,
                                epochs=e, learning_rate=l, 
                                sampling_method=s, reset=args.reset)                

        if args.mode == "summary":
            if len(history["loss"])==0:
                continue
            results[history["exp_name"]] = history["loss"][-1]
           


        if args.mode == "qualitative":
            if shape is not None:
                img_name ="/mnt/home/issam/Summaries/manSAGA/qualitative/%s.png"%dataset_name

                ut.imsave(img_name, model.x.numpy().reshape(shape))
                print(img_name)


        if args.mode == "plot_best":
            if history["exp_name_no_lr"] in results:
                continue

            results[history["exp_name_no_lr"]] = l
            
            ncols = len(dList)
            nrows = 1
            if create_plot == False:
                pp_main = pretty_plot.PrettyPlot(title="Experiment %s" % 
                                                 history["exp_name"], 
                                            ratio=0.5,
                                            legend_type="line",
                                            yscale="log",
                                            shareRowLabel=False,
                                            figsize=(5*ncols,4*1),
                                            subplots=(nrows, ncols))
                create_plot = True
   
            yx = pd.DataFrame(history["loss"])
            y_vals = np.abs(np.array(yx["loss"]))
            x_vals = yx["epoch"]
                   
            if args.cut is not None:
                y_vals = y_vals[:args.cut]
                x_vals = x_vals[:args.cut]
           
            pp_main.add_yxList(y_vals=y_vals, 
                               x_vals=x_vals, 
                               label=ut.get_plot_label(m, s, l))
    

    if args.mode == "plot_best":
        pp_main.plot(ylabel="$(f(x) - f^*)/|f^*|$ on the %s dataset" % ut.n2d[d], 
                     xlabel="Epochs",
                     yscale="log")


        #pp_main.axList[0].set_ylim(bottom=1e-7)
        vis.vis_figure(pp_main.fig)
        pp_main.fig.suptitle("")

        pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figName = history["path_plot"]
        ut.create_dirs(figName)
        pp_main.fig.savefig(figName, dpi = 600)

        figName = history["path_plot"].replace(".png", ".pdf")
        pp_main.fig.savefig(figName, dpi = 600) 
    else:
        print(results)