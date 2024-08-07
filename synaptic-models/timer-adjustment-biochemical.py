import numpy as np
import os
import timer_adjustment_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory to store output files")
parser.add_argument("iters", help="number of iterations to run",type=int)
parser.add_argument("weight", help="weight of peaked distribution (between 0 and 1)", type=float)
parser.add_argument("-i", "--itr_report", help="record output every n iterations", type=int, default=1)
parser.add_argument("-r", "--rate_init", help="Initial value of rate parameter", type=float,default=0.01)

def __main__():
    args = parser.parse_args()
    print(args.directory, args.iters, args.weight, args.itr_report)

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)
    times, scale_rates = timer_adjustment_utils.simulateTimerAdjustment(args.iters, args.weight, itr_report = args.itr_report, scale_rate_linear_initial = args.rate_init)

    np.save(args.directory+"/times"+str(args.weight)+".npy", times)
    np.save(args.directory+"/rates"+str(args.weight)+".npy", scale_rates)

if __name__ == "__main__":
    __main__()
