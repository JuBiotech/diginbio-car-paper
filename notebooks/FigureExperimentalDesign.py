import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pathlib

def ExperimentalDesign():
    """ This function plots the experimental design data for the CAR cultivation
    """
    parent_path= pathlib.Path(__file__).absolute().parent.parent
    curr_path = fr'{parent_path}\data'
    print(curr_path)
    
    exp_raw_data = pandas.read_excel(fr'{curr_path}\FullWellDescription.xlsx')
    exp_data = exp_raw_data.dropna(subset=['iptg', 'glucose'])
    print(exp_data)


    iptg = np.array(exp_data['iptg'])
    glucose =np.array( exp_data['glucose'])
    fig1, ax = plt.subplots(figsize = (8.3, 6))
    ax.plot(iptg, glucose)
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 35)


    ax.set_ylabel('Feed rate, g L$^{-1}$ h$^{-1}$')
    ax.set_xlabel('IPTG, $\mu$M')
    fig1.savefig(fr"{parent_path}\results\ExpDesign.png",  dpi = 600, bbox_inches='tight')
    

if __name__ == "__main__":
    plt.style.use('BIOVT_TUM')
    ExperimentalDesign()
