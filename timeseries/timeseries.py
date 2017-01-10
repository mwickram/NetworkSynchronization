import pandas as pd
from scipy.signal import savgol_filter,hilbert,detrend
import matplotlib.pyplot as plt
import numpy as np
import math


def filter_data(filename):

    data=np.loadtxt(filename)
    data=savgol_filter(data, 31, 2,axis=0)

    return data


def get_DataFrame(data,colIndex):

    dar = 200 #data acquisition rate
    index = np.divide(np.arange(len(data)),dar)
    df = pd.DataFrame(data[:,colIndex],index,columns = ['current'])
    df.index.name = 'time'

    return df


def subplot_design(pltObj,xlabel='Time (s)',ylabel='Current (mA)',xlimit=(10,20),ylimit=(0.05, 0.2),
                   divider=4,label='Figure'):

   # yfloat= 2
    #print(len(str(ylimit[1])))
    #decimal.getcontext().prec = yfloat
    #xfloat=decimal.Decimal(xlimit[1])
    #print(str(xfloat))
    lineDivider= divider
    yticks=np.linspace(ylimit[0],ylimit[1], num=lineDivider)
    xticks=np.linspace(xlimit[0],xlimit[1], num=lineDivider)
    xpos= xlimit[0]+(xlimit[1] - xlimit[0])*0.8
    ypos= yticks[lineDivider-2]+ (ylimit[1]- yticks[lineDivider-2])*0.6
    pltObj.set_xlabel(xlabel)
    pltObj.set_ylabel(ylabel)
    pltObj.set_xlim(xlimit)
    pltObj.set_ylim(ylimit)
    pltObj.set_yticks(yticks)
    #pltObj.yaxis.set_major_formatter(formatStr('%.1f'))
    pltObj.set_xticks(xticks)
    #pltObj.xaxis.set_major_formatter(formatStr('%.1f'))
    pltObj.text(xpos,ypos,label)


def plot_timeSeries(dataFrame,axes,objPosition):

    objrow=objPosition[0]
    objcol=objPosition[1]
    ts=dataFrame['current'].plot(ax=axes[objrow,objcol])
    ts.set_xlabel("")

    return ts

def plot_phase(dataFrame,axes,objPosition,dar=200):

    objrow=objPosition[0]
    objcol=objPosition[1]
    data=find_freq(dataFrame,dar)
    df=data[0]

    ts=df['phase'].plot.line(ax=axes[objrow,objcol],
                             color='k',marker='o',markersize=2)
    ts=df['model'].plot.line(ax=axes[objrow,objcol],color='r')

    return ts


def get_phaseData(dataFrame):

    signal = dataFrame
    mean_current=signal.mean(axis=0)#mean of column
    analytical_signal = hilbert(signal-mean_current,axis=0)#complex
    inst_phase=np.unwrap(np.angle(analytical_signal),axis=0)#real value
    newarray = np.hstack([analytical_signal,inst_phase])

    return newarray


def plot_hilbert(dataFrame,ax):

    #objrow=objPosition[0]
    #objcol=objPosition[1]
    data = get_phaseData(dataFrame)
    a_signal = data[:,0]
    #ax=axes
    ts=ax.plot(np.real(a_signal),np.imag(a_signal))

    #ax[objrow, objcol].plot(a_signal,np.imag(a_signal))

def plot_detrendPhase(dataFrame,axes,objPosition,dar=200):

    objrow=objPosition[0]
    objcol=objPosition[1]
    data=get_phaseData(dataFrame)
    inst_phase=np.real(data[:,1])
    inst_phase = pd.Series(detrend(inst_phase,0))

    ml=len(inst_phase)//20
    ntau = np.arange(200,ml,2)
    phaseVariance=pd.Series()

    for k in range(len(ntau)):
        v = phase_diffusion(inst_phase,ntau[k])
        phaseVariance.set_value(k,v)

    data=[np.divide(ntau,dar), phaseVariance]
    df=pd.DataFrame(np.transpose(data),columns=['time','pv'])
    ts=df.plot(x='time',y='pv',ax=axes[objrow,objcol],legend=False)

    return ts


def phase_diffusion(phase,ntau):

    nl = len(phase)%ntau
    n =  (len(phase) - nl)//ntau
    dp=pd.Series()
    for k in range(n):
        nstart = k*ntau
        nend = (k+1)*ntau-1
        dp.set_value(k,phase.loc[nend] - phase.loc[nstart])
    v = np.var(dp)

    return v


def find_freq(dataFrame,dar=200):

    data = get_phaseData(dataFrame)
    inst_phase = np.real(data[:,1])#do not need to have np.real
    time = np.divide(np.arange(len(inst_phase)),dar)
    df=pd.DataFrame(inst_phase,index=time,columns=['phase'])
    df.index.name = 'time'

    p=np.polyfit(time,df['phase'],1)
    fit=np.polyval(p,time)
    df['model']=fit

    return [df, p[0]]


def get_frequency(dataFrame,dar=200):

    data=find_freq(dataFrame,dar)
    freq = data[1]/2/np.pi

    return freq

