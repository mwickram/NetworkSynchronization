import timeseries as ts
import matplotlib.pyplot as plt

def main():

    #file = 'm092111_1.dat'
    file = 'm112410_1.dat'
    data = ts.filter_data(file)
    dataFrame = ts.get_DataFrame(data,0)

    fq= ts.get_frequency(dataFrame,200)
    #print('Frequency: ',format(f,'.3f'),'Hz')
    print('Frequency: ',"%.3f" % fq,'Hz')

    fig, axes= plt.subplots(nrows=2, ncols=2,figsize=(9,7))

    subplot=[0,0]
    pts=ts.plot_timeSeries(dataFrame,axes,subplot)
    ts.subplot_design(pts,label='Figure 1')

    pts=plt.subplot(222)
    ts.plot_hilbert(dataFrame,pts)
    ts.subplot_design(pts,xlabel='Current (mA)',ylabel='H(I)',
                    xlimit=(-0.06,0.08),ylimit=(-0.08, 0.06),divider=6,label='Figure 2')
    pts.grid()

    subplot=[1,0]
    pts=ts.plot_phase(dataFrame,axes,subplot)
    ts.subplot_design(pts,xlabel='Time (s)',ylabel='Phase (rad)',
                   xlimit=(0,220),ylimit=(-10, 590),label='Figure 3')

    subplot=[1,1]
    pts=ts.plot_detrendPhase(dataFrame,axes,subplot,200)
    ts.subplot_design(pts,xlabel='Time (s)',ylabel='Phase Varience',
                    xlimit=(0,11),ylimit=(-1, 1),divider=5,label='Figure 4')
    plt.show()
    #df=data[0]
    #df['phase'].plot.line(color='k',marker='o',markersize=2)
    #df['model'].plot.line(color='r')

if __name__ == '__main__':
    main()