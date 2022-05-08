def save ( seaborn_plot , nameFile ):

    fig = seaborn_plot.get_figure ()
    
    fig.savefig ( nameFile + '.png' )
