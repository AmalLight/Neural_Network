def save ( plot , nameFile ):

    print ( type ( plot ).__name__ )
    
    plot_class_name , fig = type ( plot ).__name__ , None
    
    if plot_class_name == 'module' :
    
       for i in plot.get_fignums () :

           plot.figure  ( i )
           plot.savefig ( '{}_index_{}.png'.format ( nameFile , i ) )
           
           print ( '{}_index_{}.png'.format ( nameFile , i ) )

       fig = None

    else:  fig = plot.get_figure ()
    
    if fig != None : fig.savefig ( nameFile + '.png' )
    if fig != None : print ( '{}.png'.format ( nameFile ) )
