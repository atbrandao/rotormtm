from pickle import load, dump
from rotor_mtm.results import IntegrationResults
from os import listdir, getcwd
from os.path import isfile, join
import plotly.graph_objs as go

def get_results_chain(f,
                slope=False):

    sl_str = ''
    if slope:    
        sl_str = ' slope'
    
    try:
        with open(f'results_NL_Chain{sl_str}_f_{f}.pic', 'rb') as file:
            res = load(file)
        res = IntegrationResults.update_class_object(res)
        
    except:
        f = float(f)
        with open(f'results_NL_Chain{sl_str}_f_{f}.pic', 'rb') as file:
            res = load(file)
        res = IntegrationResults.update_class_object(res)
    
           
    return res

for f in [1000., 2000., 3000., 4000., 
          5000., 6000., 7000., 8000., 9000.]:
    
    print(f)
    for slope in [True, False]:

        sl_str = ''
        if slope:    
            sl_str = ' slope'

        res = get_results_chain(f=f, 
                        slope=slope)
        lin_r = res.linear_system.calc_linear_frf(res.fl,
                                                silent=True)
        

        rig_r = res.rigid_system.calc_linear_frf(res.fl,
                                                silent=True)
        res.linear_results = lin_r
        res.rigid_results = rig_r
        
        filename = f'results_NL_Chain{sl_str}_f_{f}.pic'
        with open(filename, 'wb') as file:
            dump(res, file)