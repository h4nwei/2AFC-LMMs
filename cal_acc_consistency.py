import json
import os
import matplotlib.pyplot as plt
import numpy as np
from correlation import cal_plcc_srcc_rmse
import matplotlib
import matplotlib.pylab as pylab
import pickle


def evaluation_metrics(model_name, phase_name, round=12):
    root = os.path.join('./results/', model_name, phase_name, 'performances')
    if phase_name == 'Coarse_grain_mixed':
        dataset = ['CSIQ', 'kadid10k', 'MM21', 'LIVEC',  
                'koniq10k', 'spaq']
    elif phase_name == 'Fine_grain_CSIQ_levels':
        dataset = ['butter_flower.AWGN', 'butter_flower.JPEG', 'butter_flower.jpeg2000', 'butter_flower.fnoise', 'butter_flower.BLUR',
        'native_american.AWGN', 'native_american.JPEG', 'native_american.jpeg2000', 'native_american.fnoise', 'native_american.BLUR',
        'redwood.AWGN', 'redwood.JPEG', 'redwood.jpeg2000', 'redwood.fnoise', 'redwood.BLUR',
        'woman.AWGN', 'woman.JPEG', 'woman.jpeg2000', 'woman.fnoise', 'woman.BLUR'    
        ]
    elif phase_name == 'Fine_grain_CSIQ_types':
        dataset = [
            'butter_flower.1', 'butter_flower.2', 'butter_flower.3', 'butter_flower.4', 'butter_flower.5', 
            'native_american.1', 'native_american.2', 'native_american.3', 'native_american.4', 'native_american.5', 
            'redwood.1', 'redwood.2', 'redwood.3', 'redwood.4', 'redwood.5',  
            'woman.1', 'woman.2', 'woman.3', 'woman.4', 'woman.5'
        ]
    elif phase_name == 'Fine_grain_SPAQ':
        dataset = [
            'interval_1_25', 'interval_25_50', 'interval_50_75', 'interval_75_100'
        ]
    print("-----------------------{}----------------------------".format(phase_name))
    for set in dataset:
        P_c = 0
        P_a = 0
        P_b = 0
        for round_i in range(round):
            performance_file = os.path.join(root, set + '_Round_' + str(round_i) + '.json')
            with open(performance_file, 'r') as file:
                performance = json.load(file) 
            
            P_c += performance['Consistency ratio']
            P_a += performance['Correction ratio']
            first_frq = performance['first_frq']
            second_frq = performance['second_frq']
            P_b += (1 - abs(first_frq - second_frq))

        P_c = P_c / round
        P_a = P_a / round
        if set == 'CSIQ': #dmos
            P_a = P_c - P_a
        P_b = P_b / round
        rho = float(performance['PLCC'])
        
        print('[{}] \t Consistency: {:.3}\t  Accuracy: {:.3}\t Correlation: {:.3}'.format(
            set, P_c, P_a, rho
        ))

    return P_c, P_a, P_b, rho





if __name__ == '__main__':
    # [IDEFICS, ChatGPT, InternLM, mPLUG, Q_instruct, Co_instruct]
   
    
    model_name = 'Co_instruct'
    phase_name = 'Coarse_grain_mixed'  # Coarse_grain_mixed,  Fine_grain_CSIQ_levels, Fine_grain_CSIQ_types, Fine_grain_SPAQ 
    # root = './results/' + model + phase + '/performances'
    evaluation_metrics(model_name, phase_name)

    # root = './results/' + model + '/LVC_Phase_II/'
    # evaluation_metrics_category(root)

    

