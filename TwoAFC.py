import params
import pickle
import json
import numpy as np
import os
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import norm
import pickle
import scipy.linalg as la
import LMMs
from LMMs.utils import img_pair_generation
from correlation import cal_plcc_srcc_rmse

class TwoAFC_LMM():

    def __init__(self, args):
        self.model_name = args.model_name
        self.stage_name = args.stage_name
        self.json_dir = args.json_dir
        self.data_dir = args.data_dir

        self.round = args.round
        self.eval_interval = args.epochs_per_eval
        self.dataset_path = args.data_dir
        self.save_image = args.save_image
        self.result_root = os.path.join(args.result_dir, args.model_name, args.stage_name)
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)

        self.img_results_root = os.path.join(self.result_root, 'img_results')
        self.textual_results_root = os.path.join(self.result_root, 'textual_results')
        self.performance_results_root = os.path.join(self.result_root, 'performances')
        self.map_results_root = os.path.join(self.result_root, 'map_results')
        if not os.path.exists(self.img_results_root):
            os.makedirs(self.img_results_root)
        if not os.path.exists(self.textual_results_root):
            os.makedirs(self.textual_results_root)
        if not os.path.exists(self.performance_results_root):
            os.makedirs(self.performance_results_root)
        if not os.path.exists(self.map_results_root):
            os.makedirs(self.map_results_root)

        print('Initialize model {}'.format(self.model_name))
        self.model = LMMs.Model(args)
        


    def init_node(self, img_name, mos_mu):
        return {'img_name': img_name, 'mos_mu': mos_mu}


    def optimize_score_map(self, c, seed=0, original_seed=20020):
        np.random.seed(seed)
        num_images = c.shape[0]
        def objective(s):
            sum_log_diff = np.sum(c * np.log(np.maximum(norm.cdf(s[:,None]-s),1e-6)))
            sum_squares =np.sum(s ** 2)/2
            return -(sum_log_diff - sum_squares)
        initial_scores = np.random.rand(num_images)
        constraints ={'type':'eq','fun': lambda s: np.sum(s)}
        result = minimize(objective, initial_scores, constraints=constraints)
        optimized_scores =result.x
        min_score, max_score = np.min(optimized_scores),np.max(optimized_scores)
        scaled_scores = 100 *(optimized_scores - min_score)/(max_score - min_score) # scale to 0-100
        np.random.seed(original_seed)
        return scaled_scores

    def performance(self):
        mos = []
        for data in self.players:
            mos.append(float(self.players[data]['mos_mu']))
        counters = np.nan_to_num(self.counter_matrix.tolist())
        pred = self.optimize_score_map(c=counters)

        return cal_plcc_srcc_rmse(pred, mos)


    def update_function(self, role_package, data_package):

        if data_package['summary']['consistency']:
            data_package['consistent_rate'] += 1 
            data_package['summary']['correction'] = False
            
            # update correction ratio
            if float(self.players[role_package['index1']]['mos_mu']) > float(self.players[role_package['index2']]['mos_mu']):
                if 'CSIQ'  in self.stage_name:  
                    if data_package['summary']['winner'] == 'img2':
                        data_package['correct_rate'] += 1
                        data_package['summary']['correction'] = True  
                else:
                    if data_package['summary']['winner'] == 'img1':
                        data_package['correct_rate'] += 1
                        data_package['summary']['correction'] = True  
            else:
                if 'CSIQ' in self.stage_name:  
                    if data_package['summary']['winner'] == 'img1':
                        data_package['correct_rate'] += 1
                        data_package['summary']['correction'] = True  
                else:
                    if data_package['summary']['winner'] == 'img2':
                        data_package['correct_rate'] += 1
                        data_package['summary']['correction'] = True

            # update TrueSkill scores
            if data_package['summary']['winner'] == 'img1':
                self.counter_matrix[int(role_package['index1']), int(role_package['index2'])] += 1

            if data_package['summary']['winner'] == 'img2':
                self.counter_matrix[int(role_package['index2']), int(role_package['index1'])] += 1

        # update answer_bias
        for type in data_package['summary']['answer_bias_check']:
            data_package['ans_type'][type] += 1

        data_package['summary']['img1_mos'] = self.players[role_package['index1']]['mos_mu']
        data_package['summary']['img2_mos'] = self.players[role_package['index2']]['mos_mu']

        return data_package
    


    def init_datapackge(self):
        return {
            'current_dataset':self.current_dataset,
             'summary':None,
             'consistent_rate':0,
             'correct_rate':0,
             'ans_type':{'first':0,
                         'second':0,
                         'draw':0}
                }


    def run(self):

        start_round = 0
        data_package = self.init_datapackge()
        resume_pair = 0
        self.counter_matrix = np.zeros([self.num_players, self.num_players])
        
        for round_i in range(start_round, self.round):
            
            np.random.seed(1234 + round_i)

            # randomly match 1v1 game
            arr = np.arange(self.num_players)  
            np.random.shuffle(arr) 
            
            # if round_i != (start_round):
            #     resume_pair = 0                   

            for i in range(resume_pair, self.num_players-1, 2):
                
                if i == 0:
                    data_package = self.init_datapackge()

                role_pack = {
                    'index1':str(arr[i]),
                    'index2':str(arr[i+1])
                }

                path1 = os.path.join(self.dataset_path, self.players[str(arr[i])]['img_name'])
                path2 = os.path.join(self.dataset_path, self.players[str(arr[i+1])]['img_name'])

                summary = self.model(path1, path2)

                # update recorded data
                data_package['summary'] = summary
                data_package = self.update_function(role_pack, data_package)

                img1Name = self.players[str(arr[i])]['img_name'].split('.png')[0].replace("/", "_")
                img2Name = self.players[str(arr[i+1])]['img_name'].split('.png')[0].replace("/", "_")
                save_name = img1Name + '_VS_' + img2Name
                
                # Save image results
                if self.save_image:
                    img_pair_generation(self.img_results_root, 
                                            path1, 
                                            path2, 
                                            summary['answer1'], 
                                            summary['answer2'], 
                                            save_name)
       
                # Save textual results as JSON
                result_log = os.path.join(self.textual_results_root, save_name + '.json')
                with open(result_log, 'w') as f:
                    json.dump(summary, f, indent=4)

                if (i+2)//2 == self.num_players//2:
                    data_package['pair'] = 0
                    data_package['round'] = (round_i + 1) if (round_i + 1) < self.round else 0
                else:
                    data_package['pair'] = (i+2)//2
                    data_package['round'] = round_i 

                del data_package['summary']
                # self.AutoRun_save(data_package)

                print('[{}] Round [{}/{}], Pair [{}/{}]'.format(self.current_dataset, (round_i), self.round, i//2, self.num_players//2))

                tk_log = os.path.join(self.map_results_root, '{}_Round_{}.pkl'.format(self.current_dataset, round_i))
                with open(tk_log, "wb") as file:
                    pickle.dump(self.counter_matrix, file)

                
            if (round_i + 1) % self.eval_interval == 0:
                consistent_rate = data_package['consistent_rate']/(self.num_players//2)
                correct_rate = data_package['correct_rate']/(self.num_players//2)
                if self.num_players % 2 != 0:
                    tmp = self.num_players - 1
                else:
                    tmp = self.num_players
                first_frq = data_package['ans_type']['first']/tmp
                second_frq = data_package['ans_type']['second']/tmp
                draw_frq = data_package['ans_type']['draw']/tmp

                plcc, srcc, _ = self.performance()
                # plcc, srcc = 0, 0
                print('Round {},  SRCC: {:.3f},  PLCC: {:.3f}'.format(round_i, srcc, plcc))
                print('Consistency ratio:{:.3f}, Correction ratio:{:.3f}, first_frq:{:.3f}, second_frq:{:.3f}, draw_frq:{:.3f}'.format(
                    consistent_rate, correct_rate, first_frq, second_frq, draw_frq))
                
                _performance = {
                    'Round': round_i,  
                    'SRCC': srcc,  
                    'PLCC': plcc,
                    'Consistency ratio':consistent_rate,
                    'Correction ratio':correct_rate, 
                    'first_frq':first_frq,
                    'second_frq':second_frq,
                    'draw_frq':draw_frq
                    }
                
                performance_log = os.path.join(self.performance_results_root, '{}_Round_{}.json'.format(self.current_dataset, round_i))
                with open(performance_log, 'w') as f:
                    json.dump(_performance, f, indent=4)



    def run_by_dataset(self):
        label_file = os.path.join(self.json_dir, self.stage_name) + '.json'
        with open(label_file, 'r') as file:
            labels = json.load(file)   


        for dataset in labels:
            self.players = {}
            index = 0
            for data in labels[dataset]:
                self.players[str(index)] = self.init_node(data['img_path'], data['gt_score'])
                index += 1
            self.num_players = index
            self.current_dataset = dataset

            print('Done, LVC-{} has {} players (images)'.format(dataset, index))

            self.run()


if __name__ == '__main__':

    args = params.parse_config()

    LVLM_rank = TwoAFC_LMM(args)
    # LVLM_rank.run()
    # Define the size of the matrix
    n = 1000  # Change this to your desired matrix size

    # Generate the n x n matrix with random values in [0, 1]
    random_matrix = np.random.rand(n, n)
    
    print(LVLM_rank.optimize_score_map(random_matrix))