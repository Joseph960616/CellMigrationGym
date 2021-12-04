import numpy as np
import os
import matplotlib.pyplot as plt

AI_CELL = 'Cpaaa'
TARGET_CELL = 'ABarpaapp'
RESOLUTION = 0.254

def load_data(start,stop):
    pos_a = []
    radius_a =[]
    cell_name_a = []
    data_dicts = []
    ai_list = []
    target_list = []
    for i in range (start,stop + 1):
        path = './Embryo/utils/nuclei/t%03d-nuclei' % (i)
        pos = []
        radius = []
        cell_name = []
        with open(path) as file:
            for line in file:
                line = line[:len(line)-1]
                vec = line.split(', ')
                if vec[9] == '':
                    continue
                else:
                    id = int(vec[0])
                    pos.append(np.array(((float(vec[5])*RESOLUTION), (float(vec[6])*RESOLUTION), (float(vec[7])))))
                    radius.append(float(vec[8]) / 2)
                    cell_name.append(vec[9])
        pos_a.append(pos)
        radius_a.append(radius)
        cell_name_a.append(cell_name)
        pos_radius = list(zip(pos, radius))
        data_dict = dict(zip(cell_name, pos_radius))
        data_dicts.append(data_dict)
        
    for i in range(len(cell_name_a)):
        if AI_CELL in cell_name_a[i]:
            ai_list.append(i)
        if TARGET_CELL in cell_name_a[i]:
            target_list.append(i)
    if ai_list != []:
        ai_first_appear = ai_list[0]
        target_first_appear = target_list[0]
        ai_last_appear = ai_list[-1]
        target_last_appear = target_list[-1]
    else:
        ai_first_appear = 0
        target_first_appear = 0
    return data_dicts,ai_first_appear,ai_last_appear,target_first_appear,target_last_appear

def Embryo_dis(figname = 'HDRL_Embryo_distance'):
    """
    input:
        figname: file name of the fig wanted to be save
    output:
        Distance_to_target
        Distance to target cell over time figure
    """
    data_dicts,ai_first_appear,ai_last_appear,target_first_appear,target_last_appear = load_data(1,300)
    
    ####Starting and ending stage, starting point and the end points (x,y,z) of these two cells
    print('AI cell \nappear stage: (%s,%s)' % (ai_first_appear,ai_last_appear))
    print('starting location {} \nending location   {}'.format(data_dicts[ai_first_appear][AI_CELL][0],data_dicts[ai_last_appear][AI_CELL][0]))
    print('\nTARGET cell \nappear stage: (%s,%s)'% (target_first_appear,target_last_appear))
    print('starting location {} \nending location   {}'.format(data_dicts[target_first_appear][TARGET_CELL][0],data_dicts[target_last_appear][TARGET_CELL][0]))

    ####Maximum distance using starting stage of AI cell
    pos = []
    for key in data_dicts[target_first_appear].keys():
        pos.append(data_dicts[target_first_appear][key][0])
    distances = []
    for i in range(len(pos)):
        for j in range(len(pos)):
            distances.append(np.linalg.norm(pos[i]-pos[j])) 
    print("\nMaximum distance: %d"%max(distances))
    
    ###Distance to target
    distance_to_target = []
    begin = max(ai_first_appear,target_first_appear)
    end = min(ai_last_appear,target_last_appear)
    for i in range(end-begin):
        distance_to_target.append(np.linalg.norm(data_dicts[begin + i][AI_CELL][0]- \
                                        data_dicts[begin + i][TARGET_CELL][0]))
    ###Draw distance figure
    fig1 = plt.figure(1)
    plt.title("Distance to target cell over time")
    plt.xlabel("Time(s)")
    plt.ylabel("Distance to target cell(um)")
    plt.plot([i for i in range(len(distance_to_target))],distance_to_target)
    plt.savefig('%s.png' % figname)
    plt.show()
    return distance_to_target

if __name__ == '__main__':
    Embryo_dis()