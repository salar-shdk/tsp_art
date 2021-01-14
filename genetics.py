import cv2
import sys
import random
import math
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

QUALITY = 32     # lower means more nodes
THRESHOLD = 88 # under this value is black, otherwise white


def read_image(path):
    original_image = cv2.imread(path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('',gray_image)
    threshold, black_n_white_image = cv2.threshold(gray_image, THRESHOLD, 255, cv2.THRESH_BINARY)
    cv2.imshow('',black_n_white_image)
    
    return black_n_white_image

def create_graph(image):
    height, width = np.shape(image)
    graph = []
    counter = 0
    for i in range(0, height, QUALITY):
        for j in range(0, width, QUALITY):
            croped_image = image[i:i+QUALITY, j:j+QUALITY]
            if 0 in croped_image :
                node = {
                    'id':counter,
                    'x': np.fabs(j + random.randint(0,QUALITY-1) - width),
                    'y': np.fabs(i + random.randint(0,QUALITY-1) - height),
                }
                counter+=1
                graph.append(node)
    return graph

def special_starter(graph):
    data =[]
    result = []
    for i, node in enumerate(graph):
        for j, next_node in enumerate(graph):
            if i == j : continue            
            data.append([[i,j],math.sqrt(((node['x'] - next_node['x']) **2) + ((node['y'] - next_node['y']) **2))])
    data.sort(key= lambda x : x[1])

    result = [deepcopy(data[0][0]),deepcopy(data[0][0])]
    
    while(len(result[1])<len(graph)):
        for i in data:
            if i[0][0] in result[1] and i[0][1] in result[1]:
                data.remove(i)
                break
            if i[0][0] == result[0][1]:
                result[0][1] = i[0][1]
                result[1].append(i[0][1])
                break
            
    return result[1]


def genetic_calculate_scores(graph,generation):
    
    for child in generation:
        score = 0
        path = child[0]
        for i in range(len(path[1:])):
            current_node = graph[path[i]]
            next_node = graph[path[i+1]]
            score += math.sqrt(((current_node['x'] - next_node['x']) **2) + ((current_node['y'] - next_node['y']) **2))
        child[1] = score


def genetic_generate_child(mother, father, parts):
    new_child = []        
    chunk = len(father)/parts
    not_valid = []
    selected_values =[]
    for i in range(parts):
        
        start = random.randint(int(i * chunk), int((i+1) * chunk)-1)
        end   = random.randint(start + 1,int((i+1) * chunk))  
        selected_values.append([start,father[start:end]])
        not_valid += father[start:end]
    # print(selected_values)
    # print(not_valid)
    # input()
    v_counter = 0
    last_jump = 0
    selected_values.append([len(father),[]])
    for i in range(len(father)):
        if i< last_jump : continue
        if  i < selected_values[v_counter][0]:
            for node in mother:
                if node not in not_valid:
                    new_child.append(node)
                    not_valid.append(node)
                    break
        else:
            new_child += deepcopy(selected_values[v_counter][1])
            last_jump = i + len(selected_values[v_counter][1])
            v_counter+=1
        # print(f'log {i},{new_child[-1]}')
                    
                        
    genetic_mutate(new_child,0.015)

    return [new_child,10]


def genetic_generate_child_simple(mother , father):
    new_child = []
    start = random.randint(0, len(father)-2)
    end   = random.randint(start + 1,len(father)-1)  
    new_child += father[start:end]
    for node in mother :
        if node not in new_child:
            new_child.append(node)
    
    genetic_mutate(new_child,0.015)
    return [new_child,10]

def genetic_new_generation(generation ):
    new_generation = []
    for i in range(len(generation)):
        for j in range(i+1,len(generation)):
            mother = generation[j][0]
            father = generation[i][0]
            new_generation.append(genetic_generate_child_simple(mother,father))
            # print(len(new_generation))
    return new_generation
            

def genetic_mutate(child, mutate_ratio):
    for i, node in enumerate(child):
        r = random.randint(0,len(child)-1)
        next_node = child[r]
        if random.random() < mutate_ratio:
            tmp = node
            child[i] = next_node
            child[r] = tmp
        


def genetic(graph, count):
    population = 100 #int(len(graph) * (len(graph) - 1) / 2 )
    
    generation = [[random.sample(range(len(graph)),len(graph)),10] for i in range(population)]
    generation.append([special_starter(graph),10])

    best = [[],math.inf]
    print('start')

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for i in range(count):
        
        generation = genetic_new_generation(generation)

        # add random pathes
        #generation += [[random.sample(range(len(graph)),len(graph)),10] for i in range(population)]

        genetic_calculate_scores(graph, generation)

        if i > 10 : generation.append(best)

        generation.sort(key= lambda x: x[1])

        generation = generation[:10]
        
        print(np.array(generation)[:,1])
        if generation[0][1] < best [1] : 
            best = generation[0]
            show_anim(best[0],graph,fig)

        print(f'generation {i} - cost : {generation[0][1]}, best : {best[1]}')

        

    return best

def show(tsp_path, graph):

    plt.plot([graph[i]['x'] for i in tsp_path] , [graph[i]['y'] for i in tsp_path])
    plt.draw()
    plt.pause(0.001)

def show_anim(tsp_path, graph, fig):
    plt.cla()   
    plt.plot([graph[i]['x'] for i in tsp_path] , [graph[i]['y'] for i in tsp_path])
    fig.canvas.draw()
    plt.pause(0.001)


def show_dots(graph):

    plt.scatter([i['x'] for i in graph] , [i['y'] for i in graph])
    plt.show()

def main(args):

    image  = read_image(args[0])
    
    graph = create_graph(image)
    show_dots(graph)
    tsp_path = genetic(graph,6000)
    print(tsp_path[1])
    show(tsp_path[0], graph)


if __name__ == "__main__":
    main(sys.argv[1:])