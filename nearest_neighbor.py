import cv2
import sys
import random
import math
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

QUALITY =   3   # lower means more nodes
THRESHOLD = 88   # under this value is black, otherwise white


def read_image(path):
    original_image = cv2.imread(path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print('close images to continue')
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
            # capacity = ((QUALITY ** 2) - np.count_nonzero(croped_image)) // QUALITY 
            if 0 in croped_image:
                node = {
                    'id':counter,
                    'x': np.fabs(j + random.randint(0,QUALITY-1) ),
                    'y': np.fabs(i + random.randint(0,QUALITY-1) - height),
                }
                counter+=1
                graph.append(node)
    return graph

def nearest_neighbor(graph,count):
    data ={}
    result = []
    for i, node in enumerate(graph):
        semi_data = []
        for j, next_node in enumerate(graph):
            if i == j : continue            
            semi_data.append([[i,j],math.sqrt(((node['x'] - next_node['x']) **2) + ((node['y'] - next_node['y']) **2))])
        semi_data.sort(key= lambda x : x[1])
        data[i] = deepcopy(semi_data)
    
    best_score = math.inf
    
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for i in range(count):
        print(f'round {i}')        
        r = random.randint(0,len(graph)-1)
        rr = random.randint(0,len(graph)-2)
        result = [deepcopy(data[r][rr][0]),deepcopy(data[r][rr][0])]
        current_score = data[r][rr][1]
        while(len(result[1])<len(graph)):
            for i in data[result[0][1]]:            
                if i[0][0] == result[0][1] and i[0][1] not in result[1]:
                    result[0][1] = i[0][1]
                    result[1].append(i[0][1])                    
                    current_score += i[1]
                    break
        if current_score < best_score : 
            best = result[1] 
            best_score = current_score
            show_anim(best,graph,fig)        
                
    return [best,best_score]


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
    tsp_path = nearest_neighbor(graph,666)
    print(tsp_path[1])
    show(tsp_path[0], graph)
    input('enter enter to close')


if __name__ == "__main__":
    main(sys.argv[1:])