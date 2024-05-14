import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphv2 import TransGraph, MoGraph
import options as opt
from viz_utils import *

args = opt.get_args_parser()

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def mse_loss(input, target):
    assert input.shape == target.shape, "motions must be of the same shape"

    if len(input.shape) == 3:
        sr = np.sum(np.sum(np.array(input - target)**2, axis=2), axis=1)
    else:
        sr = np.sum(np.array(input - target)**2, axis=1)    
    rms = np.sqrt(np.mean(sr, axis=0))
    return rms

def trans_graph_loss(motInd1, motInd2):
    # load the motions
    motion1 = np.load(f"indices/{motInd1}.npy")
    motion2 = np.load(f"indices/{motInd2}.npy")
    
    return np.linalg.norm(motion1 - motion2)

def findWeights():
    # find the weights of all the edges in the graph
    weights = np.zeros((512, 512))
    for i in range(512):
        for j in range(i+1, 512):
            weights[i][j] = trans_graph_loss(i, j)

    min_weight = np.min(np.min(weights))
    max_weight = np.max(np.max(weights))

    # normalize the weights
    weights = (weights - min_weight)/(max_weight - min_weight)

    # invert the weights 
    print(f"weights of the graph are: {weights}")
    return weights

def create_trans_graph(weights, threshold):
    g = TransGraph()
    # loop through all 512 indices
    for i in range(512):
        for j in range(i+1, 512):
            weight = weights[i,j]
            if weight < threshold:
                g.graph.add_edge(i, j, weight=weight)
            elif g.graph.has_edge(i, j):
                g.graph.remove_edge(i, j)

    # save the graph
    with open('graphs/trans_graph.pickle', 'wb') as f:
        pickle.dump(g.graph, f)
    return g

def create_mograph(index_sequences,existgraph=None, label=0):
    if existgraph:
        g = MoGraph(existgraph)
    else:
        g = MoGraph()

    for i ,index_sequence in enumerate(index_sequences):
        index_sequence = np.array(index_sequence).reshape(-1)
        start_node = index_sequence[0]
        end_node = index_sequence[-1]
        # both index sequence is of shape Tx1 and mean_pos is of shape (T-1)x3
        # add egde between start_node and end_node with edge_index and mean_pos as edge attributes
        g.graph.add_edge(start_node,end_node, motion_index = i, label=label)

    # save the graph
    with open('graphs/motion_graph.pickle', 'wb') as f:
        pickle.dump(g.graph, f)
    return g

def load_graph(filename, graph_name=TransGraph):
    # load the graph
    with open(filename, 'rb') as f:
        loaded_G = pickle.load(f)
    
    newG = graph_name(loaded_G)
    return newG
    
def plot_graph(graph, filename="results/graph.png", ignore_weights=False):
    # Plot the graph
    plt.figure(figsize=(12, 8))

    if not ignore_weights:
        edge_weights = {(u, v): f"{graph[u][v]['weight']}" for u, v in graph.edges()}
        nx.draw_networkx(G=graph, pos=nx.random_layout(G=graph), node_color='red', with_labels=True, edge_color=edge_weights.values(),
                      edge_cmap=plt.cm.viridis, edge_vmin=0,edge_vmax=1, width=2)
    else:
        nx.draw_networkx(G=graph, pos=nx.random_layout(G=graph), node_color='red', with_labels=True, width=2)
    
    plt.title("Graph", fontsize=16)
    plt.savefig(filename)

def path_loss(global_orient,travPath, bezierPath):
    global_orient = global_orient[:, 1:]

    # part1: loss between global_orient of motion and bezier path
    # path[x+1]-path[x] ---> [1,2,3,4,...n] - [0, 1,2,3,4 , n-1]
    BezierDirs = bezierPath[1:,[0,2]] - bezierPath[:-1,[0,2]]
    # normalize the bezier directions
    BezierDirs = BezierDirs/np.linalg.norm(BezierDirs, axis=1, keepdims=True)
    angles = np.array(np.arctan(BezierDirs[:,0]/BezierDirs[:,1])).reshape([1, -1])
    angloss = np.sqrt(np.mean((angles - global_orient)**2))

    # part2: loss between travPath and bezierPath
    # travPath and bezierPath are of shape Tx3
    #  perform mean squared error between the two
    # weights = np.arange(1,travPath.shape[0]+1)**2 #(weights =  x^2 curve)
    squared_diff = np.sqrt((travPath[:,0] - bezierPath[:,0]) ** 2 + (travPath[:,2] - bezierPath[:,2]) ** 2)
    mse = np.mean(squared_diff)
    alpha = 0.9
    beta = 1 - alpha
    return alpha*mse + beta*angloss

def draw_path(global_orient, traversedPath, bezierPath, filename):
    # draw the traversed path and the bezier path
    plt.figure(figsize=(12, 8))
    # add scatter plot for traversed path
    plt.scatter(traversedPath[:, 0], traversedPath[:, 2], color="red", label="traversed path")
    # add scatter plot for bezier path
    plt.scatter(bezierPath[:, 0], bezierPath[:, 2], color='blue', label="bezier path")
    
    plt.legend()
    # find the mse loss between the 2 paths 
    mse = path_loss(global_orient, traversedPath, bezierPath)
    plt.title(f"Traversed path vs Bezier path with loss {mse}", fontsize=16)
    # add text 
    plt.savefig(filename)

def get_path(filename="paths/path.npy"):
    path = np.load(filename)

    # ensure the path starts from origin
    # subtract the start position from the path

    # path = path - path[0]
    return path

def get_traversed_anim(motion_dict, motion_indices, path):
    # assume no transition graph! 

    total_motion = []
    for motion_index in motion_indices:
        motion = motion_dict[motion_index]
        total_motion.append(motion)
    total_motion = np.array(total_motion).reshape(1, -1, 263)

    total_motion = torch.tensor(np.array(total_motion)).float()
    pred_xyz = recover_from_ric(total_motion, 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    draw_to_batch(xyz.numpy(), bezierPath=path, title_batch=["traversed_motion"], outname=[f"no_label_traversal.gif"])

# # traversal with path complete! 
# def traverse(start_node, path, trans_graph, motion_graph, label=0):
#     index_list = []
#     motion_index = []
#     traversal_flag = True
#     path_index = 0

#     path_left = len(path) - path_index 
#     while traversal_flag and path_left != 0:
#         # check if start_node is a node in motion graph (directed graph), and is the source node
#         if start_node in motion_graph.graph and len(motion_graph.graph[start_node]) > 0:
#             nextPath = path[path_index:path_index+args.mot_seq_size].reshape([-1,3])
#             # pick the best potential paths based on the motion and path!
#             losses  = []
#             print(f"number of neighbours: {len(motion_graph.graph[start_node])}")
#             for neigh in motion_graph.graph[start_node]:
#                 edge_pos = motion_graph.graph.get_edge_data(start_node, neigh)[0]["mean_pos"].reshape([-1, 3])
#                 if path_index == 0:
#                     T = edge_pos.shape[0]
#                 else:
#                     T = edge_pos.shape[0] - 1
                
#                 if path_left < T:
#                     print(f"Path index overshot! and path left is {path_left}")
#                     path_pos = path[path_index:path_index+path_left].reshape([path_left,3])
#                     edge_pos = edge_pos[:path_left, :]
#                 else:
#                     path_pos = path[path_index:path_index+T].reshape([T,3])
#                     edge_pos = edge_pos[:T, :]

#                 loss_1 = loss(edge_pos, path_pos)
#                 losses.append([neigh, loss_1])
            
#             best_neigh = min(losses, key=lambda x: x[1])[0]
#             edge_index = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["edge_index"]
#             motion_index.append(motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["motion_index"])
#             if path_left < args.mot_seq_size:
#                 edge_index = edge_index[:path_left]
#                 path_left = 0
#                 break

#             if path_index == 0:
#                 T = edge_index.shape[0] - 1
#                 index_list.extend(edge_index)
#                 path_left -= T
#             else:
#                 T = edge_index.shape[0] - 2
#                 index_list.extend(edge_index[1:])
#                 path_left -= T

#             print(f"indices -- > {edge_index}")
#             start_node = best_neigh
#         else:
#             # there is no path in motion edges in which start_node is the source node
#             # get the best neighbour of start node in transition graph
#             # if start_node is not in transition graph, then pick a different start node
#             if not trans_graph.graph.has_node(start_node):
#                 print(f"Start node- {start_node} is not in transition graph, pick a different start node!")
#                 traversal_flag = False
#             neighbors = trans_graph.graph[start_node]
#             if not neighbors:
#                 # the start node has no neighbours pick a different start node! 
#                 print(f"Start node- {start_node} has no neighbours, pick a different start node!")
#                 traversal_flag = False
#             else: 
#                 # arrange the neighbours in ascending order of weights
#                 sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbors[neighbor].get('weight', 0))
#                 start_node = sorted_neighbors[0]
#                 index_list.extend([start_node])
#                 print(f"node {start_node} has been picked for transitioning")

#     return index_list, motion_index

# # traversal without path complete!
# def traverse(start_node, path, trans_graph, motion_graph, label=0):
#     # traversal works but isnt optimal
#     index_list = []
#     motion_index = []
#     traversal_flag = True
#     path_index = 0

#     while traversal_flag and path_index != len(path):
#         # check if start_node is a node in motion graph (directed graph), and is the source node
#         if start_node in motion_graph.graph and len(motion_graph.graph[start_node]) > 0:
#             nextPath = path[path_index:path_index+args.mot_seq_size].reshape([-1,3])
#             # make changes to next path 
#             nextPath = nextPath - nextPath[0]

#             # pick the best potential paths based on the motion and path!
#             losses  = []
#             print(f"number of neighbours: {len(motion_graph.graph[start_node])}")
#             for neigh in motion_graph.graph[start_node]:
#                 edge_pos = motion_graph.graph.get_edge_data(start_node, neigh)[0]["mean_pos"].reshape([-1, 3])
#                 loss_1 = loss(edge_pos, nextPath)
#                 losses.append([neigh, loss_1])

#             best_neigh = min(losses, key=lambda x: x[1])[0]
#             edge_index = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["edge_index"]
#             motion = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["motion_index"]
#             best_edge_pos = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["mean_pos"].reshape([-1, 3])

#             print(f"next path is: {nextPath}")
#             print(f"best edge pos is: {best_edge_pos}")
#             draw_path(best_edge_pos, nextPath, f"images/{motion}.png")
#             path_index += args.mot_seq_size
#             if len(path) - path_index < args.mot_seq_size:
#                 print(f"Path index overshot! and path left is {len(path) - path_index}")
#                 break

#             if path_index == 0:
#                 motion_index.append(motion)
#                 index_list.extend(edge_index)

#             else:
#                 motion_index.append(motion)
#                 index_list.extend(edge_index[1:])

#             print(f"indices -- > {edge_index}")   
#             start_node = best_neigh
#         else:
#             # there is no path in motion edges in which start_node is the source node
#             # get the best neighbour of start node in transition graph
#             # if start_node is not in transition graph, then pick a different start node
#             if not trans_graph.graph.has_node(start_node):
#                 print(f"Start node- {start_node} is not in transition graph, pick a different start node!")
#                 traversal_flag = False
#             neighbors = trans_graph.graph[start_node]
#             if not neighbors:
#                 # the start node has no neighbours pick a different start node! 
#                 print(f"Start node- {start_node} has no neighbours, pick a different start node!")
#                 traversal_flag = False
#             else: 
#                 # arrange the neighbours in ascending order of weights
#                 sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbors[neighbor].get('weight', 0))
#                 start_node = sorted_neighbors[0]
#                 index_list.extend([start_node])
#                 print(f"node {start_node} has been picked for transitioning")

#     return index_list, motion_index
    
# def traverse(start_node, path, trans_graph, motion_graph, label=0):
#     # traversal works but isnt optimal
#     index_list = []
#     motion_index = []
#     traversal_flag = True
#     path_index = 0

#     while traversal_flag and path_index != len(path):
#         # check if start_node is a node in motion graph (directed graph), and is the source node
#         if start_node in motion_graph.graph and len(motion_graph.graph[start_node]) > 0:
#             nextPath = path[path_index:path_index+args.mot_seq_size].reshape([-1,3])
#             # make changes to next path 
#             nextPath = nextPath - nextPath[0]

#             # pick the best potential paths based on the motion and path!
#             losses  = []
#             print(f"number of neighbours: {len(motion_graph.graph[start_node])}")
#             for neigh in motion_graph.graph[start_node]:
#                 edge_pos = motion_graph.graph.get_edge_data(start_node, neigh)[0]["mean_pos"].reshape([-1, 3])
#                 loss_1 = loss(edge_pos, nextPath)
#                 losses.append([neigh, loss_1])

#             best_neigh = min(losses, key=lambda x: x[1])[0]
#             edge_index = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["edge_index"]
#             motion = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["motion_index"]
#             best_edge_pos = motion_graph.graph.get_edge_data(start_node, best_neigh)[0]["mean_pos"].reshape([-1, 3])

#             print(f"next path is: {nextPath}")
#             print(f"best edge pos is: {best_edge_pos}")
#             draw_path(best_edge_pos, nextPath, f"images/{motion}.png")
#             path_index += args.mot_seq_size
#             if len(path) - path_index < args.mot_seq_size:
#                 print(f"Path index overshot! and path left is {len(path) - path_index}")
#                 break

#             if path_index == 0:
#                 motion_index.append(motion)
#                 index_list.extend(edge_index)

#             else:
#                 motion_index.append(motion)
#                 index_list.extend(edge_index[1:])

#             print(f"indices -- > {edge_index}")   
#             start_node = best_neigh
#         else:
#             # there is no path in motion edges in which start_node is the source node
#             # get the best neighbour of start node in transition graph
#             # if start_node is not in transition graph, then pick a different start node
#             if not trans_graph.graph.has_node(start_node):
#                 print(f"Start node- {start_node} is not in transition graph, pick a different start node!")
#                 traversal_flag = False
#             neighbors = trans_graph.graph[start_node]
#             if not neighbors:
#                 # the start node has no neighbours pick a different start node! 
#                 print(f"Start node- {start_node} has no neighbours, pick a different start node!")
#                 traversal_flag = False
#             else: 
#                 # arrange the neighbours in ascending order of weights
#                 sorted_neighbors = sorted(neighbors, key=lambda neighbor: neighbors[neighbor].get('weight', 0))
#                 start_node = sorted_neighbors[0]
#                 index_list.extend([start_node])
#                 print(f"node {start_node} has been picked for transitioning")

#     return index_list, motion_index  

    # # lets draw the full path! (concatenating only x,y positions!)
    # positions = []
    # r_pos = torch.zeros(1, 60, 3) # root positions for 60 frames! 
    # for i, index in enumerate(motion_index):
    #     # lets edit the motion (60 frame motion for 15 indices)
    #     motion = motionDict[index]
    #     motion = torch.tensor(motion).float()

    #     pos = recover_from_ric(motion, 22) # get the positions of each joint i.e., 60x22x3 from 263 dimensional motion

    #     # the last frame location of the root is the first frame location of the next motion
    #     pos[:, :, :, [0,2]] =  pos[:, :, :, [0,2]] + r_pos[:, -1, [0,2]]
    #     _, new_r_pos = recover_root_rot_pos(motion)
    #     r_pos+=new_r_pos
    #     pos = np.array(pos.detach().numpy()).reshape([1, -1, 22, 3])
    #     positions.append(pos)

    # positions = np.array(positions).reshape(1, -1, 22, 3)

    # new_path = positions[: ,:, 0, :].reshape([-1, 3])

    # draw_path(new_path, path, f"images/full_path.png")





    