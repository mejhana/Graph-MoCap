# The KIT text prompts produce several index sequences, but these need to be merged and 
# processed before they can be used to create a graph.

# if there are n common index sequences, then create a split! 

import torch
import numpy as np
from graphv2 import TransGraph, MoGraph
from utils import *
import options as opt

args = opt.get_args_parser()

def traversal(path,trans_graph,motion_graph,labels):
    # segment the path into 60 frame segments
    num_segments = len(path)//60
    # find the sequence of indices that can be docoded to get a motion that follows bezier curve
    index_seq = []
    # motion_index is which motion from motionDict I am picking! 
    motion_index = []
    # sample 60 points from the bezier curve as a segment
    full_motion = []
    
    last_index = 0
    for i in range(num_segments):
        pathSeg = path[i*60:(i+1)*60] # extract 60 samples from bezier curve
        pathSeg = pathSeg - pathSeg[0] # make sure the first point of extracted curve is the origin (to perform loss function)

        label = labels[i]
        print(f"FOR BEZIER SEGMENT {i} with label {label}")

        # N = 4549 different motions (for label 0)
        # of shape Nx15x1 (these are my series of indices)
        sampleIndices = np.load(f"processed_text_index/{label}_sample_indices.npy", allow_pickle=True)
        # of shape Nx1,60,263 (these are the corresponding motions for each sequence of indices )
        motionDict = np.load(f"processed_text_index/{label}_sampled_motions.pickle", allow_pickle=True)

        global_rot = np.zeros(motionDict[0][:, :, 0].shape)
        if i == 0:
            # for the first segment use greedy approach
            losses = []
            for mot_index in range(len(motionDict)): # greedy method: look at all N motions and pick the best one
                motion = motionDict[mot_index]
                global_orient = global_rot + motion[:, :, 0]
                _, r_pos = recover_root_rot_pos(torch.tensor(motion).float())
                r_pos = np.array(r_pos.detach().numpy()).reshape([-1, 3])
                loss_1 = path_loss(global_orient, r_pos, pathSeg)
                losses.append([mot_index, loss_1])
            best_motion_index = min(losses, key=lambda x: x[1])[0]
        else:
            neighbours = trans_graph.graph[last_index]
            favourable_starts = []
            for neigh in neighbours:
                favourable_starts.append(neigh)
            favourable_starts.extend([last_index])
            print(f"the favourable starts are {favourable_starts}")

            if not neighbours:
                # the last index has no neighbours pick a different start node! 
                print(f"Last index- {last_index} has no neighbours, cannot traverse the graph!")
                break

            losses = []
            for start in favourable_starts:
                # if start node is in the motion graph and motion graph's label is same as the current label
                if start in motion_graph.graph :
                    for neigh in motion_graph.graph[start]:
                        edge_label = motion_graph.graph.get_edge_data(start, neigh)[0]["label"]
                        if edge_label != label:
                            continue
                        else:
                            mot_index = motion_graph.graph.get_edge_data(start, neigh)[0]["motion_index"]
                            motion = motionDict[mot_index]
                            global_orient = global_rot + motion[:, :, 0]
                            _, r_pos = recover_root_rot_pos(torch.tensor(motion).float())
                            r_pos = np.array(r_pos.detach().numpy()).reshape([-1, 3])
                            loss_1 = path_loss(global_orient, r_pos, pathSeg)
                            losses.append([mot_index, loss_1])
                else:
                    print(f"Node {start} is not in motion graph! or has no neighbours!")
                    continue
                if len(losses) == 0:
                    print(f"could not find a path!")
                    break
                else:
                    best_motion_index = min(losses, key=lambda x: x[1])[0]

        print(f"best motion index is {best_motion_index}")
        motion_index.append(best_motion_index)
        best_motion = motionDict[best_motion_index]
        full_motion.append(best_motion)
        
        # update global_rot
        global_rot += best_motion[:, :, 0]
        best_index_seq = sampleIndices[best_motion_index]
        index_seq.extend(best_index_seq)
        last_index = best_index_seq[-1] # last index of the first segment

        # DEBUG:
        print(f"indices -- > {sampleIndices[best_motion_index]}")
        # print best loss
        if len(losses) != 0:
            print(f"best loss is {min(losses, key=lambda x: x[1])[1]}")
        
        global_orient = best_motion[:, :, 0]
        _, r_pos = recover_root_rot_pos(torch.tensor(best_motion).float())
        r_pos = np.array(r_pos.detach().numpy()).reshape([-1, 3])
        draw_path(global_orient, r_pos, pathSeg, f"images/{i}.png")

    # lets draw the full path! 
    full_motion = np.array(full_motion).reshape([1, -1, 263])
    _, r_pos = recover_root_rot_pos(torch.tensor(full_motion).float())
    r_pos = np.array(r_pos.detach().numpy()).reshape([-1, 3])
    global_orient = np.array(full_motion[:, :, 0])
    draw_path(global_orient,  r_pos, path, f"images/full_path.png")
    return index_seq, motion_index, full_motion
        
if __name__ == "__main__":

    # sample indices 
    label = args.label
    threshold = args.threshold
    alpha = args.alpha
    beta = args.beta
    start_node = args.start_node
    mot_seq_size = args.mot_seq_size

    path = get_path("paths/new_samples.npy")
    # swap y and z axis
    path[:,[1,2]] = path[:,[2,1]]
    # start from origin
    path = path - path[0]
    print(f"Path length: {len(path)}")

    num_segments = len(path)//60
    print(f"Number of segments: {num_segments}")
    
    # create an array called labels of size num_segments with entires randomly selected from [0,1,2]
    # labels = np.random.randint(0,3,num_segments)
    labels = np.zeros(num_segments).astype(int)
    print(f"Labels: {labels}")

    # # create a new transition graph
    # weights = findWeights()
    # tr = create_trans_graph(weights, threshold)
    # plot_graph(tr.graph, filename='results/trans_graph.png', ignore_weights=False)

    # # create a new motion graph
    # for label in range(3):
    #     index_seq = np.load(f"processed_text_index/{label}_sample_indices.npy", allow_pickle=True)
    #     if label == 0:
    #         # create a new motion graph
    #         mg = create_mograph(index_seq, label=label)
    #     else:
    #         # create a new motion graph
    #         mg = create_mograph(index_seq, existgraph=mg.graph, label=label)

    # plot_graph(mg.graph, filename='results/motion_graph.png', ignore_weights=True)

    # lets load the graph
    tr = load_graph('graphs/trans_graph.pickle', graph_name=TransGraph)
    mg = load_graph('graphs/motion_graph.pickle', graph_name=MoGraph)

    # DEBUG TRANSITION GRAPH! 
    # find the neighbour with minimum weight for a given node
    # print(tr.bestNeighbor(165)) # 165 is hand together
    # print(tr.bestNeighbor(134)) # 457 is the plank node
    # print(tr.bestNeighbor(467)) # 411 is the plank node
    # print(tr.bestNeighbor(412)) # 411 is the plank node
    # print(tr.bestNeighbor(495)) # 411 is the plank node

    indices, motion_index, motion = traversal(path,tr,mg,labels)

    # lets convert motion to gif
    motion = torch.tensor(np.array(motion).reshape([1, -1, 263])).float()
    print(motion.shape)
    pos = recover_from_ric(motion, 22)

    draw_to_batch(np.array(pos.detach().numpy()), bezierPath=path, last_frame=False, title_batch=["path fit"], outname=["images/pathFit.gif"])

    print(indices)

    





