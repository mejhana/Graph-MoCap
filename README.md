<h1 align="center">Q-MoGraph—Quantised Motion Graph</h1>

<h2 align="center">Section 1: Video Results For Introduction</h2>

<h3 align="center"> Motion Sub-actions </h3> 

<h4> The results below show the 4 frame animation when a single index $s_i \in \{0, 1, 2, ... , 511\}$ (in this case $i = 2$ and $i = 5$) are decoded using VQ-VAE's decoder. The VQ-VAE helps quantize the dataset into 4 frames worth of animation that can be concatenated to get a longer and more coherent animation. </h4>

| Taking a step forward with left leg | Taking a step forward with right leg |
| ----------------------------------- | ----------------------------------- |
| ![Taking a step forward with left leg](loopable_results/sub-motion_1.gif) | ![Taking a step forward with right leg](loopable_results/sub-motion_2.gif) |

<h3 align="center"> Sequential Actions </h3> 

The following results showcase the capabilities of the T2M-GPT model. It is worth noting that the transformer in the T2M-GPT model performs poorly on longer text prompts and fails to give the correct ordered set of index sequences, thereby producing incorrect or incomplete animations. The last figure shows how we can simply concatenate the results of two distinct text prompts, i.e., the output of the decoder (produced by the indicies given by transformer)  is concatenated together. Then this concatenated motion is processed such that the global angular velocities are concatenated to find the global orientation, the global linear velocities are concatenated to find the global location in the X-Z plane and the joint locations are calculated appropriately using the kinematic chain. 

| Action 1| Action 2| Combination of Actions using Transformer| Concatenation of Actions |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |----------------------------------- |
| ![Action 1](loopable_results/Intro_1.gif) | ![Action 2](loopable_results/intro_2.gif) | ![Combination of actions](loopable_results/Intro_3.gif) | ![Combination of actions](loopable_results/intro_4.gif) | 

<h3 align="center"> Motion Generated from Text Prompts. </h3> 

Motion generated from a long text prompt using T2M-GPT are shown below. 

|Example 1| Concatenated Example 1 | Example 2 | Concatenated Example 2 |
| ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | 
| ![Motion generated from a text prompt](loopable_results/e3.gif) | ![Motion generated concatenated motions](loopable_results/e4.gif) | ![Motion generated from a text prompt](loopable_results/p3.gif) | ![Motion generated concatenated motions](loopable_results/p4.gif) |

<h2 align="center">Section 2: Video Results For Method</h2>

<h3 align="center"> decoded motion of a sampled index sequence </h3> 

| Randomly sampled index sequence from label 1 (run) | Randomly sampled index sequence from label 2 (special) |
| ----------------------------------- | ----------------------------------- |
| ![run](loopable_results/run.gif) | ![special](loopable_results/special.gif) |

<h2 align="center">Section 3: Video Results For Experiments and Analysis</h2>

<h3 align="center"> Experiments with decoding index sequences </h3> 

| decoding each index at a time | decoding 15 indices at a time | decoding all the indices at once |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![decoding each index at a time](loopable_results/decode_index_by_index.gif) | ![decoding 15 indices at a time](loopable_results/15_at_a_time.gif) | ![decoding all the indices at once](loopable_results/decode_all_at_once.gif) | !

<h3 align="center"> Experiments with Bezier Curve </h3> 

| Path of human walking forward| A human walking forward|
| ----------------------------------- | ----------------------------------- |
| ![path of human walking forward](loopable_results/forward.png) | ![gif of human walking forward](loopable_results/forward.gif) |

| Path of human walking backward| A human walking backward|
| ----------------------------------- | ----------------------------------- |
| ![path of human walking forward](loopable_results/backward.png) | ![gif of human walking forward](loopable_results/backward.gif) |

| Path of human walking left| A human walking left|
| ----------------------------------- | ----------------------------------- |
| ![path of human walking left](loopable_results/left.png) | ![gif of human walking left](loopable_results/left.gif) |

| Path of human walking right| A human walking right|
| ----------------------------------- | ----------------------------------- |
| ![path of human walking right](loopable_results/right.png) | ![gif of human walking right](loopable_results/right.gif) |

| Path of human walking forward and then right| A human walking forward and then right|
| ----------------------------------- | ----------------------------------- |
| ![path of human walking forward and then right](loopable_results/forwardSide.png) | ![gif of human walking forward and then right](loopable_results/forwardSide.gif) |


<h2 align="center">Section 4: Video Results For loopable_results</h2>

A few more examples of the path fit are shown below 

| Path of Example 1| Gif of Example 1|
| ----------------------------------- | ----------------------------------- |
| ![path of example 1](loopable_results/ReE1.png) | ![gif of example 1](loopable_results/ReE1.gif) |

| Path of Example 2| Gif of Example 2|
| ----------------------------------- | ----------------------------------- |
| ![path of example 2](loopable_results/ReE2.png) | ![gif of example 2](loopable_results/ReE2.gif) |

| Path of Example 3| Gif of Example 3|
| ----------------------------------- | ----------------------------------- |
| ![path of example 3](loopable_results/ReE3.png) | ![gif of example 2](loopable_results/ReE3.gif) |