#
# LEADER

This is the implementation of the LEADER introduced in this paper [LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty](https://arxiv.org/abs/2209.11422).

If you find it useful in your research, please cite it using :

```
@inproceedings{daneshleader,
  title={LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty},
  author={Danesh, Mohamad H and Cai, Panpan and Hsu, David},
  booktitle={6th Annual Conference on Robot Learning},
  year={2022}
}
```


## Overview
We have used [SUMMIT](https://github.com/AdaCompNUS/summit) as the simulation of driving in real-world. SUMMIT captures the full complexity of real-world, unregulated, densely-crowded urban environments, such as complex road structures and traffic behaviors, and are thus insufficient for testing or training robust driving algorithms. It is a high-fidelity simulator that facilitates the development and testing of crowd-driving algorithm extending CARLA to support the following additional features:

1. _Real-World Maps:_ generates real-world maps from online open sources (e.g. OpenStreetMap) to provide a virtually unlimited source of complex environments. 

2. _Unregulated Behaviors:_ agents may demonstrate variable behavioral types (e.g. demonstrating aggressive or distracted driving) and violate simple rule-based traffic assumptions (e.g. stopping at a stop sign). 

3. _Dense Traffic:_  controllable parameter for the density of heterogeneous agents such as pedestrians, buses, bicycles, and motorcycles.

4. _Realistic Visuals and Sensors:_ extending off CARLA there is support for a rich set of sensors such as cameras, Lidar, depth cameras, semantic segmentation etc. 

### Planner: DESPOT
We used an expert planner in SUMMIT that explicitly reasons about interactions among traffic agents and the uncertainty on human driver intentions and types. The core is a POMDP model conditioned on human hidden states and urban road contexts. The model is solved using an efficient parallel planner, [HyP-DESPOT](https://github.com/AdaCompNUS/HyP-DESPOT). A detailed description of the model can be found in this [paper](https://arxiv.org/abs/1911.04074). 

### Architecture and Components

The repository structure has the following conceptual architecture:

* [**summit connector**](summit_connector): A python package for communicating with SUMMIT. It publishes ROS topics on state and context information.

* [**crowd pomdp planner**](crowd_pomdp_planner): The POMDP planner package. It receives ROS topics from the Summit_connector package and executes the belief tracking and POMDP planning loop.

* [**car hyp despot**](car_hyp_despot): A static library package that implements the context-based POMDP model and the HyP-DESPOT solver. It exposes planning and belief tracking functions to be called in crowd_pomdp_planner.

* [**py scripts**](py_scripts): Where model definition and initialization, model training and testing, and communication with the compiled simulation and planner happens. 


## Installation
### Pre-requisites
* Ubuntu 18.04
* CUDA 10.0
* Python 3.6
* ROS-Melodic
* catkin_tools: catkin_tools is used for code building in this package instead of the default catkin_make that comes with ROS.

### Setup planner
#### Prepare catkin workspace
```shell
cd && mkdir -p catkin_ws/src
cd catkin_ws
catkin config --merge-devel
catkin build
```

#### Clone the repo
```shell
cd ~/catkin_ws/src
git clone https://github.com/modanesh/LEADER.git
mv LEADER/* .
mv LEADER/.git .
mv LEADER/.gitignore .
rm -r LEADER
```

Now all ROS packages should be in `~/catkin_ws/src`.

#### Compile the repository
```shell
cd ~/catkin_ws
catkin config --merge-devel
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

And then run: `source ~/catkin_ws/devel/setup.bash`

#### Set up the SUMMIT simulator
Download the [SUMMIT simulator](https://github.com/AdaCompNUS/summit.git). Compile from source or download a stable release. Install the simulator to `~/summit`.

## Running the code
Launch the planner and start training the agent using the following commands:

```shell
cd ~/catkin_ws/src/py_scripts
./run_training.sh
```

Once the training is done, it will save models in the `~/catkin_ws/src/py_scripts/models`. Then, the following command may test the agent:
```shell
./run_testing.sh
```


## Getting statistics
```shell
cd ~/catkin_ws/src/py_scripts
python statistics.py --folder /path/to/driving_data/joint_pomdp_baseline/ 
```
