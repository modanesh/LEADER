import os
import zmq
import sys
import time
import glob
import rospy
import torch
import random
import argparse
import numpy as np
from clear_process import *
from argparse import Namespace
from replay import ReplayBuffer
from itertools import zip_longest
from timeout import TimeoutMonitor
from multiprocessing import Process
from models import AttentionNet
from summit_simulator import print_flush, SimulatorAccessories
from crowd_pomdp_planner.srv import ValueService, ValueServiceResponse
from car_hyp_despot.srv import ImportanceDistService, ImportanceDistServiceResponse


def normal_rand_attentions(att_len):
    weights = np.random.normal(0.5, 0.5, att_len)
    weights[weights < 0] = 0
    return weights / sum(weights)


def uniform_rand_attentions(att_len):
    weights = np.random.random(att_len)
    return weights / sum(weights)


def init_case_dirs():
    global subfolder, result_subfolder
    subfolder = config.summit_maploc
    result_subfolder = os.path.join(root_path, 'result', 'joint_pomdp_drive_mode', subfolder)
    mak_dir(result_subfolder)
    mak_dir(result_subfolder + '_debug')


def mak_dir(path):
    if sys.version_info[0] > 2:
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


def get_bag_file_name(run):
    dir = result_subfolder
    file_name = 'pomdp_search_log-' + str(run) + '_pid-' + str(os.getpid()) + '_r-' + str(config.random_seed)
    existing_bags = glob.glob(dir + "*.bag")
    # remove existing bags for the same run
    for bag_name in existing_bags:
        if file_name in bag_name:
            print_flush("[attention_planner_training.py] removing {}".format(bag_name))
            os.remove(bag_name)
    existing_active_bags = glob.glob(dir + "*.bag.active")
    # remove existing bags for the same run
    for active_bag_name in existing_active_bags:
        if file_name in active_bag_name:
            print_flush("[attention_planner_training.py] removing {}".format(active_bag_name))
            os.remove(active_bag_name)
    return os.path.join(dir, file_name)


def monitor_subprocess(queue):
    global monitor_worker
    monitor_worker.feed_queue(queue)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    monitor_worker.daemon = True
    monitor_worker.start()
    if config.verbosity > 0:
        print_flush("[attention_planner_training.py] SubprocessMonitor started")


def launch_ros():
    print_flush("[attention_planner_training.py] Launching ros")
    sys.stdout.flush()
    cmd_args = "roscore -p {}".format(config.ros_port)
    if config.verbosity > 0:
        print_flush(cmd_args)
    ros_proc = subprocess.Popen(cmd_args.split(), env=config.ros_env)

    while check_ros(config.ros_master_url, config.verbosity) is False:
        time.sleep(1)

    if config.verbosity > 0:
        print_flush("[attention_planner_training.py] roscore started")
    return ros_proc


def launch_summit_simulator(cmd_args):
    shell_cmd = './CarlaUE4.sh -opengl -carla-rpc-port={} -carla-streaming-port={}'.format(config.port, config.port + 1)
    if config.verbosity > 0:
        print_flush('')
        print_flush('[attention_planner_training.py] {}'.format(shell_cmd))

    summit_proc = subprocess.Popen(shell_cmd, cwd=os.path.join(os.path.expanduser("~"), "summit"),
                                   env=dict(config.ros_env, DISPLAY=''), shell=True, preexec_fn=os.setsid)

    wait_for(config.max_launch_wait, summit_proc, 'summit')
    global_proc_queue.append((summit_proc, "summit", None))
    time.sleep(4)

    sim_accesories = SimulatorAccessories(cmd_args, config)
    sim_accesories.start()

    # ros connector for summit
    shell_cmd = 'roslaunch summit_connector connector.launch port:=' + \
                str(config.port) + ' pyro_port:=' + str(config.pyro_port) + \
                ' map_location:=' + str(config.summit_maploc) + \
                ' random_seed:=' + str(config.random_seed) + \
                ' num_car:=' + str(cmd_args.num_car) + \
                ' num_bike:=' + str(cmd_args.num_bike) + \
                ' num_ped:=' + str(cmd_args.num_pedestrian)
    shell_cmd = shell_cmd + ' ego_control_mode:=other ego_speed_mode:=vel'
    if config.verbosity > 0:
        print_flush('[attention_planner_training.py] {}'.format(shell_cmd))
    summit_connector_proc = subprocess.Popen(shell_cmd.split(), env=config.ros_env,
                                             cwd=os.path.join(ws_root, "src/summit_connector/launch"))
    wait_for(config.max_launch_wait, summit_connector_proc, '[launch] summit_connector')
    global_proc_queue.append((summit_connector_proc, "summit_connector_proc", None))

    return sim_accesories


def handle_imp_weight(req, socket):
    belief = req.belief
    context = req.context
    socket.send_pyobj(("SEEK_ATTENTION".encode('ascii'), belief, context, replay_buffer))
    is_weight = socket.recv_pyobj()
    p0_to_be_stored_data = np.array(
        list(zip_longest(*np.array((belief, context, is_weight), dtype=object), fillvalue=0))).T
    trajectory_replay_buffer.extend(p0_to_be_stored_data)
    return ImportanceDistServiceResponse(is_weight)


def handle_value(req, socket):
    value = req.value
    socket.send_pyobj(("RECORD_REWARDS".encode('ascii'), value))
    socket.recv_pyobj()
    return ValueServiceResponse(1)


def launch_pomdp_planner(run, socket):
    rospy.init_node('launch_pomdp_planner')
    pomdp_proc, rviz_out = None, None
    launch_file = 'planner.launch'
    if config.debug:
        launch_file = 'planner_debug.launch'

    shell_cmd = 'roslaunch --wait crowd_pomdp_planner ' + launch_file + ' gpu_id:=' + str(config.gpu_id) + ' mode:=' + \
                str(config.drive_mode) + ' summit_port:=' + str(config.port) + ' time_scale:=' + \
                str.format("%.2f" % config.time_scale) + ' map_location:=' + config.summit_maploc
    pomdp_out = open(get_bag_file_name(run) + '.txt', 'w')
    print_flush("=> Search log {}".format(pomdp_out.name))
    if config.verbosity > 0:
        print_flush('[attention_planner_training.py] {}'.format(shell_cmd))
    start_t = time.time()
    try:
        pomdp_proc = subprocess.Popen(shell_cmd.split(), env=config.ros_env, stdout=pomdp_out, stderr=pomdp_out)
        print_flush('[attention_planner_training.py] POMDP planning...')
        monitor_subprocess(global_proc_queue)
        s = rospy.Service('importance_weight_calculation', ImportanceDistService,
                          lambda msg: handle_imp_weight(msg, socket))
        s = rospy.Service('adding_value', ValueService, lambda msg: handle_value(msg, socket))

        pomdp_proc.wait(timeout=int(config.eps_length / config.time_scale))

        print_flush("[attention_planner_training.py] episode successfully ended")
    except subprocess.TimeoutExpired:
        print_flush("[attention_planner_training.py] episode reaches full length {} s".format(
            config.eps_length / config.time_scale))
    finally:
        elapsed_time = time.time() - start_t
        print_flush('[attention_planner_training.py] POMDP planner exited in {} s'.format(elapsed_time))
    return pomdp_proc


def zmqserver_learning(zmq_port):
    # MAX_TRAJECTORY_LENGTH * 2 : because only 2 features out of 3 are fed into the attention generator, thus
    # the MAX_TRAJECTORY_LENGTH should be multiplied by 2
    attention_model = AttentionNet(2 * MAX_FEATURE_LEN).float().to(DEVICE)
    attention_loaded = False
    accumulated_returns = 0
    for file_path in os.listdir("./models"):
        if file_path.__contains__("gen"):
            attention_model.load_state_dict(torch.load(os.path.join("./models", file_path), map_location=DEVICE))
            attention_model.eval()
            attention_loaded = True
            print_flush("ATTENTION MODEL LOADED SUCCESSFULLY")
    assert (attention_loaded), "Not all models have been loaded!"

    if DEVICE is not 'cpu':
        attention_memory = torch.Tensor(np.random.random((1, MEMORY_SIZE))).cuda()
    else:
        attention_memory = torch.Tensor(np.random.random((1, MEMORY_SIZE)))
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % zmq_port)
    print_flush("Running server on port: {}".format(zmq_port))
    while True:
        message = socket.recv_pyobj()
        instruction = message[0].decode("utf-8")
        instruction_data = message[1:]

        if instruction == "Terminate":
            socket.send_pyobj(0)
            socket.close()
            print_flush("SERVER PROCESS TERMINATED!")
            break
        elif instruction == "SEEK_ATTENTION":
            if HANDCRAFTED_ATT:
                is_weights = uniform_rand_attentions(ATTENTION_SIZE)
            else:
                with torch.no_grad():
                    is_weights, attention_memory = attention_model(torch.tensor(np.array(list(
                        zip_longest(*np.array((instruction_data[0], instruction_data[1]), dtype=object),
                                    fillvalue=0))).reshape(1, -1), device=DEVICE).float(), attention_memory)
                    is_weights = is_weights.squeeze(0).cpu().numpy()
            socket.send_pyobj(is_weights)
        elif instruction == "RECORD_REWARDS":
            socket.send_pyobj(0)
            accumulated_returns += instruction_data[0]
            print_flush("Accumulated returns: {}".format(accumulated_returns))


def zmqclient_process_env(zmq_port, run):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%s" % zmq_port)
    pid = os.getpid()
    print_flush("---------> ITERATION: {}".format(run))
    update_global_config(cmd_args)
    global monitor_worker
    monitor_worker = SubprocessMonitor(config.ros_port, config.verbosity)
    outter_timer = TimeoutMonitor(pid, int(config.timeout / config.time_scale), "ego_script_timer", config.verbosity)
    outter_timer.start()
    global_proc_queue.clear()
    init_case_dirs()
    sim_accesories = launch_summit_simulator(cmd_args)
    pomdp_proc = launch_pomdp_planner(run, socket)

    # terminating everything:
    print_flush('[run_data_collection.py] is ending! Clearing ros nodes...')
    kill_ros_nodes(config.ros_pref)
    print_flush('[run_data_collection.py] is ending! Clearing Processes...')
    try:
        monitor_worker.terminate()
    except Exception as e:
        print_flush(e)
    try:
        sim_accesories.terminate()
    except Exception as e:
        print_flush(e)
    print_flush('[run_data_collection.py] is ending! Clearing timer...')
    try:
        outter_timer.terminate()
    except Exception as e:
        print_flush(e)
    print_flush('[run_data_collection.py] is ending! Clearing subprocesses...')
    clear_queue(global_proc_queue)
    print_flush('exit [run_data_collection.py]')

    socket.send_pyobj(("Terminate".encode('ascii'), 0))
    socket.recv_pyobj()


def update_global_config(cmd_args):
    # Update the global configurations according to command line
    print_flush("Parsing config")
    config.verbosity = cmd_args.verb
    config.gpu_id = cmd_args.gpu_id
    config.port = cmd_args.port
    # config.ros_port = config.port + 111
    config.ros_port = config.port + 9311
    config.pyro_port = config.port + 6100
    config.ros_master_url = "http://localhost:{}".format(config.ros_port)
    config.ros_pref = "ROS_MASTER_URI=http://localhost:{} ".format(config.ros_port)
    config.ros_env = os.environ.copy()
    config.ros_env['ROS_MASTER_URI'] = 'http://localhost:{}'.format(config.ros_port)

    config.summit_maploc = random.choice(['meskel_square', 'magic', 'highway'])
    config.random_seed = cmd_args.rands
    config.eps_length = cmd_args.eps_len
    config.time_scale = cmd_args.t_scale
    config.timeout = 11 * 120 * 4
    config.max_launch_wait = 10
    config.make = bool(cmd_args.make)
    config.debug = bool(cmd_args.debug)
    compile_mode = 'Release'
    if config.debug:
        compile_mode = 'Debug'
    if config.make:
        try:
            shell_cmds = ["catkin config --merge-devel", "catkin build --cmake-args -DCMAKE_BUILD_TYPE=" + compile_mode]
            for shell_cmd in shell_cmds:
                print_flush('[attention_planner_training.py] {}'.format(shell_cmd))
                make_proc = subprocess.call(shell_cmd, cwd=ws_root, shell=True)
        except Exception as e:
            print_flush(e)
            exit(12)
        print_flush("[attention_planner_training.py] make done")

    # drive mode: joint-pomdp
    config.drive_mode = 1


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verb', type=int, default=1,
                        help='Verbosity')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for hyp-despot')
    parser.add_argument('--t_scale', type=float, default=1.0,
                        help='Factor for scaling down the time in simulation (to search for longer time)')
    parser.add_argument('--make', type=int, default=1,
                        help='Make the simulator package')
    parser.add_argument('--port', type=int, default=2000,
                        help='Summit port')
    parser.add_argument('--rands', type=int, default=0,
                        help='Random seed in summit simulator')
    parser.add_argument('--eps_len', type=float, default=120.0,
                        help='Length of episodes in terms of seconds')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Whether to use GPU or CPU')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug mode')
    parser.add_argument('--num_car', default='75', type=int,
                        help='Number of cars to spawn')
    parser.add_argument('--num_bike', default='25', type=int,
                        help='Number of bikes to spawn')
    parser.add_argument('--num_pedestrian', default='10', type=int,
                        help='Number of pedestrians to spawn')
    parser.add_argument('--handcraft_attention', action='store_true', default=False,
                        help='Use hand-crafted attention (uniform dist) or not')
    return parser.parse_args()


if __name__ == "__main__":
    REPLAY_MIN = 5
    REPLAY_MAX = 100000
    MEMORY_SIZE = 1024
    MAX_TRAJECTORY_LENGTH = 1200
    MAX_FEATURE_LEN = 14

    trajectory_replay_buffer = []
    replay_buffer = ReplayBuffer(REPLAY_MAX)

    ws_root = os.getcwd()
    ws_root = os.path.dirname(ws_root)
    ws_root = os.path.dirname(ws_root)
    print_flush("workspace root: {}".format(ws_root))

    ATTENTION_SIZE = 10
    config = Namespace()
    cmd_args = parse_cmd_args()
    DEVICE = cmd_args.device
    HANDCRAFTED_ATT = cmd_args.handcraft_attention
    global_proc_queue = []
    root_path = os.path.join(os.path.expanduser("~"), 'driving_data')

    run_number = 0
    env_p = Process(target=zmqclient_process_env, args=(5550, run_number))
    env_p.start()
    learn_p = Process(target=zmqserver_learning, args=(5550,))
    learn_p.start()
