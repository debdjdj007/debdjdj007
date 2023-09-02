import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")
import pickle
import os
import collections
import math
import copy
import torch.multiprocessing as mp
import datetime
import logging
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

logger1 = logging.getLogger(__file__)
logger2 = logging.getLogger(__file__)
logger3 = logging.getLogger(__file__)
logger4 = logging.getLogger(__file__)

#Represents a board for a connect game.
class ConnectBoard():
    #Initializes the board.
    def __init__(self):
        self.init_board = np.zeros([6,7]).astype(str)
        self.init_board[self.init_board == "0.0"] = " "
        self.player = 0
        self.current_board = self.init_board

    #Drops a piece onto the board.
    def drop(self, column):
        if self.current_board[0, column] != " ":
            return "Invalid move"
        else:
            row = 0; pos = " "
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.current_board[row, column]
                row += 1
            if self.player == 0:
                self.current_board[row-2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_board[row-2, column] = "X"
                self.player = 0

    #Determines the winner of the game.
    def winner(self):
        if self.player == 1:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col] == "O" and \
                                self.current_board[row + 2, col] == "O" and self.current_board[row + 3, col] == "O":
                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row, col + 1] == "O" and \
                                self.current_board[row, col + 2] == "O" and self.current_board[row, col + 3] == "O":

                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col + 1] == "O" and \
                                self.current_board[row + 2, col + 2] == "O" and self.current_board[row + 3, col + 3] == "O":
                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "O" and self.current_board[row + 1, col - 1] == "O" and \
                                self.current_board[row + 2, col - 2] == "O" and self.current_board[row + 3, col - 3] == "O"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next
        if self.player == 0:
            for row in range(6):
                for col in range(7):
                    if self.current_board[row, col] != " ":
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col] == "X" and \
                                self.current_board[row + 2, col] == "X" and self.current_board[row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row, col + 1] == "X" and \
                                self.current_board[row, col + 2] == "X" and self.current_board[row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col + 1] == "X" and \
                                self.current_board[row + 2, col + 2] == "X" and self.current_board[row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        try:
                            if self.current_board[row, col] == "X" and self.current_board[row + 1, col - 1] == "X" and \
                                self.current_board[row + 2, col - 2] == "X" and self.current_board[row + 3, col - 3] == "X"\
                                and (col-3) >= 0:
                                return True
                        except IndexError:
                            next

    #Computes possible moves.
    def moves(self):
        acts = []
        for col in range(7):
            if self.current_board[0, col] == " ":
                acts.append(col)
        return acts

#Encodes board data.
def encodeB(ConnectBoard):
    board_state = ConnectBoard.current_board
    encoded = np.zeros([6,7,3]).astype(int)
    encoder_dict = {"O":0, "X":1}
    for row in range(6):
        for col in range(7):
            if board_state[row,col] != " ":
                encoded[row, col, encoder_dict[board_state[row,col]]] = 1
    if ConnectBoard.player == 1:
        encoded[:,:,2] = 1
    return encoded

#Decodes board data.
def decodeB(encoded):
    decoded = np.zeros([6,7]).astype(str)
    decoded[decoded == "0.0"] = " "
    decoder_dict = {0:"O", 1:"X"}
    for row in range(6):
        for col in range(7):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    cboard = ConnectBoard()
    cboard.current_board = decoded
    cboard.player = encoded[0,0,2]
    return cboard

#Represents the data structure for a game board.
class BoardData(Dataset):
    #Initializes board data.
    def __init__(self, dataset):
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]

    #Returns the size of the board.
    def __len__(self):
        return len(self.X)

    #Retrieves an item from the board.
    def __getitem__(self,idx):
        return np.int64(self.X[idx].transpose(2,0,1)), self.y_p[idx], self.y_v[idx]

#Represents a convolutional block for neural network operations.
class ConvolutionalBlock(nn.Module):
    #Initializes the convolutional block.
    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    #Processes input through the block.
    def forward(self, s):
        s = s.view(-1, 3, 6, 7)
        s = F.relu(self.bn1(self.conv1(s)))
        return s

#Represents a residual block commonly used in deep learning.
class ResidualBlock(nn.Module):
    #nitializes the residual block.
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    #Processes input through the block.
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

#Represents the output layer or block of a network.
class OutputBlock(nn.Module):
    #Initializes the output block.
    def __init__(self):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 7)

    #Produces the output for given input.
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 3*6*7)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v

#Represents a network to determine connections or relationships.
class ConnectionNetwork(nn.Module):
    #Initializes the connection network.
    def __init__(self):
        super(ConnectionNetwork, self).__init__()
        self.conv = ConvolutionalBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResidualBlock())
        self.outblock = OutputBlock()

    #Processes input through the network.
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

#Computes the error loss using the Alpha method.
class AlphaErrorLoss(torch.nn.Module):
    #Initializes the error computation method.
    def __init__(self):
        super(AlphaErrorLoss, self).__init__()

    #Computes the error for the given input.
    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy*
                                (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


#Saves data in pickle format (version 1).
def savePkl1(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

#Loads data from pickle format (version 1).
def loadPkl1(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

#Represents a node in a tree for the UCT algorithm.
class UpperConfidenceTreeNode():
    #Initializes the node.
    def __init__(self, game, move, parent=None):
        self.game = game
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_idxes = []

    #Retrieves or sets the visit count for the node.
    @property
    def visit_count(self):
        return self.parent.child_number_visits[self.move]

    @visit_count.setter
    def visit_count(self, value):
        self.parent.child_number_visits[self.move] = value

    #Retrieves or sets the value sum for the node.
    @property
    def value_sum(self):
        return self.parent.child_total_value[self.move]

    @value_sum.setter
    def value_sum(self, value):
        self.parent.child_total_value[self.move] = value

    #Computes the quality of child nodes.
    def child_quality(self):
        return self.child_total_value / (1 + self.child_number_visits)

    #Computes the utility of child nodes.
    def child_utility(self):
        return math.sqrt(self.visit_count) * (
            abs(self.child_priors) / (1 + self.child_number_visits))

    #Determines the best node from the children.
    def best_node(self):
        if self.action_idxes != []:
            bestmove = self.child_quality() + self.child_utility()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_quality() + self.child_utility())
        return bestmove

    #Chooses a leaf node.
    def choose_leaf(self):
        current = self
        while current.is_expanded:
          best_move = current.best_node()
          current = current.add_node_maybe(best_move)
        return current

    #Introduces noise to the node.
    def add_noise(self,action_idxs,child_priors):
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                          dtype=np.float32)+192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    #Grows or expands the node.
    def grow(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.moves(); c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000
        if self.parent.parent == None:
            c_p = self.add_noise(action_idxs,c_p)
        self.child_priors = c_p

    #Decodes a given move.
    def decode_move(self,board,move):
        board.drop(move)
        return board

    #Adds a node conditionally.
    def add_node_maybe(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)
            copy_board = self.decode_move(copy_board,move)
            self.children[move] = UpperConfidenceTreeNode(
              copy_board, move, parent=self)
        return self.children[move]

    #Backs up the node data.
    def backup_data(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.visit_count += 1
            if current.game.player == 1:
                current.value_sum += (1*value_estimate)
            elif current.game.player == 0:
                current.value_sum += (-1*value_estimate)
            current = current.parent

#Represents a placeholder or dummy node in a tree or structure.
class PlaceholderNode(object):
    # Initializes the placeholder node.
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)

#Performs a search using the UCT algorithm.
def uctSearch(game_state, num_reads,net,temp):
    root = UpperConfidenceTreeNode(game_state, move=None, parent=PlaceholderNode())
    for i in range(num_reads):
        leaf = root.choose_leaf()
        encoded_s = encodeB(leaf.game); encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if leaf.game.winner() == True or leaf.game.moves() == []:
            leaf.backup_data(value_estimate); continue
        leaf.grow(child_priors)
        leaf.backup_data(value_estimate)
    return root

#Decodes a game move.
def decodeMove(board,move):
    board.drop(move)
    return board

#Retrieves policies or rules.
def getPol(root, temp=1):
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))


#Plays a game using the MCTS algorithm.
def mctsPlay(connectnet, num_games, start_idx, cpu, args, iteration):
    logger1.info("[CPU: %d]: Starting MCTS self-play..." % cpu)

    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)

    for idxx in tqdm(range(start_idx, num_games + start_idx)):
        logger1.info("[CPU: %d]: Game %d" % (cpu, idxx))
        current_board = ConnectBoard()
        checkmate = False
        dataset = []
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.moves() != []:
            if move_count < 11:
                t = args.temperature_MCTS
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.current_board))
            board_state = copy.deepcopy(encodeB(current_board))
            root = uctSearch(current_board,777,connectnet,t)
            policy = getPol(root, t); print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            current_board = decodeMove(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy))
            dataset.append([board_state,policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (iteration, cpu, idxx), current_board.current_board,current_board.player); print(" ")
            if current_board.winner() == True:
                if current_board.player == 0:
                    value = -1
                elif current_board.player == 1:
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx,data in enumerate(dataset):
            s,p = data
            if idx == 0:
                dataset_p.append([s,p,0])
            else:
                dataset_p.append([s,p,value])
        del dataset
        savePkl1("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)

#Runs the MCTS algorithm.
def runMcts(args, start_idx=0, iteration=0):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = ConnectionNetwork()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()

    if args.MCTS_num_processes > 1:
        logger1.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn",force=True)
        net.share_memory()
        net.eval()

        current_net_filename = os.path.join("./model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger1.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                        net_to_play))
            logger1.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger1.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logger1.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=mctsPlay, args=(net, args.num_games_per_MCTS_process, start_idx, i, args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger1.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logger1.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = os.path.join("./model_data/",\
                                        net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger1.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                        net_to_play))
            logger1.info("Initialized model.")

        with torch.no_grad():
            mctsPlay(net, args.num_games_per_MCTS_process, start_idx, 0, args, iteration)
        logger1.info("Finished MCTS!")


#Saves data in pickle format.
def savePkl(filename, data):
    completeName = os.path.join("./model_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

#Loads data from pickle format.
def loadPkl(filename):
    completeName = os.path.join("./model_data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

#Loads a state or configuration.
def loadSt(net, optimizer, scheduler, args, iteration, new_optim_state=True):
    base_path = "./model_data/"
    checkpoint_path = os.path.join(base_path, "%s_iter%d.pth.tar" % (args.neural_net_name, iteration))
    start_epoch, checkpoint = 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    if checkpoint != None:
        if (len(checkpoint) == 1) or (new_optim_state == True):
            net.load_state_dict(checkpoint['state_dict'])
            logger2.info("Loaded checkpoint model %s." % checkpoint_path)
        else:
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger2.info("Loaded checkpoint model %s, and optimizer, scheduler." % checkpoint_path)
    return start_epoch

#Loads results or outcomes.
def loadRes(iteration):
    losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
    if os.path.isfile(losses_path):
        losses_per_epoch = loadPkl("losses_per_epoch_iter%d.pkl" % iteration)
        logger2.info("Loaded results buffer")
    else:
        losses_per_epoch = []
    return losses_per_epoch

#Trains a model or algorithm.
def train(net, dataset, optimizer, scheduler, start_epoch, cpu, args, iteration):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaErrorLoss()

    train_set = BoardData(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = loadRes(iteration + 1)

    logger2.info("Starting training process...")
    update_size = len(train_loader)//10
    print("Update step size: %d" % update_size)
    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0.0
        losses_per_batch = []
        for i,data in enumerate(train_loader,0):
            state, policy, value = data
            state, policy, value = state.float(), policy.float(), value.float()
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss = loss/args.gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            if (epoch % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if i % update_size == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                print('[Iteration %d] Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (iteration, os.getpid(), epoch + 1, (i + 1)*args.batch_size, len(train_set), losses_per_batch[-1]))
                print("Policy (actual, predicted):",policy[0].argmax().item(),policy_pred[0].argmax().item())
                print("Policy data:", policy[0]); print("Policy pred:", policy_pred[0])
                print("Value (actual, predicted):", value[0].item(), value_pred[0,0].item())
                print(" ")
                total_loss = 0.0

        scheduler.step()
        if len(losses_per_batch) >= 1:
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if (epoch % 2) == 0:
            savePkl("losses_per_epoch_iter%d.pkl" % (iteration + 1), losses_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./model_data/",\
                    "%s_iter%d.pth.tar" % (args.neural_net_name, (iteration + 1))))
        '''

        if len(losses_per_epoch) > 50:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.00017:
                break
        '''
    logger2.info("Finished Training!")
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_iter%d_%s.png" % ((iteration + 1), datetime.datetime.today().strftime("%Y-%m-%d"))))
    plt.show()

#Trains the connection network.
def trainCNet(args, iteration, new_optim_state):
    logger2.info("Loading training data...")
    data_path="./datasets/iter_%d/" % iteration
    datasets = []
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)
    logger2.info("Loaded data from %s." % data_path)

    net = ConnectionNetwork()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
    start_epoch = loadSt(net, optimizer, scheduler, args, iteration, new_optim_state)

    train(net, datasets, optimizer, scheduler, start_epoch, 0, args, iteration)

#Saves data in pickle format (version 2).
def savePkl2(filename, data):
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

#Loads data from pickle format (version 2).
def loadPkl2(filename):
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

#Manages game arenas or play areas.
class ArenaManager():
    #Initializes the arena manager.
    def __init__(self, current_cnet, best_cnet):
        self.current = current_cnet
        self.best = best_cnet

    #Initiates or manages a play session.
    def play(self):
        logger3.info("Starting game round...")
        if np.random.uniform(0,1) <= 0.5:
            white = self.current; black = self.best; w = "current"; b = "best"
        else:
            white = self.best; black = self.current; w = "best"; b = "current"
        current_board = ConnectBoard()
        checkmate = False
        dataset = []
        value = 0; t = 0.1
        while checkmate == False and current_board.moves() != []:
            dataset.append(copy.deepcopy(encodeB(current_board)))
            print(""); print(current_board.current_board)
            if current_board.player == 0:
                root = uctSearch(current_board,777,white,t)
                policy = getPol(root, t); print("Policy: ", policy, "white = %s" %(str(w)))
            elif current_board.player == 1:
                root = uctSearch(current_board,777,black,t)
                policy = getPol(root, t); print("Policy: ", policy, "black = %s" %(str(b)))
            current_board = decodeMove(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy))
            if current_board.winner() == True:
                if current_board.player == 0:
                    value = -1
                elif current_board.player == 1:
                    value = 1
                checkmate = True
        dataset.append(encodeB(current_board))
        if value == -1:
            dataset.append(f"{b} as black wins")
            return b, dataset
        elif value == 1:
            dataset.append(f"{w} as white wins")
            return w, dataset
        else:
            dataset.append("Nobody wins")
            return None, dataset

    #Assesses or evaluates the play.
    def assess(self, num_games, cpu):
        current_wins = 0
        logger3.info("[CPU %d]: Starting games..." % cpu)
        for i in range(num_games):
            with torch.no_grad():
                winner, dataset = self.play(); print("%s wins!" % winner)
            if winner == "current":
                current_wins += 1
            savePkl2("evaluate_net_dataset_cpu%i_%i_%s_%s" % (cpu,i,datetime.datetime.today().strftime("%Y-%m-%d"),\
                                                                     str(winner)),dataset)
        print("Current_net wins ratio: %.5f" % (current_wins/num_games))
        savePkl2("wins_cpu_%i" % (cpu),\
                                             {"best_win_ratio": current_wins/num_games, "num_games":num_games})
        logger3.info("[CPU %d]: Finished ArenaManager games!" % cpu)

#Forks or duplicates a process.
def forkProc(ArenaManager_obj, num_games, cpu):
    ArenaManager_obj.assess(num_games, cpu)

#Evaluates neural networks.
def evalNets(args, iteration_1, iteration_2) :
    logger3.info("Loading nets...")
    current_net="%s_iter%d.pth.tar" % (args.neural_net_name, iteration_2); best_net="%s_iter%d.pth.tar" % (args.neural_net_name, iteration_1)
    current_net_filename = os.path.join("./model_data/",\
                                    current_net)
    best_net_filename = os.path.join("./model_data/",\
                                    best_net)

    logger3.info("Current net: %s" % current_net)
    logger3.info("Previous (Best) net: %s" % best_net)

    current_cnet = ConnectionNetwork()
    best_cnet = ConnectionNetwork()
    cuda = torch.cuda.is_available()
    if cuda:
        current_cnet.cuda()
        best_cnet.cuda()

    if not os.path.isdir("./evaluator_data/"):
        os.mkdir("evaluator_data")

    if args.MCTS_num_processes > 1:
        mp.set_start_method("spawn",force=True)

        current_cnet.share_memory(); best_cnet.share_memory()
        current_cnet.eval(); best_cnet.eval()

        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger3.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes
        logger3.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=forkProc,args=(ArenaManager(current_cnet,best_cnet), args.num_evaluator_games, i))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        wins_ratio = 0.0
        for i in range(num_processes):
            stats = loadPkl2("wins_cpu_%i" % (i))
            wins_ratio += stats['best_win_ratio']
        wins_ratio = wins_ratio/num_processes
        if wins_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1

    elif args.MCTS_num_processes == 1:
        current_cnet.eval(); best_cnet.eval()
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
        ArenaManager1 = ArenaManager(current_cnet=current_cnet, best_cnet=best_cnet)
        ArenaManager1.assess(num_games=args.num_evaluator_games, cpu=0)

        stats = loadPkl2("wins_cpu_%i" % (0))
        if stats.best_win_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number to continue training from")
    parser.add_argument("--total_iterations", type=int, default=10000, help="Total iterations for the training process")
    parser.add_argument("--MCTS_num_processes", type=int, default=5, help="Processes count for running MCTS simulations")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=200, help="Games count simulated for each MCTS simulation process")
    parser.add_argument("--temperature_MCTS", type=float, default=1, help="Exploration factor for the initial 10 moves in MCTS simulations")
    parser.add_argument("--num_evaluator_games", type=int, default=200, help="Number of games for evaluating the neural networks")
    parser.add_argument("--neural_net_name", type=str, default="current_game", help="Identifier for the neural network model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during model training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for model optimization")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Steps for accumulating gradients before update")
    parser.add_argument("--max_norm", type=float, default=0.5, help="Maximum allowed gradient norm for clipping during optimization")
    args = parser.parse_args()

    logger4.info("Starting iteration pipeline...")
    for i in range(args.iteration, args.total_iterations):
        runMcts(args, start_idx=0, iteration=i)
        trainCNet(args, iteration=i, new_optim_state=True)
        if i >= 1:
            winner = evalNets(args, i, i + 1)
            counts = 0
            while (winner != (i + 1)):
                logger4.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                runMcts(args, start_idx=(counts + 1)*args.num_games_per_MCTS_process, iteration=i)
                counts += 1
                trainCNet(args, iteration=i, new_optim_state=True)
                winner = evalNets(args, i, i + 1)
