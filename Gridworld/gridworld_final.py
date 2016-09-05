from pylab import *                                                                                                  
import numpy as np
from time import sleep
import math

ion()

class Gridworld:
    """
    A class that implements a quadratic NxN gridworld. 
    
    Methods:
    
    learn(N_trials=100)  : Run 'N_trials' trials. A trial is finished, when the agent reaches the reward location.
    visualize_trial()  : Run a single trial with graphical output.
    reset()            : Make the agent forget everything he has learned.
    plot_Q()           : Plot of the Q-values .
    learning_curve()   : Plot the time it takes the agent to reach the target as a function of trial number. 
    navigation_map()     : Plot the movement direction with the highest Q-value for all positions.
    integrated_reward_curve()     : Plot the movement direction with the highest Q-value for all positions.
    """    
        
    def __init__(self,N,reward_position=(0.8*20,0.8*20),obstacle=False, lambda_eligibility=0.95):
        """
        Creates a quadratic NxN gridworld. 

        Mandatory argument:
        N: size of the gridworld

        Optional arguments:
        reward_position = (x_coordinate,y_coordinate): the reward location
        obstacle = True:  Add a wall to the gridworld.
        """    
        
        # gridworld size
        self.N = N

        # reward location
        self.reward_position = reward_position        

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 10.
        self.reward_at_wall   = -2.
        
        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = 0.5
                                                                                                  
        # learning rate
        self.eta = 0.005

        # discount factor - quantifies how far into the futurereward
        # a reward is still considered important for the
        # current action
        self.gamma = 0.95

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility
    
        # is there an obstacle in the room?
        self.obstacle = obstacle

        # initialize the Q-values etc.
        self._init_run()

    def run(self,N_trials=10,N_runs=10):  
        self.latencies = zeros(N_trials)
        self.tot_reward = zeros(N_trials)
        for run in range(N_runs):
            print("run: "+str(run))
            self._init_run()
            self._learn_run(N_trials=N_trials)
            self.latencies += array(self.latency_list)/N_runs
            self.tot_reward += array(self.tot_reward_list)/N_runs

    def visualize_trial(self):
        """
        Run a single trial with a graphical display that shows in
                red   - the position of the agent
                blue  - walls/obstacles
                green - the reward position
        """
        # store the old exploration/exploitation parameter
        epsilon = self.epsilon

        # favor exploitation, i.e. use the action with the
        # highest Q-value most of the time
        self.epsilon = 0.5

        self._run_trial(visualize=True)

        # restore the old exploration/exploitation factor
        self.epsilon = epsilon

    def learning_curve(self,log=False,filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        figure()
        xlabel('trials')
        ylabel('time to reach target')
        latencies = array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)

        if not log:
            plot(self.latencies)
        else:
            semilogy(self.latencies)
        show()

    def integrated_reward_curve(self):
        """
        Show a running average of the total reward accumulated by the agent during one run.
        """
        figure()
        xlabel('trials')
        ylabel('integrated reward')
        plot(self.tot_reward)
        show()


    def navigation_map(self, saveas=False):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = np.zeros((self.N,self.N))
        self.y_direction = np.zeros((self.N,self.N))
        self.actions = np.zeros((self.N,self.N))
        
        Qtemp = np.zeros(8);
        for i in range(self.N):
          for j in range(self.N):
            for a in range(8):
              r = self.activity_r(i,j)
              Qtemp[a] = self.compute_Q(r, a)
            self.actions[i,j] = argmax(Qtemp[:],axis=0)
            
            if self.actions[i,j]==0:
              self.y_direction[i,j] = 1.
              self.x_direction[i,j] = 0.
            elif self.actions[i,j] == 1:
              self.y_direction[i,j]= -1.
              self.x_direction[i,j] = 0.
            elif self.actions[i,j]==2:
              self.y_direction[i,j] = 0.
              self.x_direction[i,j] = +1.
            elif self.actions[i,j] == 3:
              self.y_direction[i,j] = 0.
              self.x_direction[i,j]= -1.
            elif self.actions[i,j] == 4:
              self.y_direction[i,j] = 1/math.sqrt(2)
              self.x_direction[i,j] = 1/math.sqrt(2)
            elif self.actions[i,j] == 5:
              self.y_direction[i,j] = -1/math.sqrt(2)
              self.x_direction[i,j] = +1/math.sqrt(2)
            elif self.actions[i,j] == 6:
              self.y_direction[i,j] = -1/math.sqrt(2)
              self.x_direction[i,j] = -1/math.sqrt(2)
            elif self.actions[i,j] == 7:
              self.y_direction[i,j] = +1/math.sqrt(2)
              self.x_direction[i,j]= -1/math.sqrt(2)
            else:
              print "There must be a bug. This is not a valid action!"

        figure()
        quiver(self.x_direction,self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])
        if saveas:
          savefig(str(self.trial)+'_nav_map.pdf', bbox_inches='tight')
        else:
          show()

    def reset(self):
        """
        Reset the Q-values (and the latency_list).
        
        Instant amnesia -  the agent forgets everything he has learned before    
        """
	self.Q = np.zeros((8))
	self.w = np.zeros((self.N,self.N,8))
	self.r = np.zeros((self.N,self.N))
        self.e = np.zeros((self.N,self.N,8))
        self.latency_list = []
        self.tot_reward_list = []


    
    ###############################################################################################
    # The remainder of methods is for internal use and only relevant to those of you
    # that are interested in the implementation details
    ###############################################################################################
        
    
    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        self.Q = np.zeros(8)
        self.w = np.zeros((self.N,self.N,8))
        self.r = np.zeros((self.N,self.N))
        self.e = np.zeros((self.N,self.N,8))
        
        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []
        self.tot_reward_list = []

        # initialize the state and action variables
        self.x_position = 0.1*self.N
        self.y_position = 0.1*self.N
        self.action = None

    def _learn_run(self,N_trials=10):
        """
        Run a learning period consisting of N_trials trials. 
        
        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.
        
        """
        for trial in range(N_trials):
            self.trial = trial # used for nav map (saving image)
            print("trial: "+str(trial))
#            activate next line for varying epsilon experiment
#            self.epsilon = 0.8 * (1. - trial/N_trials)
            # run a trial and store the time it takes to the target
            latency,reward = self._run_trial(visualize=False)
            self.latency_list.append(latency)
            self.tot_reward_list.append(reward)
#            if trial == 0 or trial == 9 or trial == 99:
#           change to true to save images (default is false)
#              self.navigation_map(saveas=False)
        return


    def _run_trial(self,visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """
        # choose the initial position and make sure that its not in the wall
        self.x_position = 0.1*self.N
        self.y_position = 0.1*self.N
        
        # initialize the latency (time to reach the target) for this trial and the reward for the run
        latency = 0.
        reward=0.

        # start the visualization, if asked for
        if visualize:
            self._init_visualization()    
            
        # run the trial
        self._choose_action()
        Nmax=10000
        while not self._arrived() and latency<Nmax:
            self._update_state()
            self._choose_action()    
            self._update_w()
            if visualize:
                self._visualize_current_state()
            latency = latency + 1
            if self._wall_touch:
            	reward=reward+ self.reward_at_wall

        if self._arrived():
            reward=reward+ self.reward_at_target
        if visualize:
            self._close_visualization()
        return latency, reward

    def activity_r(self,x,y):
      sigma=0.05*self.N
      i = np.arange(self.N)
      j = np.arange(self.N)
      ii,jj = meshgrid(i,j)
      return exp(-((ii-x)**2+(jj-y)**2)/(2*(sigma**2)))

    def compute_Q(self, r, action):
      return np.sum(np.multiply(self.w[:,:,action], r))
	
    def _update_w(self):   
        """
	here I kept the name update_Q, but actually the w is updated and the Qs are just computed

        Update the current estimate of the Q-values according to SARSA.
        """

        self.r_old = self.r
        self.r = self.activity_r(self.x_position,self.y_position)
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.gamma * self.e
        self.e[:,:,self.action_old] += self.r_old

        # update the Q-values   this is just the SARSA algo

        if self.action_old != None:
            deltat=self._reward() - ( self.compute_Q(self.r_old,self.action_old) \
                - self.gamma * self.compute_Q(self.r, self.action) )
            self.w+= self.eta*deltat*self.e
        for act in range(8):
            self.Q[act]=self.compute_Q(self.r, act)


    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action 
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if np.random.rand() < self.epsilon:
            self.action = np.random.randint(8)
        else:
            self.action = argmax(self.Q[:])    
    
    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        target_radius=0.1*self.N  
        return ((self.reward_position[0]-self.x_position)**2+(self.reward_position[1]-self.y_position)**2<=(target_radius**2))

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the 
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        this must be modify to have 8 possible actions and l =0.03
	
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        # move to the down
        if self.action == 0:
            self.x_position += 0.03*self.N
            #print "direction 0"
        # move up
        elif self.action == 1:
            self.x_position -= 0.03*self.N
            #print "direction 1"
        # move right
        elif self.action == 2:
            self.y_position += 0.03*self.N
            #print "direction 2"
        # move left
        elif self.action == 3:
            self.y_position -= 0.03*self.N
            #print "direction 3"
        
        #move down right
        elif self.action == 4:
            self.x_position += 0.03*self.N*(1/math.sqrt( 2 ))
            self.y_position += 0.03*self.N*(1/math.sqrt( 2 ))
            #print "direction 4"
        #move up right
        elif self.action == 5:
            self.x_position -= 0.03*self.N*(1/math.sqrt( 2 ))
            self.y_position += 0.03*self.N*(1/math.sqrt( 2 ))
            #print "direction 5"
        #move up left
        elif self.action == 7:
            self.x_position -= 0.03*self.N*(1/math.sqrt( 2 ))
            self.y_position -= 0.03*self.N*(1/math.sqrt( 2 ))
            #print "direction 7"
        #move up left
        elif self.action == 6:
            self.x_position += 0.03*self.N*(1/math.sqrt( 2 ))
            self.y_position -= 0.03*self.N*(1/math.sqrt( 2 ))
            #print "direction 6"
        
        else:
            print "There must be a bug. This is not a valid action!"
                        
        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
        else:
            self._wall_touch = False

    def _is_wall(self,x_position=None,y_position=None):    
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have 
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position <= 0 or x_position >= self.N or y_position <= 0 or y_position >= self.N:
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            if y_position == self.N/2 and x_position>self.N/2:
                return True

        # if none of the above is the case, this position is not a wall
        return False 
            
    def _visualize_current_state(self):
        """
        Show the gridworld. The squares are colored in 
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """

        # set the agents color
        self._display[self.x_position_old,self.y_position_old,0] = 0
        self._display[self.x_position_old,self.y_position_old,1] = 0
        self._display[self.x_position,self.y_position,0] = 1
        if self._wall_touch:
            self._display[self.x_position,self.y_position,1] = 1
            
        # set the reward locations
        self._display[self.reward_position[0],self.reward_position[1],1] = 1

        # update the figure
        self._visualization.set_data(self._display)
        #close()
        imshow(self._display,interpolation='nearest',origin='lower')
        show(block=False)
        
        draw()
        
        # and wait a little while to control the speed of the presentation
        #sleep(0.2)
        
    def _init_visualization(self):
        
        # create the figure
        figure()
        # initialize the content of the figure (RGB at each position)
        self._display = np.zeros((self.N,self.N,3))

        # position of the agent
        self._display[self.x_position,self.y_position,0] = 1
        self._display[self.reward_position[0],self.reward_position[1],1] = 1
        
        for x in range(self.N):
            for y in range(self.N):
                if self._is_wall(x_position=x,y_position=y):
                    self._display[x,y,2] = 1.

                self._visualization = imshow(self._display,interpolation='nearest',origin='lower')
        
    def _close_visualization(self):
        print "Press <return> to proceed..."
        raw_input()
        close()
