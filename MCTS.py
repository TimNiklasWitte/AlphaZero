import numpy as np
import copy


c_puct = 1

class Node:
    def __init__(self, parent, env, state, action, p, reward, done):

        
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p

        self.reward = reward

        self.parent = parent
        self.childrens = []

        self.env = env
        self.state = state

        self.action = action

        self.done = done

    def getU(self):
        return c_puct * self.p * (np.sqrt(self.parent.n)/(1 + self.n))


class MCTS:

    def __init__(self, env, state, policyValueNetwork):
        
        self.N = 0
        self.gamma = 0.9

        env = copy.deepcopy(env)
        self.env = env
        self.root = Node(parent=None, env=env, action=-1, state=state, p=1, reward=0, done=False)

        self.policyValueNetwork = policyValueNetwork
 
    def run(self, num_iterations: int):
        
        for i in range(num_iterations):
            
            current = self.root

            #
            # Select
            #
      
            while len(current.childrens) != 0:
     
                UCB_values = [node.q + node.getU() for node in current.childrens]
                
                max_idx = np.argmax(UCB_values)
                current = current.childrens[max_idx]

            
            #
            # Expand and evaluate
            #
            
            if current.done:
                value = 0

            else:
                # Evaluate
                state = np.expand_dims(current.state, axis=0)
                policy, value = self.policyValueNetwork(state)

                policy = policy.numpy()[0]
                value = value.numpy()[0][0]
              
                # Expand
                for action in range(2):
                                    
                    env_current = copy.deepcopy(current.env)
                                    
                    next_state, reward, done, _, _ = env_current.step(action)
                    next_state = np.copy(next_state)
                    node = Node(parent=current, env=env_current, state=next_state, action=action, p=policy[action], reward=reward,done=done)
                    current.childrens.append(node)
                
          
            #
            # Backup
            #
            self.backup(current, value)

       
        return self.get_policy()
        
    
    def backup(self, node, G):

        current = node

        while current:
            
            current.n += 1

            current.w = current.w + G
            current.q = current.w / current.n

            G = current.reward + self.gamma * G

            current = current.parent

         
        
    def get_policy(self):

        # softmax
        n_values = np.array([np.exp(node.n) for node in self.root.childrens])
        n_total = np.sum(n_values)

        if n_total == 0:
            return np.full(shape=(2,), fill_value=1/2, dtype=np.float32)
        
        policy = n_values / n_total

        return policy


        


    

           