import numpy as np
import gym
from gym.spaces import Discrete, Box

# poicies
class DiscreteAction(object):
    def __init__(self, theta, ob_space, ac_space):
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.n
        self.W = theta[0: ob_dim * ac_dim].reshape(ob_dim, ac_dim)
        self.b = theta[ob_dim * ac_dim:].reshape(1, ac_dim)

    def act(self, ob):
        y = np.dot(ob, self.W) + self.b
        a = np.argmax(y)
        return a

class ContinuousAction(object):
    def __init__(self, theta, ob_space, ac_space):
        self.ac_space = ac_space
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.shape[0]
        self.W = theta[0: ob_dim * ac_dim].reshape(ob_dim, ac_dim)
        self.b = theta[ob_dim * ac_dim: ]

    def act(self, ob):
        y = np.dot(ob, self.W) + self.b
        a = np.clip(y, self.ac_space.low, self.ac_space.high)
        return a

def run_episode(policy, env, render=False):
    max_steps = 1000
    total_rew = 0
    ob = env.reset()
    for t in range(max_steps):
        a = policy.act(ob)
        ob, reward, done, _info = env.step(a)
        total_rew += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_rew

def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DiscreteAction(theta, env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return ContinuousAction(theta, env.observation_space, env.action_space)
    else:
        raise NotImplementedError

def eval_policy(theta):
    policy = make_policy(theta)
    reward = run_episode(policy, env)
    return reward

# env = gym.make('CartPole-v0')
env = gym.make('BipedalWalker-v2')
n_iter = 100
batch_size = 25
elite_frac = 0.2

if isinstance(env.action_space, Discrete):
    dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# init policy
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

for iteration in range(n_iter):
    # sample patameter vector
    thetas = np.random.multivariate_normal(theta_mean, np.diag(theta_std), size=batch_size)

    rewards = [eval_policy(theta) for theta in thetas]
    n_elite = int(batch_size * elite_frac)
    elite_idxs = np.argsort(rewards)[batch_size - n_elite: batch_size]
    elite_thetas = [thetas[i] for i in elite_idxs]

    # update theta_mean and theta_std
    theta_mean = np.mean(np.asarray(elite_thetas), axis=0)
    theta_std = np.std(np.asarray(elite_thetas), axis=0)

    print 'Ep %d: mean f: %8.3f. max f: %4.3f' % (iteration, np.mean(rewards), np.max(rewards))
    print 'Eval reward: ', run_episode(make_policy(theta_mean), env, render=True)
