import gym
import gym.spaces as spaces
import tensorflow as tf
import numpy as np
import mpi4py.MPI as MPI
from policies import MlpPolicy
import ppo_model

#ENV_NAME ='BipedalWalkerHardcore-v2'
ENV_NAME = 'LunarLanderContinuous-v2'
#ENV_NAME = 'Pendulum-v0'
Path = './'

if __name__ == '__main__':
	
	# train env
	env = gym.make(ENV_NAME)
	test_env = gym.make(ENV_NAME)
	eval_env = gym.make(ENV_NAME)
	# monitoring env
	if MPI.COMM_WORLD.Get_rank() == 0:
		test_env = gym.wrappers.Monitor(
			env=test_env,
			directory=Path+'/video'+'/'+ENV_NAME,
			video_callable=lambda x: True,
			force = True
			)


	# train policy model
	ppo_model.learn(env=env, test_env=test_env, eval_env=eval_env,
				timestep_per_actor=4096,
				clipparam=0.2,
				c_entropy=0.01, c_vf=1.0,
			optim_epchos=10, optim_batchsize=64, optim_stepsize=3e-4,
			gamma=0.99, lam=0.95,
			max_timesteps=0, max_episode=0, max_iters=500, max_second=0,
			schedule='linear', file_path=Path, record_turn=50, cur_episode=100,
			terminate_reward=220
			)

	env.close()
	test_env.close()
	eval_env.close()