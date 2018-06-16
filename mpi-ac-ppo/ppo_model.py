import numpy as np
import tensorflow as tf
import mpi4py.MPI as MPI
import gym
from policies import MlpPolicy
import time

class Synchronizer(object):
	def __init__(self, env):
		self.env = env
		self.env.seed(0)

	def evaluate_with_episode(self, agent, n_episode, seed=None):
		eprews = np.zeros(n_episode, dtype=np.float32)
		ep_reward = 0
		if seed is not None:
			self.env.seed(seed)
		for ep in range(n_episode):
			ob = self.env.reset()
			ep_reward = 0
			while True:
				ac, _ = agent.step(ob)
				ob, r, done, _ = self.env.step(np.clip(ac, self.env.action_space.low, self.env.action_space.high))
				ep_reward += r
				if done:
					break
			eprews[ep] = ep_reward
		avg_eprew = eprews.mean()
		return avg_eprew
	
	def evaluate_with_step(self, agent, n_step, seed=None):
		total_rew = 0
		if seed is not None:
			self.env.seed(seed)
		ob = self.env.reset()
		for step in range(n_step):
			ac, _ = agent.step(ob)
			ob, r, done, _ = self.env.step(np.clip(ac, self.env.action_space.low, self.env.action_space.high))
			total_rew += r
			if done:
				ob = self.env.reset()
		return total_rew

		
	def sync_wrt_eprew(self, model, n_turn, seed=None):
		assert isinstance(model, PPOModel), 'Should input a PPOModel class'
		value = self.evaluate_with_episode(model.agent_model, n_turn, seed)
		local_eprew = np.array([value], dtype=np.float32)
		min_eprew = np.zeros(1, dtype=np.float32)
		MPI.COMM_WORLD.Allreduce(local_eprew, min_eprew, op=MPI.MIN)
		local_eprew -= min_eprew
		total_eprew = np.zeros(1, dtype=np.float32)
		MPI.COMM_WORLD.Allreduce(local_eprew, total_eprew, op=MPI.SUM)
		weight = local_eprew / total_eprew

		local_p = model.get_params()
		global_p = np.zeros_like(local_p, dtype=np.float32)
		local_p = local_p * weight
		MPI.COMM_WORLD.Allreduce(local_p, global_p, op=MPI.SUM)
		#global_p.astype(np.float32)
		model.apply_params(global_p,tau=1.0)
		return value, weight

	def sync_with_best(self, model, n_turn, seed=None):
		assert isinstance(model, PPOModel), 'Should input a PPOModel class'
		value = self.evaluate_with_episode(model.agent_model, n_turn, seed)
		local_eprew = np.array([value], dtype=np.float32)
		max_eprew = np.zeros(1, dtype=np.float32)
		MPI.COMM_WORLD.Allreduce(local_eprew, max_eprew, op=MPI.MAX)
		if local_eprew == max_eprew:
			weight = 1.0
		else:
			weight = 0.0
		
		local_p = model.get_params()
		global_p = np.zeros_like(local_p, dtype=np.float32)
		if local_eprew != max_eprew:
			local_p = np.zeros_like(local_p, dtype=np.float32)
	
		MPI.COMM_WORLD.Allreduce(local_p, global_p, op=MPI.SUM)
		model.apply_params(global_p,tau=0.9)
		return value, weight

class Runner(object):
	def __init__(self, env, model, n_step, gamma, lam):
		# make sure every agent start with a same env
		#env.seed(0)
		self.env = env
		self.model = model
		self.n_step = n_step
		self.gamma = gamma
		self.lam = lam
		self.ob = np.zeros(env.observation_space.shape)
		self.ob[:] = env.reset().copy()
		self.ep_reward = 0
		self.episode = 0
		self.enable_reset = True
	def run(self):
		# reset replay
		self.enable_reset = False	
		mb_obs, mb_rews, mb_acs, mb_vs, mb_dones mb_nextobs= [], [], [], [], [], []
		eprews = []
		for _ in range(self.n_step):
			ac, v = self.model.step(self.ob)
			ob_, rew, done, _ = self.env.step(np.clip(ac, self.env.action_space.low, self.env.action_space.high))
			self.ep_reward += rew
			mb_obs.append(self.ob.copy())
			mb_acs.append(ac.copy())
			mb_rews.append(rew)
			mb_vs.append(v)
			mb_dones.append(done)
			if done:
				ob_ = self.env.reset()
				eprews.append(self.ep_reward)
				self.ep_reward = 0
				self.episode += 1
			mb_nextobs.append(ob_.copy())
			self.ob[:] = ob_
		v_final = self.model.get_v(ob_)
		mb_vs.append(v_final)

		# calculate adv and target_V
		mb_adv = np.zeros(len(mb_rews))
		assert len(mb_vs) == len(mb_rews) + 1
		assert len(mb_obs) == len(mb_nextobs)
		lastgaelam = 0
		for t in reversed(range(len(mb_rews))):
			delta = mb_rews[t] + self.gamma * mb_vs[t+1] * (1 - mb_dones[t]) - mb_vs[t]
			lastgaelam = delta + self.gamma * self.lam * (1 - mb_dones[t]) * lastgaelam
			mb_adv[t] = lastgaelam
		mb_vs.pop()
		assert len(mb_vs) == len(mb_rews)

		#convert to numpy array
		mb_adv = np.vstack(mb_adv)
		mb_obs = np.vstack(mb_obs)
		mb_nextobs = np.vstack(mb_nextobs)
		mb_vs = np.vstack(mb_vs)
		mb_acs = np.vstack(mb_acs)
		mb_targv = mb_adv.copy() + mb_vs.copy()

		self.enable_reset = True
		return dict(ob=mb_obs.copy(), adv=mb_adv.copy(), vs=mb_vs.copy(), ac=mb_acs.copy(), targv=mb_targv.copy(), epreward=eprews.copy(), nextob=mb_nextobs.copy())

	def reset_with_seed(self, seed=np.random.rand()*4):
		assert self.enable_reset == True, 'reset is not allowed while generating replay'
		self.env.seed(seed)
		self.ob[:] = self.env.reset().copy()




class PPOModel(object):
	def __init__(self, ob_space, ac_space, c_entropy, c_vf, session, max_grad_norm=0.5):

		sess = session

		agent_model = MlpPolicy('Mlp_agent', ob_space, ac_space, session)
		pi = agent_model.pi
		old_pi = agent_model.oldpi
		v = agent_model.vf
		critic = agent_model.critic
		oldcritic = agent_model.oldcritic

		r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="reward")
		a = tf.placeholder(dtype=tf.float32, shape=[None]+list(ac_space.shape), name="a")
		adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantage")
		target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target_v")
		old_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="old_v")
		LR = tf.placeholder(dtype=tf.float32, name="lr")
		CLIP_RANGE = tf.placeholder(dtype=tf.float32, shape=(), name="cliprange")
		TAU_LOCAL = tf.placeholder(dtype=tf.float32, shape=(), name="TAU_LOCAL")
		TAU_GLOBAL = tf.placeholder(dtype=tf.float32, shape=(), name="TAU_GLOBAL")

		with tf.variable_scope('losses'):
			NegLogPac = pi.neglogp(a)
			OldNegLogPac = old_pi.neglogp(a)
			ratio = tf.exp(OldNegLogPac - NegLogPac)
			surr1 = adv * ratio
			surr2 = adv * tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
			pg_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

			entropy = tf.reduce_mean(pi.entropy())
			v_clipped = old_v + tf.clip_by_value(v - old_v, -CLIP_RANGE, CLIP_RANGE)
			vf_losses1 = tf.square(v - target_v)
			vf_losses2 = tf.square(v_clipped - target_v)
			vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
			simple_vf_loss = tf.reduce_mean(vf_losses1)

			approxkl = 0.5 * tf.reduce_mean(tf.square(NegLogPac - OldNegLogPac))
			clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIP_RANGE)))
			
			# critic 
			q_loss = tf.reduce_mean(tf.square(critic - (r + 0.95*oldcritic)))

			#loss = pg_loss - entropy * c_entropy + vf_loss * c_vf
			loss = pg_loss + simple_vf_loss - entropy * c_entropy
			
			#tf.summary.scalar('total_loss', loss)
			#tf.summary.scalar('pol_loss', pg_loss)
			#tf.summary.scalar('vf_loss', simple_vf_loss)

		pi_params = agent_model.get_pol_variables()
		oldpi_params = agent_model.get_oldpol_variables()
		with tf.variable_scope('update_old_pi'):
			_updatepi = [old.assign(old*(1.0-TAU_LOCAL) + new*TAU_LOCAL) for old, new in zip(oldpi_params, pi_params)]

		cri_params = agent_model.get_critic_variables()
		oldcri_params = agent_model.get_oldcritic_variables()
		with tf.variable_scope('update_old_critic'):
			_updatecritic = [old.assign(old*(1.0-TAU_LOCAL) + new*TAU_LOCAL) for old, new in zip(oldcri_params, cri_params)]




		#self.all_params = agent_model.get_trainable_variables()
		self.all_params = agent_model.get_variables()
		self.train_params = tf.trainable_variables(scope=agent_model.scope)
		
		grads = tf.gradients(loss, self.train_params)
		if max_grad_norm is not None:
			grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
		
		# get flattened local gradients
		self.flat_grads = tf.concat(axis=0,
			values=[tf.reshape(gg, shape=[int(np.prod(gg.shape))]) for gg in grads])

		# placeholder for flatten gradients
		feed_grads = tf.placeholder(dtype=tf.float32, shape=self.flat_grads.shape, name='feed_grads')

		# get flattened agent parameters
		self.flat_params = tf.concat(axis=0,
			values=[tf.reshape(ap, shape=[int(np.prod(ap.shape))]) for ap in self.all_params])
		
		# placeholder for flatten params
		feed_params = tf.placeholder(dtype=tf.float32, shape=self.flat_params.shape, name='feed_params')

		## opt for gradient assignment and update
		with tf.name_scope("Apply_grads"):
			update_list = []
			start = 0
			for p in self.train_params:
				end = start + int(np.prod(p.shape))
				update_list.append(tf.reshape(feed_grads[start:end], shape=p.shape))
				start = end
			# create grad-params pair list
			grads_list = list(zip(update_list, self.train_params))
			self.optimizer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
			_train = self.optimizer.apply_gradients(grads_list)
		
		## opt for params assignment
		with tf.name_scope("Apply_params"):
			p_list = []
			start = 0
			for p in self.all_params:
				end = start + int(np.prod(p.shape))
				p_list.append(tf.reshape(feed_params[start:end], shape=p.shape))
				start = end
			_apply_params = [old.assign(new*TAU_GLOBAL + (1-TAU_GLOBAL)*old) for old, new in zip(self.all_params, p_list)]

		# minibatch train
		def train(lr, cliprange, mb_obs, mb_acs, mb_adv, mb_vs, mb_targv, use_global_grad, apply_noise, scale_by_procs=True):
			mb_adv = (mb_adv - mb_adv.mean()) / mb_adv.std()
			feeddict = {agent_model.ob: mb_obs,
						a: mb_acs,
						adv: mb_adv,
						target_v: mb_targv,
						old_v: mb_vs,
						CLIP_RANGE: cliprange}

			# get loss
			ploss, vloss = sess.run([pg_loss, simple_vf_loss], feed_dict=feeddict)

			# get local gradients list
			local_grad = sess.run(self.flat_grads, feed_dict=feeddict)
			assert local_grad.ndim == 1, 'gradients not flattened!'
			if apply_noise:
				local_grad += np.random.normal(loc=0, scale=0.05, size=local_grad.shape)
			# initialize global gradients list
			final_grad = local_grad.copy()

			if use_global_grad:					
				# sync gradients in global gradients buffer
				MPI.COMM_WORLD.Allreduce(local_grad, final_grad, op=MPI.SUM)	
				# scale the global gradients with mpirun number
				if scale_by_procs:
					final_grad = final_grad / MPI.COMM_WORLD.Get_size()
			
			sess.run(_train, feed_dict={LR:lr, feed_grads: final_grad})
			
			return ploss, vloss

		# update old pi with pi
		def update_old_pi(tau=1.0):
			sess.run(_updatepi, feed_dict={TAU_LOCAL: tau})

		def sync_params(tau=1.0):
			# get local params
			local_p = sess.run(self.flat_params)
			# prepare global buffer
			global_p = np.zeros_like(local_p)
			# sync
			MPI.COMM_WORLD.Allreduce(local_p, global_p, op=MPI.SUM)
			# scale params with agent_number
			global_p = global_p / MPI.COMM_WORLD.Get_size()
			sess.run(_apply_params, feed_dict={feed_params: global_p, TAU_GLOBAL:tau})
		
		def apply_noise(sd=0.05):
			p = sess.run(self.flat_params)
			p += np.random.normal(loc=0, scale=sd, size=p.shape)
			sess.run(_apply_params, feed_dict={feed_params: p, TAU_GLOBAL: 1.0})		

		def get_params():
			return sess.run(self.flat_params)

		def apply_params(p, tau=1.0):
			sess.run(_apply_params, feed_dict={feed_params: p, TAU_GLOBAL: tau})

		self.train = train
		self.update_old_pi = update_old_pi
		self.sync_params = sync_params
		self.agent_model = agent_model
		self.get_params = get_params
		self.apply_params = apply_params
		self.apply_noise = apply_noise


def learn(*, policy=None, env, test_env, eval_env,
			timestep_per_actor,
			clipparam,
			c_entropy, c_vf,
			optim_epchos, optim_batchsize, optim_stepsize,
			gamma, lam,
			max_timesteps=0, max_episode=0, max_iters=0, max_second=0,
			schedule='constant', file_path='/home/wang/Research/MODELS/PPO-MODEL/default',
			record_turn=100, cur_episode = 100, terminate_reward = 300
			):

	gpu_option = tf.GPUOptions(
			#per_process_gpu_memory_fraction=0.4,
			allow_growth=True
			)
	sess = tf.Session(
			config=tf.ConfigProto(
				gpu_options=gpu_option,
				log_device_placement=False
				)
			)

	#initialize ppo_model and runner	
	ppo_model = PPOModel(
						ob_space=env.observation_space, ac_space=env.action_space,
						c_entropy=c_entropy, c_vf=c_vf,
						session=sess, max_grad_norm=0.5)

	runner = Runner(env=env, model=ppo_model.agent_model,
					n_step=timestep_per_actor, gamma=gamma, lam=lam)

	synchronizer = Synchronizer(eval_env)

	# Tensorboard initialize
	if MPI.COMM_WORLD.Get_rank() == 0:
		writer = tf.summary.FileWriter(file_path+'/log', sess.graph)
		merged = tf.summary.merge_all()
	
	sess.run(tf.global_variables_initializer())
	#ppo_model.sync_params()

	# initialize pointers
	timestep = 0
	episode = 0
	iters = 0
	second = 0
	avg = 0
	global_end_flag = np.zeros(1, dtype=np.float32)
	local_end_flag = np.zeros(1, dtype=np.float32)
	
	# initialize reward buffer
	eprew_buffer = np.zeros(cur_episode)
	buffer_pointer = 0
	record = []

	assert (max_timesteps or max_iters or max_episode or max_second) != 0, "at least one constrain!"
	while True:
		# condition that break from trainning loop
		if max_timesteps and timestep >= max_timesteps:
			break
		if max_episode and episode >= max_episode:
			break
		if max_iters and iters >= max_iters:
			break
		if max_second and second >= max_second:
			break

		if schedule == 'linear':
			if (max_timesteps == 0) and max_iters:
				cur_lr_att = max(1.0 - float(timestep) / (max_iters*timestep_per_actor), 0.0)
			else:
				cur_lr_att = max(1.0 - float(timestep) / max_timesteps, 0.0)
		else:
			cur_lr_att = 1.0
		
		if MPI.COMM_WORLD.Get_rank() == 0: 
			print("********************** Iteration %i **********************" %iters)
			print("--- Generating replay...")
		batch = runner.run()

		if MPI.COMM_WORLD.Get_rank() == 0: 
			print("--- Generated")	
		ob, ac, adv, vs, targv = batch["ob"], batch["ac"], batch["adv"], batch["vs"], batch["targv"]
		
		"""
		eprew = batch["epreward"]
		# update average ep_reward
		for epr in eprew:
			eprew_buffer[buffer_pointer%cur_episode] = epr
			buffer_pointer += 1
		if buffer_pointer < cur_episode:
			avg = eprew_buffer[0:buffer_pointer].mean()
		else:
			avg = eprew_buffer.mean()
		
		record.append(avg)
		np.save("Worker-"+str(MPI.COMM_WORLD.Get_rank())+"-rewards.npy",np.asarray(record))
		"""

		# update ob fiter
		#ppo_model.agent_model.ob_rms.update(ob)

		# update old pi
		ppo_model.update_old_pi()
		
		# train model with replay
		index = np.arange(timestep_per_actor)
		#adv = (adv - adv.mean()) / adv.std()

		assert index.shape[0] == ob.shape[0]

		vflosses = []
		pollosses = []
		if MPI.COMM_WORLD.Get_rank() == 0:
			print("--- Optimizing...")
		
		#ppo_model.apply_noise()
		for oe in range(optim_epchos):
	
			np.random.shuffle(index)
			for start in range(0, timestep_per_actor, optim_batchsize):
				end = start + optim_batchsize
				mb_index = index[start:end]
				ploss, vloss = ppo_model.train(lr=cur_lr_att*optim_stepsize,
							cliprange=clipparam,
							mb_obs=ob[mb_index],
							mb_acs=ac[mb_index],
							mb_adv=adv[mb_index],
							mb_vs=vs[mb_index],
							mb_targv=targv[mb_index],
							apply_noise=False,
							use_global_grad=False)
				vflosses.append(np.squeeze(vloss))
				pollosses.append(np.squeeze(ploss))
		if MPI.COMM_WORLD.Get_rank() == 0:
			print("--- Synchronizing Parameters...")
		dis_eprew, dis_weight = synchronizer.sync_with_best(ppo_model, 3)
		print("---Worker %i, eprew: %.2f, weight: %.2f" %(MPI.COMM_WORLD.Get_rank(), dis_eprew, dis_weight))
		#ppo_model.sync_params()
		

		# update tensorboard
		if MPI.COMM_WORLD.Get_rank() == 0:
			summary = sess.run(merged)
			writer.add_summary(summary, iters)

		local_avg = synchronizer.evaluate_with_episode(ppo_model.agent_model, 1) 
		global_avg = np.zeros(MPI.COMM_WORLD.Get_size(), dtype=np.float32)
		MPI.COMM_WORLD.Allgather(local_avg, global_avg)

		# update average ep_reward
		for epr in global_avg:
			eprew_buffer[buffer_pointer%cur_episode] = epr
			buffer_pointer += 1
		if buffer_pointer < cur_episode:
			avg = eprew_buffer[0:buffer_pointer].mean()
		else:
			avg = eprew_buffer.mean()

		record.append(avg)
		np.save("Worker-"+str(MPI.COMM_WORLD.Get_rank())+"-rewards.npy",np.asarray(record))

		episode = runner.episode
		timestep += timestep_per_actor
		iters += 1

		n_batch_trained = len(vflosses)
		vflosses = np.asarray(vflosses)
		pollosses = np.asarray(pollosses)
		curvfloss = vflosses.mean()
		curpolloss = pollosses.mean()
		if MPI.COMM_WORLD.Get_rank() == 0:
			print("------------avg: %.2f" %avg)
			print("-buffer_pointer: %i" %buffer_pointer)
			print("--------episode: %i" %episode)
			print("-----cur_lr_att: %.2f" %cur_lr_att)
			print("--n_batch_trained: %i" %n_batch_trained)
			print("---vf loss: %.4f" %curvfloss)
			print("--pol loss: %.4f" %curpolloss)

		
		# if terminate reward reached, end training and start evaluate
		#if (buffer_pointer > cur_episode) and (avg >= terminate_reward) and (MPI.COMM_WORLD.Get_rank() == 0):
		# finally record 
		if iters >= max_iters:
			print("target reached, end trainning")
			print("|")
			print("|")
			print("|")
			print("---- start evaluating...")
			for i in range(10):
				testob = test_env.reset()
				while True:
					a, v = ppo_model.agent_model.step(testob)
					testob, _, done, _ = test_env.step(np.clip(a, test_env.action_space.low, test_env.action_space.high))
					if done:
						break
			# terminate global process
			local_end_flag = np.array([10.0], dtype=np.float32)
		
		# check global end flag
		MPI.COMM_WORLD.Allreduce(local_end_flag, global_end_flag, op=MPI.SUM)
		if global_end_flag.mean() > 0:
			break


		if (iters > 0) and (iters % record_turn == 0) and (MPI.COMM_WORLD.Get_rank() == 0):
			for i in range(3):
				testob = test_env.reset()
				while True:
					a, _ = ppo_model.agent_model.step(testob)
					testob, _, done, _  = test_env.step(np.clip(a, test_env.action_space.low, test_env.action_space.high))
					if done:
						#testob = test_env.reset()
						break












