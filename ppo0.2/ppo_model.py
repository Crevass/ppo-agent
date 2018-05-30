import numpy as np
import tensorflow as tf
import mpi4py.MPI as MPI
import gym
from policies import MlpPolicy
import time

class Runner(object):
	def __init__(self, env, model, n_step, gamma, lam):
		self.env = env
		self.model = model
		self.n_step = n_step
		self.gamma = gamma
		self.lam = lam
		self.ob = np.zeros(env.observation_space.shape)
		self.ob[:] = env.reset().copy()
		self.ep_reward = 0
		self.episode = 0
	def run(self):
		# reset replay	
		mb_obs, mb_rews, mb_acs, mb_vs, mb_dones = [], [], [], [], []
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
			self.ob[:] = ob_
		v_final = self.model.get_v(ob_)
		mb_vs.append(v_final)

		# calculate adv and target_V
		mb_adv = np.zeros(len(mb_rews))
		assert len(mb_vs) == len(mb_rews) + 1
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
		mb_vs = np.vstack(mb_vs)
		mb_acs = np.vstack(mb_acs)
		mb_targv = mb_adv.copy() + mb_vs.copy()

		return dict(ob=mb_obs.copy(), adv=mb_adv.copy(), vs=mb_vs.copy(), ac=mb_acs.copy(), targv=mb_targv.copy(), epreward=eprews.copy())

class PPOModel(object):
	def __init__(self, ob_space, ac_space, c_entropy, c_vf, session, max_grad_norm=0.5):

		sess = session

		agent_model = MlpPolicy('Mlp_agent', ob_space, ac_space, session)
		pi = agent_model.pi
		old_pi = agent_model.oldpi
		v = agent_model.vf

		a = tf.placeholder(dtype=tf.float32, shape=[None]+list(ac_space.shape), name="a")
		adv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantage")
		target_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target_v")
		old_v = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="old_v")
		LR = tf.placeholder(dtype=tf.float32, name="lr")
		CLIP_RANGE = tf.placeholder(dtype=tf.float32, shape=(), name="cliprange")
		TAU = tf.placeholder(dtype=tf.float32, shape=(), name="TAU")

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
			#loss = pg_loss - entropy * c_entropy + vf_loss * c_vf
			loss = pg_loss + simple_vf_loss - entropy * c_entropy
			
		pi_params = agent_model.get_pol_variables()
		oldpi_params = agent_model.get_oldpol_variables()
		with tf.variable_scope('update_old_pi'):
			_updatepi = [old.assign(old*(1.0-TAU) + new*TAU) for old, new in zip(oldpi_params, pi_params)]

		params = tf.trainable_variables(scope=agent_model.scope)
		grads = tf.gradients(loss, params)
		if max_grad_norm is not None:
			grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
		grads = list(zip(grads, params))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
		_train = self.optimizer.apply_gradients(grads)
		"""
		def global_train(*, lr, cliprange, bobs, bacs, badv, bvs, btargv, scale_by_procs = True):
			badv = (badv - badv.mean()) / (badv.std() + 1e-8)
			feeddict={pi.ob: bobs, old_pi.ob: bobs, a: bacs, adv: badv, old_v: bvs, target_v: btargv, LR: lr, CLIP_RANGE: cliprange}
			localg = sess.run(tf.gradients(loss, params), feed_dict=feeddict)
			globalg = np.zeros_like(localg)
			MPI.COMM_WORLD.Allreduce(localg, globalg, op=MPI.SUM)
			if scale_by_procs:
				globalg /= MPI.COMM_WORLD.Get_size()
			if max_grad_norm is not None:
				globalg, _grad_norm = tf.clip_by_global_norm(globalg, max_grad_norm)
			grads = list(zip(globalg, params))
			sess.run(optimizer.apply_gradients(grads))
		"""
		def train(lr, cliprange, mb_obs, mb_acs, mb_adv, mb_vs, mb_targv):
			mb_adv = (mb_adv - mb_adv.mean()) / mb_adv.std()
			feeddict = {agent_model.ob: mb_obs,
						a: mb_acs,
						adv: mb_adv,
						target_v: mb_targv,
						old_v: mb_vs,
						LR: lr,
						CLIP_RANGE: cliprange}
			sess.run(_train, feed_dict=feeddict)
			return sess.run([pg_loss, simple_vf_loss], feed_dict=feeddict)

		def update_old_pi(tau=0.5):
			sess.run(_updatepi, feed_dict={TAU: tau})

		self.train = train
		#self.global_train = global_train
		self.update_old_pi = update_old_pi
		self.agent_model = agent_model

def learn(*, policy=None, env, test_env,
			timestep_per_actor,
			clipparam,
			c_entropy, c_vf,
			optim_epchos, optim_batchsize, optim_stepsize,
			gamma, lam,
			max_timesteps=0, max_episode=0, max_iters=0, max_second=0,
			schedule='constant', file_path='/home/wang/Research/MODELS/PPO-MODEL/default',
			record_turn=10, cur_iters=100
			):

	gpu_option = tf.GPUOptions(
			per_process_gpu_memory_fraction=0.4,
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
	
	# Tensorboard initialize
	writer = tf.summary.FileWriter(file_path+'/log', sess.graph)
	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	
	# initialize pointers
	timestep = 0
	episode = 0
	iters = 0
	second = 0
	# initialize reward buffer
	eprew_buffer = np.zeros(cur_iters)
	buffer_pointer = 0

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
	
		print("-----------------------Iteration %i-----------------------" %iters)
		# get new replay
		batch = runner.run()
		ob, ac, adv, vs, targv = batch["ob"], batch["ac"], batch["adv"], batch["vs"], batch["targv"]
		eprew = batch["epreward"]
		# update ob fiter
		ppo_model.agent_model.ob_rms.update(ob)
		# update old pi
		ppo_model.update_old_pi()

		# train model with replay
		index = np.arange(timestep_per_actor)
		#adv = (adv - adv.mean()) / adv.std()

		assert index.shape[0] == ob.shape[0]

		vflosses = []
		pollosses = []
		for _ in range(optim_epchos):
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
							mb_targv=targv[mb_index])
				vflosses.append(np.squeeze(vloss))
				pollosses.append(np.squeeze(ploss))
		episode = runner.episode
		for epr in eprew:
			eprew_buffer[buffer_pointer%cur_iters] = epr
			buffer_pointer += 1
		avg = eprew_buffer.mean()
		timestep += timestep_per_actor
		iters += 1

		# update tensorboard
		summary = sess.run(merged)
		writer.add_summary(summary, iters)

		n_batch_trained = len(vflosses)
		vflosses = np.asarray(vflosses)
		pollosses = np.asarray(pollosses)
		curvfloss = vflosses.mean()
		curpolloss = pollosses.mean()

		print("avg: %.2f" %avg)
		print("episode: %i" %episode)
		print("cur_lr_att: %.2f" %cur_lr_att)
		print("n_batch_trained: %i" %n_batch_trained)
		print("vf loss: %.4f" %curvfloss)
		print("pol loss: %.4f" %curpolloss)

		if (iters > 0) and (iters % record_turn == 0):
			testob = test_env.reset()
			while True:
				a, _ = ppo_model.agent_model.step(testob)
				testob, _, done, _  = test_env.step(np.clip(a, test_env.action_space.low, test_env.action_space.high))
				if done:
					break














