import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'data/cifar'
if len(DATA_DIR) == 0:
	raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 8 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

N_NODES = 10
NODES = range(N_NODES)

with open('C3.csv','r') as file:
	C=np.loadtxt(file,delimiter=',',dtype=float)

#C = np.ones([N_NODES,N_NODES]) / N_NODES

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
	return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
	output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
	return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
	output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
	return LeakyReLU(output)

def Generator(n_samples, noise=None, index = ''):
	if noise is None:
		noise = tf.random_normal([n_samples, 128])

	output = lib.ops.linear.Linear('Generator{}.Input'.format(index), 128, 4*4*4*DIM, noise)
	output = lib.ops.batchnorm.Batchnorm('Generator{}.BN1'.format(index), [0], output)
	output = tf.nn.relu(output)
	output = tf.reshape(output, [-1, 4*DIM, 4, 4])

	output = lib.ops.deconv2d.Deconv2D('Generator{}.2'.format(index), 4*DIM, 2*DIM, 5, output)
	output = lib.ops.batchnorm.Batchnorm('Generator{}.BN2'.format(index), [0,2,3], output)
	output = tf.nn.relu(output)

	output = lib.ops.deconv2d.Deconv2D('Generator{}.3'.format(index), 2*DIM, DIM, 5, output)
	output = lib.ops.batchnorm.Batchnorm('Generator{}.BN3'.format(index), [0,2,3], output)
	output = tf.nn.relu(output)

	output = lib.ops.deconv2d.Deconv2D('Generator{}.5'.format(index), DIM, 3, 5, output)

	output = tf.tanh(output)

	return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, index = ''):
	output = tf.reshape(inputs, [-1, 3, 32, 32])

	output = lib.ops.conv2d.Conv2D('Discriminator{}.1'.format(index), 3, DIM, 5, output, stride=2)
	output = LeakyReLU(output)

	output = lib.ops.conv2d.Conv2D('Discriminator{}.2'.format(index), DIM, 2*DIM, 5, output, stride=2)
	if MODE != 'wgan-gp':
		output = lib.ops.batchnorm.Batchnorm('Discriminator{}.BN2'.format(index), [0,2,3], output)
	output = LeakyReLU(output)

	output = lib.ops.conv2d.Conv2D('Discriminator{}.3'.format(index), 2*DIM, 4*DIM, 5, output, stride=2)
	if MODE != 'wgan-gp':
		output = lib.ops.batchnorm.Batchnorm('Discriminator{}.BN3'.format(index), [0,2,3], output)
	output = LeakyReLU(output)

	output = tf.reshape(output, [-1, 4*4*4*DIM])
	output = lib.ops.linear.Linear('Discriminator{}.Output'.format(index), 4*4*4*DIM, 1, output)

	return tf.reshape(output, [-1])

real_data_int = []
real_data = []
fake_data = []

disc_real = []
disc_fake = []

gen_params = []
disc_params = []

for i in NODES:
	real_data_int.append(tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM]))
	real_data.append(2*((tf.cast(real_data_int[i], tf.float32)/255.)-.5))
	fake_data.append(Generator(BATCH_SIZE, index=i))

	disc_real.append(Discriminator(real_data[i], index = i))
	disc_fake.append(Discriminator(fake_data[i], index = i))

	#gen_params.append(lib.params_with_name('Generator{}'.format(i)))
	#disc_params.append(lib.params_with_name('Discriminator{}'.format(i)))

	gen_params.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator{}'.format(i)))
	disc_params.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator{}'.format(i)))


gen_cost = []
disc_cost = []

gen_train_op = []
disc_train_op = []

clip_disc_weights = []

for i in NODES:

	if MODE == 'wgan':
		gen_cost.append( -tf.reduce_mean(disc_fake[i]) )
		disc_cost.append( tf.reduce_mean(disc_fake[i]) - tf.reduce_mean(disc_real[i]) )

		gen_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost[i], var_list=gen_params[i]))
		disc_train_op.append(tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost[i], var_list=disc_params[i]))

		clip_ops = []
		for var in disc_params[i]:
			clip_bounds = [-.01, .01]
			clip_ops.append(
				tf.assign(
					var, 
					tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
				)
			)
		clip_disc_weights.append(tf.group(*clip_ops))

	elif MODE == 'wgan-gp':
		# Standard WGAN loss
		gen_cost.append( -tf.reduce_mean(disc_fake[i]) )
		disc_cost.append( tf.reduce_mean(disc_fake[i]) - tf.reduce_mean(disc_real[i]) )

		# Gradient penalty
		alpha = tf.random_uniform(
			shape=[BATCH_SIZE,1], 
			minval=0.,
			maxval=1.
		)
		differences = fake_data[i] - real_data[i]
		interpolates = real_data[i] + (alpha*differences)
		gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
		gradient_penalty = tf.reduce_mean((slopes-1.)**2)
		disc_cost[i] += LAMBDA*gradient_penalty

		gen_train_op.append(tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9, use_nesterov=True).minimize(gen_cost[i], var_list=gen_params[i]))
		disc_train_op.append(tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9, use_nesterov=True).minimize(disc_cost[i], var_list=disc_params[i]))

		#gen_train_op.append(tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost[i], var_list=gen_params[i]))
		#disc_train_op.append(tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost[i], var_list=disc_params[i]))

	elif MODE == 'dcgan':
		gen_cost.append( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake[i], labels = tf.ones_like(disc_fake[i]))) )
		disc_cost.append( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake[i], labels = tf.zeros_like(disc_fake[i]))) )
		disc_cost[i] += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real[i], labels = tf.ones_like(disc_real[i])))
		disc_cost[i] /= 2.

		gen_train_op.append( tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9, use_nesterov=True).minimize(gen_cost[i],
													   var_list=lib.params_with_name('Generator{}'.format(i))) )
		disc_train_op.append( tf.train.MomentumOptimizer(learning_rate=2e-4, momentum=0.9, use_nesterov=True).minimize(disc_cost[i],
													   var_list=lib.params_with_name('Discriminator{}'.format(i))) )
# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128_3 = Generator(128, noise=fixed_noise_128, index=3)
fixed_noise_samples_128_6 = Generator(128, noise=fixed_noise_128, index=6)

def generate_image_3(frame):
	samples = session.run(fixed_noise_samples_128_3)
	samples = ((samples+1.)*(255./2)).astype('int32')
	lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), 'samples_3_{}.png'.format(frame))

def generate_image_6(frame):
		samples = session.run(fixed_noise_samples_128_6)
		samples = ((samples+1.)*(255./2)).astype('int32')
		lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), 'samples_6_{}.png'.format(frame))


# For calculating inception score

#samples_100_array = [Generator(100, index=ind) for ind in NODES]
#def get_inception_score_node(node):
#        all_samples = []
#        for i in xrange(10):
#                all_samples.append(session.run(samples_100_array[node]))
#        all_samples = np.concatenate(all_samples, axis=0)
#        all_samples = ((all_samples+1.)*(255./2)).astype('int32')
#        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
#        return lib.inception_score.get_inception_score(list(all_samples))


samples_100 = Generator(100, index = 0)
def get_inception_score():
	all_samples = []
	for i in xrange(10):
		all_samples.append(session.run(samples_100))
	all_samples = np.concatenate(all_samples, axis=0)
	all_samples = ((all_samples+1.)*(255./2)).astype('int32')
	all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
	return lib.inception_score.get_inception_score(list(all_samples))

samples_100_3 = Generator(100, index = 3)
def get_inception_score_3():
		all_samples = []
		for i in xrange(10):
				all_samples.append(session.run(samples_100_3))
		all_samples = np.concatenate(all_samples, axis=0)
		all_samples = ((all_samples+1.)*(255./2)).astype('int32')
		all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
		return lib.inception_score.get_inception_score(list(all_samples))

samples_100_6 = Generator(100, index = 6)
def get_inception_score_6():
		all_samples = []
		for i in xrange(10):
				all_samples.append(session.run(samples_100_6))
		all_samples = np.concatenate(all_samples, axis=0)
		all_samples = ((all_samples+1.)*(255./2)).astype('int32')
		all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
		return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen = []
dev_gen = []

for nod in NODES:
	tt, dd = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR, index=None)
	train_gen.append(tt)
	dev_gen.append(dd)

def inf_train_gen(nod):
	while True:
		for images,_ in train_gen[nod]():
			yield images


# COMBINATION OP
C_tf = tf.constant(C)

ops = []
for i in NODES:
	for o in range(len(gen_params[i])):
		ops.append(
			tf.assign(gen_params[i][o], tf.add_n( [ C[i,j] * gen_params[j][o] for j in NODES] )
				))
	for p in range(len(disc_params[i])):
		ops.append(
			tf.assign(disc_params[i][p], tf.add_n( [ C[i,j] * disc_params[j][p] for j in NODES] )
				))

combination_op = tf.group(*ops)

gen_params_mean = []
disc_params_mean = []
for o in range(len(gen_params[0])):
	gen_params_mean.append(
					1.0/N_NODES * tf.add_n( [ gen_params[j][o] for j in NODES] )
							)
for p in range(len(disc_params[0])):
	disc_params_mean.append(
					1.0/N_NODES * tf.add_n( [ disc_params[j][p] for j in NODES] )
							)

gen_dist =  [
			tf.reduce_sum( [ tf.reduce_sum( tf.squared_difference(gen_params[i][o],gen_params_mean[o]) ) for o in range(len(gen_params[i]))] ) for i in NODES]

disc_dist = [
						tf.reduce_sum( [ tf.reduce_sum( tf.squared_difference(disc_params[i][o],disc_params_mean[o]) ) for o in range(len(disc_params[i]))] ) for i in NODES]

gen_mean_norm = tf.reduce_sum( [ tf.reduce_sum( tf.square(gen_params_mean[o]) ) for o in range(len(gen_params_mean))] )
disc_mean_norm = tf.reduce_sum( [ tf.reduce_sum( tf.square(disc_params_mean[o]) ) for o in range(len(disc_params_mean))] )

saver = tf.train.Saver()

# Train loop
with tf.Session() as session:
	session.run(tf.initialize_all_variables())

	gen = []
	for nod in NODES:
		gen.append(inf_train_gen(nod))


	for iteration in xrange(ITERS):
		start_time = time.time()

		for node in NODES:
			# Train generator
			if iteration > 0:
				_ = session.run(gen_train_op[node])
			# Train critic
			if MODE == 'dcgan':
				disc_iters = 1
			else:
				disc_iters = CRITIC_ITERS
			for i in xrange(disc_iters):
				_data = gen[nod].next()
				_disc_cost, _ = session.run([disc_cost[node], disc_train_op[node]], feed_dict={real_data_int[node]: _data})
				if MODE == 'wgan':
					_ = session.run(clip_disc_weights[node])

					if (iteration % 100 == 99 or iteration < 10):
							#lib.plot.plot('NODE {}: train disc cost'.format(node), _disc_cost)
							print('iter {} NODE {}: train disc cost : {}, time: {}'.format(iteration,node,_disc_cost,time.time() - start_time) )
							#lib.plot.plot('NODE {}: time'.format(node), time.time() - start_time)
							#print('iter {} NODE {}: time'.format(iteration,node), time.time() - start_time)

		
		#print('NODE 0',[session.run(gen_params[0][o]).shape for o in range(len(gen_params[0])) ] )
		#print('NODE 1',[session.run(gen_params[1][o]).shape for o in range(len(gen_params[1])) ] )
		if (iteration <= 500 or iteration % 100 == 99):
			dm = session.run(disc_dist)
			gm = session.run(gen_dist)

			gw = session.run(gen_mean_norm)
			dw = session.run(disc_mean_norm)

			# IMPORTANT: second position is the norm of the mean!
			with open('gen_mean.dat','ab') as file:
				file.write(str(iteration)+','+str(gw)+','+','.join([str(g) for g in gm])+'\n')
			with open('disc_mean.dat','ab') as file:
				file.write(str(iteration)+','+str(dw)+','.join([str(d) for d in dm])+'\n')
			print('iter {}  gen_dists : {}'.format(iteration,gm))
			print('iter {}  disc_dists : {}'.format(iteration,dm))

		session.run(combination_op)
		if (iteration % 100 == 99 or iteration < 10):
			print('Time of combination: {}'.format(time.time() - start_time) )
		
		#if (iteration % 100 == 99 or iteration < 10):
		#	#lib.plot.plot('NODE {}: train disc cost'.format(node), _disc_cost)
		#	print('iter {} NODE {}: train disc cost'.format(iteration,nod), _disc_cost)
		#	#lib.plot.plot('NODE {}: time'.format(node), time.time() - start_time)
		#	print('iter {} NODE {}: time'.format(iteration,nod), time.time() - start_time)

		# Calculate inception score every 1K iters
		if iteration % 1000 == 999:
			#inception_score_array = [get_inception_score_node(nod) for nod in NODES]
			inception_score_3 = get_inception_score_3()
			inception_score_6 = get_inception_score_6()
			#lib.plot.plot('NODE 0: inception score', inception_score[0])
			#for nnod in NODES:
			#	print('NODE {}: inception score {}'.format(nnod,inception_score_array[nnod][0]) )
			#	with open('inception_score_dist_{}.dat'.format(nnod),'ab') as file:
			#		file.write(str(iteration)+','+str(inception_score_array[nnod][0])+'\n')
			print('NODE 3: inception score {}'.format(inception_score_3[0]) ) 
			with open('inception_score_dist_3.dat','ab') as file: 
				file.write(str(iteration)+','+str(inception_score_3[0])+'\n') 


			print('NODE 6: inception score {}'.format(inception_score_6[0]) )
			with open('inception_score_dist_6.dat','ab') as file:
				file.write(str(iteration)+','+str(inception_score_6[0])+'\n')


		if iteration % 5000 == 4999:
			save_path = saver.save(session, "/tmp/model.ckpt")
			print("Model saved in file: %s" % save_path)
			generate_image_3(iteration)
			generate_image_6(iteration)

		# Calculate dev loss and generate samples every 100 iters
		#if iteration % 100 == 99:
		#	dev_disc_costs = []
		#	for images,_ in dev_gen[0]():
		#		_dev_disc_cost = session.run(disc_cost[0], feed_dict={real_data_int[0]: images}) 
		#		dev_disc_costs.append(_dev_disc_cost)
			#lib.plot.plot('NODE {}: dev disc cost'.format(node), np.mean(dev_disc_costs))
		#	print('iter {} NODE 0: dev disc cost'.format(iteration), np.mean(dev_disc_costs))
			#generate_image(iteration, _data)

		if (iteration % 100 == 99 or iteration < 10):
			print('Total time: {}'.format(time.time() - start_time) )


		# Save logs every 100 iters
		#if (iteration < 5) or (iteration % 100 == 99):
		#	lib.plot.flush()

		#lib.plot.tick()
