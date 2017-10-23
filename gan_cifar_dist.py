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

MODE = 'dcgan' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

NODES = range(1)

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

    gen_params.append(lib.params_with_name('Generator{}'.format(i)))
    disc_params.append(lib.params_with_name('Discriminator{}'.format(i)))


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

        gen_train_op.append(tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost[i], var_list=gen_params[i]))
        disc_train_op.append(tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost[i], var_list=disc_params[i]))

    elif MODE == 'dcgan':
        gen_cost.append( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake[i], labels = tf.ones_like(disc_fake[i]))) )
        disc_cost.append( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake[i], labels = tf.zeros_like(disc_fake[i]))) )
        disc_cost[i] += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real[i], labels = tf.ones_like(disc_real[i])))
        disc_cost[i] /= 2.

        gen_train_op.append( tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost[i],
                                                                                      var_list=lib.params_with_name('Generator{}'.format(i))) )
        disc_train_op.append( tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost[i],
                                                                                       var_list=lib.params_with_name('Discriminator{}'.format(i))) )

# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), 'samples_{}.jpg'.format(frame))

# For calculating inception score
samples_100 = Generator(100, index = 0)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen = []
dev_gen = []

for nod in NODES:
    tt, dd = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR, index=nod)
    train_gen.append(tt)
    dev_gen.append(dd)

def inf_train_gen(nod):
    while True:
        for images,_ in train_gen[nod]():
            yield images

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

            lib.plot.plot('NODE {}: train disc cost'.format(node), _disc_cost)
	    #print('NODE {}: train disc cost'.format(i), _disc_cost)
            lib.plot.plot('NODE {}: time'.format(node), time.time() - start_time)
	    #print('NODE {}: time'.format(i), time.time() - start_time)

            # Calculate inception score every 1K iters
            if iteration % 1000 == 999:
                inception_score = get_inception_score()
                lib.plot.plot('NODE 0: inception score', inception_score[0])

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                dev_disc_costs = []
                for images,_ in dev_gen[nod]():
                    _dev_disc_cost = session.run(disc_cost[i], feed_dict={real_data_int[i]: images}) 
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('NODE {}: dev disc cost'.format(node), np.mean(dev_disc_costs))
                #generate_image(iteration, _data)

            # Save logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()
