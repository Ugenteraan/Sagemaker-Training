import tensorflow as tf

class model():


	def __init__(self):

		self.num_classes   = 10
		self.learning_rate = 1e-3
		self.image_height  = 28
		self.image_width   = 28
		


	

		self.X = tf.placeholder(tf.float32, shape=(None, self.image_height, self.image_width,1), name='inputs')
		self.Y = tf.placeholder(tf.float32, shape=(None, self.num_classes), name='targets')


		self.dropout = tf.placeholder(tf.float32, name='dropout')


		conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=64, kernel_size=3, stride=2, 
											padding='SAME', activation_fn=tf.nn.leaky_relu)

		conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # size // 4


		output_shape = (self.image_height // 4) * (self.image_width // 4) * 64

		self.feature_vector = tf.reshape(conv1_pool, (-1, output_shape))


		#Weight and bias variables for Fully connected layers
		W1 = tf.Variable(tf.truncated_normal([output_shape, 256], stddev=0.3))
		B1 = tf.Variable(tf.constant(1.0, shape=[256]))
		W2 = tf.Variable(tf.truncated_normal([256, self.num_classes], stddev=0.3))
		B2 = tf.Variable(tf.constant(1.0, shape=[self.num_classes]))

		self.fc1 = tf.add(tf.matmul(self.feature_vector, W1), B1)
		fc1_actv = tf.nn.leaky_relu(self.fc1)

		#dropout
		self.dropout_layer = tf.nn.dropout(fc1_actv, self.dropout)

		self.logits = tf.add(tf.matmul(self.dropout_layer, W2), B2, name='logits')

		Y_pred = tf.nn.softmax(self.logits, name='outputs')

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits),name = 'loss')

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'), name='accuracy')