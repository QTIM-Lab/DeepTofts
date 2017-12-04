import tensorflow as tf
import numpy as np
import csv

from generate_data import ToySequenceData, ToftsSequenceData, SineSequenceData, DCESequenceData, DCEReconstructionData, ToyPatchData, ToftsPatchData
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

def dynamicRNN(x, seqlen, weights, biases, max_seq_len, num_hidden_lstm = 30):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, max_seq_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_lstm)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden_lstm]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['lstm_out']) + biases['lstm_out']

def stacked_dynamicRNN(x, seqlen, weights, biases, max_seq_len, num_hidden_lstm, num_layers, model_type):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, max_seq_len, 1)

    if model_type == 'rnn':
        cell_fn = tf.contrib.rnn.BasicRNNCell
    elif model_type == 'gru':
        cell_fn = tf.contrib.rnn.GRUCell
    elif model_type == 'lstm':
        cell_fn = tf.contrib.rnn.BasicLSTMCell
    elif model_type == 'nas':
        cell_fn = tf.contrib.rnn.NASCell
    else:
        raise Exception("model type not supported: {}".format(args.model))

    cells = []
    for _ in range(num_layers):
        cell = cell_fn(num_hidden_lstm)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.

    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden_lstm]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['lstm_out']) + biases['lstm_out']

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def timeDistributed_CNN(x, seqlen, weights, biases, max_seq_len, num_hidden_lstm, num_layers, model_type):

    # Input Shape - (batch_size, patch_x, patch_y, max_seq_len, n_inputs)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, patch_x, patch_y, n_inputs)
    # x = tf.unstack(x, max_seq_len, 3)

    # flat_x = tf.reshape(x, shape=[tf.shape(x)[0]*max_seq_len, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[4]])

    flat_x = tf.reshape(x, shape=[tf.shape(x)[0]*max_seq_len, int(x.get_shape()[1]), int(x.get_shape()[2]), int(x.get_shape()[4])])

    print flat_x.get_shape()
    print x.get_shape()

    # Convolution Layer
    conv1 = conv2d(flat_x, weights['conv1'], biases['conv1'])
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv1, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    print fc1.get_shape()

    fc1 = tf.reshape(fc1, shape=[tf.shape(x)[0], int(x.get_shape()[3]), int(fc1.get_shape()[1])])

    print fc1.get_shape()

    return fc1

def basic_NN():

    pass

class Model():

    def __init__(self, max_seq_len=65, patch_x=3, patch_y=3, num_classes=2, cnn_filters=32, cnn_features_out=32, num_hidden_lstm = 50, num_layers = 4, model_type = 'lstm', optimizer_type="regression", n_samples_train_test=[50000, 10000], total_epochs=300000, batch_size=400, display_epoch=10, test_batch_size=10000, load_data=False, old_model=False, train=False, test=False, reconstruct=True, dce_filepath=None, ktrans_filepath=None, ve_filepath=None, output_test_results='results.csv', output_model='model', output_ktrans_filepath='ktrans.nii.gz', output_ve_filepath='ve.nii.gz'):

        self.max_seq_len = max_seq_len
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.num_classes = num_classes
        self.cnn_filters = cnn_filters
        self.cnn_features_out = cnn_features_out
        self.num_hidden_lstm = num_hidden_lstm
        self.num_layers = num_layers

        self.model_type = model_type
        self.optimizer_type = optimizer_type

        self.n_samples_train_test = n_samples_train_test
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.display_epoch = display_epoch

        self.test_batch_size = test_batch_size

        self.load_data = load_data
        self.old_model = old_model
        self.train = train
        self.test = test
        self.reconstruct = reconstruct

        self.dce_filepath = dce_filepath
        self.ktrans_filepath = ktrans_filepath
        self.ve_filepath = ve_filepath

        self.output_test_results = output_test_results
        self.output_model = output_model
        self.output_ktrans_filepath = output_ktrans_filepath
        self.output_ve_filepath = output_ve_filepath

        self.sess = None

    def run_model(self):

        # RNN-CASE
        self.data = tf.placeholder(tf.float32, [None, self.max_seq_len, 1])

        print self.data.get_shape()

        # CNN-RNN-CASE
        # self.data = tf.placeholder(tf.float32, [None, self.patch_x, self.patch_y, self.max_seq_len, 1])

        # print self.data.get_shape()

        self.target = tf.placeholder(tf.float32, [None, self.num_classes])
        self.seqlen = tf.placeholder(tf.int32, [None])

        self.weights = {
            'conv1': tf.Variable(tf.random_normal([3, 3, 1, self.cnn_filters])),
            'fc1': tf.Variable(tf.random_normal([5*5*self.cnn_filters, self.cnn_features_out])),
            'lstm': 'in_tensorflow',
            'lstm_out': tf.Variable(tf.random_normal([self.num_hidden_lstm, self.num_classes]))
        }

        self.biases = {
            'conv1': tf.Variable(tf.random_normal([self.cnn_filters])),
            'fc1': tf.Variable(tf.random_normal([self.cnn_features_out])),
            'lstm': 'in_tensorflow',
            'lstm_out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        # self.prediction = stacked_dynamicRNN(self.data, self.seqlen, self.weights, self.biases, self.max_seq_len, self.num_hidden_lstm, self.num_layers, self.model_type)
        self.prediction = dynamicRNN(self.data, self.seqlen, self.weights, self.biases, self.max_seq_len, self.num_hidden_lstm)
        
        # self.cnn_output = timeDistributed_CNN(self.data, self.seqlen, self.weights, self.biases, self.max_seq_len, self.num_hidden_lstm, self.num_layers, self.model_type)
        # self.prediction = stacked_dynamicRNN(self.cnn_output, self.seqlen, self.weights, self.biases, self.max_seq_len, self.num_hidden_lstm, self.num_layers, self.model_type)


        # print self.cnn_output.get_shape()

        if self.optimizer_type == 'regression':

            self.cost = tf.reduce_mean(tf.pow(self.prediction-self.target, 2))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
            self.metric = tf.reduce_mean(tf.abs(self.prediction-self.target))

        elif self.optimizer_type == 'classification':

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
            self.metric = tf.reduce_mean(tf.cast(self.mistakes, tf.float32)) # Will throw error.

        self.init_op = tf.initialize_all_variables()

        if self.load_data:
            self.load_old_data()

        if self.old_model:
            self.load_old_model()

        if self.train:
            self.train_model()

        if self.test:
            self.test_model()

        if self.reconstruct:
            self.reconstruct_parameter_maps()

        return

    def load_old_data(self):

        # Toy Data
        # self.trainset = ToyPatchData(n_samples=self.n_samples_train_test[0])

        # self.trainset = ToftsPatchData(n_samples=self.n_samples_train_test[0])
        # self.testset = ToftsPatchData(n_samples=self.n_samples_train_test[1])

        # DCE_filepaths = ['/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_PHANTOM.nii.gz', '/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ktrans.nii.gz', '/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ve.nii.gz']

        # Model Data
        self.trainset = ToftsSequenceData(n_samples=self.n_samples_train_test[0], max_seq_len=self.max_seq_len)
        # self.testset = ToftsSequenceData(n_samples=self.n_samples_train_test[1], max_seq_len=self.max_seq_len)

        # Model Data with same params as Phantom
        # self.trainset = ToftsSequenceData(n_samples=self.n_samples_train_test[0], max_seq_len=self.max_seq_len, min_seq_len=self.max_seq_len, ktrans_range=[.001,3], ve_range=[0.01,.99], gaussian_noise=[0,0], T1_range=[800,1200], TR_range=[3,5], flip_angle_degrees_range=[15, 40], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[65,65], time_interval_seconds_range=[3.5, 6.5], injection_start_time_seconds_range=[20,25], T1_blood_range=[1440,1440], baseline_intensity=[1,400])
        # self.testset = ToftsSequenceData(n_samples=self.n_samples_train_test[1], max_seq_len=self.max_seq_len, min_seq_len=self.max_seq_len, ktrans_range=[.001,3], ve_range=[0.01,.99], gaussian_noise=[0,0], T1_range=[800,1200], TR_range=[3,5], flip_angle_degrees_range=[15, 40], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[65,65], time_interval_seconds_range=[3.5, 6.5], injection_start_time_seconds_range=[20,25], T1_blood_range=[1440,1440], baseline_intensity=[1,400])

        # Training Data from Phantom
        # trainset = DCESequenceData(dce_data=DCE_filepaths[0], ktrans_data=DCE_filepaths[1], ve_data=DCE_filepaths[2], n_samples=n_samples_train_test[0])

        self.testset = DCEReconstructionData(dce_data = self.dce_filepath, ktrans_data = self.ktrans_filepath, ve_data = self.ve_filepath,n_samples=self.n_samples_train_test[1])

        # self.testset = DCEReconstructionData(dce_data = '/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_PHANTOM.nii.gz', ktrans_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ktrans.nii.gz', ve_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ve.nii.gz', n_samples=n_samples_train_test[1])

        # testset = DCEReconstructionData(dce_data = '/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1086100996_19040720_PHANTOM.nii.gz', ktrans_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1086100996_19040720_ktrans.nii.gz', ve_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1086100996_19040720_ve.nii.gz', n_samples=n_samples_train_test[1], reconstruct=reconstruct)
        # testset = DCESequenceData(dce_data = '/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_PHANTOM.nii.gz', ktrans_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ktrans.nii.gz', ve_data='/home/anderff/Documents/Data/RIDER_PHANTOMS/QTIM_RIDER_DCE_1023805636_19040901_ve.nii.gz', n_samples=n_samples_train_test[1])

        return

    def load_old_model(self):
        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)

        self.saver.restore(self.sess, self.output_model)     

    def train_model(self):

        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)

        # try:
        epoch = 1
        while epoch * self.batch_size < self.total_epochs:
            batch_x, batch_y, batch_seqlen = self.trainset.next(self.batch_size)

            self.sess.run(self.optimizer, feed_dict={self.data: batch_x, self.target: batch_y, self.seqlen: batch_seqlen})

            if epoch % self.display_epoch == 0:

                acc = self.sess.run(self.metric, feed_dict={self.data: batch_x, self.target: batch_y, self.seqlen: batch_seqlen})

                print("Iter " + str(epoch*self.batch_size) + ", Training Accuracy= " + str(acc))

            epoch += 1
        print("Optimization Finished!")
        # except:
            # print("Optimization Interrupted!")

        self.saver.save(self.sess, self.output_model)

        return

    def test_model(self):

        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)

        # Calculate accuracy
        test_data = self.testset.data
        test_label = self.testset.labels
        test_seqlen = self.testset.seqlen

        preds = self.sess.run(self.prediction, feed_dict={self.data: test_data, self.target: test_label, self.seqlen: test_seqlen})

        with open(self.output_test_results, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for pred_idx, pred in enumerate(preds):
                prediction = [round(x,3) for x in pred.tolist() + test_label[pred_idx]]
                print(prediction)
                writer.writerow(prediction)

        return

    def reconstruct_parameter_maps(self):

        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)

        output_array = np.zeros((self.testset.voxel_count, 2), dtype=float)
        remainder = self.testset.voxel_count % self.test_batch_size
        output_row_idx = 0
        completed = True # This is dumb, come back to it

        while not completed:
            batch_x, batch_seqlen = self.testset.next(test_batch_size)
            preds = self.sess.run(self.prediction, feed_dict={data: batch_x, seqlen: batch_seqlen})

            print preds.shape
            print output_array.shape

            output_array[output_row_idx:min(output_row_idx + test_batch_size, output_row_idx + preds.shape[0]), :] = preds

            output_row_idx += test_batch_size

            if output_array.shape[0] == self.testset.voxel_count:
                completed = True

        ktrans_array = output_array[:,0].reshape(self.testset.data_shape)
        ve_array = output_array[:,1].reshape(self.testset.data_shape)

        # Modfiy reference file to create 3D from 4D.
        save_numpy_2_nifti(ktrans_array, self.ktrans_filepath, output_filepath=self.output_ktrans_filepath)
        save_numpy_2_nifti(ve_array, self.ktrans_filepath, output_filepath=self.output_ve_filepath)

        return

if __name__ == '__main__':
    pass