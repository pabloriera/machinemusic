import tensorflow as tf

class CAE():


    def __init__(self,input_shape, row_bins, col_bins, n_filters, filter_sizes, strides, learning_rate=0.002):


        n_features = row_bins*col_bins

        tf.reset_default_graph()

        # input to the network
        X = tf.placeholder( tf.float32, input_shape, name='x')

        X_tensor = tf.reshape(X, [-1, row_bins, col_bins, 1])

        current_input = X_tensor

        # notice instead of having 784 as our input features, we're going to have
        # just 1, corresponding to the number of channels in the image.
        # We're going to use convolution to find 16 filters, or 16 channels of information in each spatial location we perform convolution at.
        n_input = 1

        # We're going to keep every matrix we create so let's create a list to hold them all
        Ws = []
        shapes = []
        layers_inputs = []

        # We'll create a for loop to create each layer:
        for layer_i, n_output in enumerate(n_filters):
            # just like in the last session,
            # we'll use a variable scope to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                # we'll keep track of the shapes of each layer
                # As we'll need these for the decoder
                shapes.append(current_input.get_shape().as_list())
                layers_inputs.append(current_input)
                # Create a weight matrix which will increasingly reduce
                # down the amount of information in the input by performing
                # a matrix multiplication
                W = tf.get_variable(
                    name='W',
                    shape=[
                        filter_sizes[layer_i],
                        filter_sizes[layer_i],
                        n_input,
                        n_output],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))#, regularizer = tf.contrib.layers.l2_regularizer(l2scale))


                # Now we'll convolve our input by our newly created W matrix
                stri = strides[layer_i]
                h = tf.nn.conv2d(current_input, W,
                    strides=stri, padding='SAME')

                # And then use a relu activation function on its output
                current_input = tf.nn.relu(h)
                

                # Finally we'll store the weight matrix so we can build the decoder.
                Ws.append(W)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                n_input = n_output

        z = current_input

        print('Encoding')
        print('N Filters',n_filters)
        print('Filter sizes',filter_sizes)
        print('Shapes',shapes)

        # %%
        # store the latent representation
        Ws.reverse()
        # and the shapes of each layer
        shapes.reverse()
        # and the number of filters (which is the same but could have been different)
        n_filters.reverse()
        # and append the last filter size which is our input image's number of channels
        n_filters = n_filters[1:] + [1]

        print('Decoding')
        print('N Filters',n_filters)
        print('Filter sizes',filter_sizes)
        print('Shapes',shapes)

        # %%
        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):
            # we'll use a variable scope to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                layers_inputs.append(current_input)
                # Create a weight matrix which will increasingly reduce
                # down the amount of information in the input by performing
                # a matrix multiplication
                W = Ws[layer_i]

                stri = strides[len(shapes)-layer_i-1]
                
                # Now we'll convolve by the transpose of our previous convolution tensor
                h = tf.nn.conv2d_transpose(current_input, W,
                    tf.pack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
                    strides=stri, padding='SAME')

                # And then use a relu activation function on its output
                current_input = tf.nn.relu(h)
                
        layers_inputs.append(current_input)

        # %%
        # now have the reconstruction through the network
        Y = current_input
        Y = tf.reshape(Y, [-1, n_features])

        # cost function measures pixel-wise difference

        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_constant = 0.0  # Choose an appropriate one.
        # reg = reg_constant * tf.reduce_mean(reg_losses)

        reg = 0
        cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1)) + reg

        self.ae = {'X': X, 'z': z, 'Y': Y, 'cost': cost, 'layers_inputs':layers_inputs}

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)    

    def train(self,data,batch_size, n_epochs,display_step=10 ):

            # %%
        # We create a session to use the graph
        config = tf.ConfigProto( device_count = {'GPU': 1} )
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())

        # %%
        # Fit all training data
        self.costs = []
        total_batch = int(data.length/batch_size)
        for epoch_i in range(n_epochs):
            for batch_i in range(total_batch):
                batch_xs = data.next_batch(batch_size)

                self.sess.run(self.optimizer, feed_dict={self.ae['X']: batch_xs })
                cost_value = self.sess.run(self.ae['cost'],  feed_dict={self.ae['X']: batch_xs})
                self.costs.append(cost_value)
                    
            # Display logs per epoch step
            if epoch_i % display_step == 0:
                print("Epoch:", '%04d' % (epoch_i),"cost=", "{:.9f}".format(cost_value))

    def get_aedict(self):
        return self.ae

    def get_session(self):
        return self.sess

    def get_costlist(self):
        return self.costs
