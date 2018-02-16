def build_netowrk(mnist):
    data_in = mnist

    with tf.variable_scope("conv1") as scope:
        Z = tf.layers.conv2d(board.board_grid, filters=32, kernel_size=3, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv2") as scope:
        Z = tf.layers.conv2d(tf_get_variable("conv1/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + data_in
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("conv3") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv2/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv4") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv3/A"), filters=32, kernel_size=3, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + tf.get_variable("conv2/A")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("pool1") as scope:
        A = tf.layers.max_pooling2d(tf.get_variable("conv4/A"), pool_size=2, strides=2, padding="VALID")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("conv5") as scope:
        Z = tf.layers.conv2d(tf.get_variable("pool1/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv6") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv5/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + tf.get_variable("pool1/A")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("conv7") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv6/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv8") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv7/A"), filters=64, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + tf.get_variable("conv6/A")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("pool2") as scope:
        A = tf.layers.max_pooling2d(tf.get_variable("conv8/A"), pool_size=2, strides=2, padding="VALID")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("conv9") as scope:
        Z = tf.layers.conv2d(tf.get_variable("pool2/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv10") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv9/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + tf.get_variable("pool2/A")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("conv11") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv10/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv12") as scope:
        Z = tf.layers.conv2d(tf.get_variable("conv11/A"), filters=128, kernel_size=5, strides=1, padding="SAME")
        A = tf.nn.relu(Z)
        A = A + tf.get_variable("conv10/A")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("pool3") as scope:
        A = tf.layers.max_pooling2d(tf.get_variable("conv12/A"), pool_size=2, strides=2, padding="VALID")
        tf.get_variable_scope().reuse_variables()

    with tf.variable_scope("fc") as scope:
        P = tf.contrib.layers.flatten(tf.get_variable("pool3/A"))
        P = tf.nn.relu(P)
        Z = tf.contrib.layers.fully_connected(P, 100)
        A = tf.nn.relu(Z)
        Z = tf.contrib.layers.fully_connected(A, 2)
        return Z
