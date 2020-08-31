# # 3.1 tf.train.Checkpoint 变量的保存与恢复
# import tensorflow as tf
# import numpy as np
# import argparse
# from zh.model.mnist.mlp import MLP
# from zh.model.utils import MNISTLoader
#
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--mode', default='train', help='train or test')
# parser.add_argument('--num_epochs', default=1)
# parser.add_argument('--batch_size', default=50)
# parser.add_argument('--learning_rate', default=0.001)
# args = parser.parse_args()
# data_loader = MNISTLoader()
#
# def train():
#     model = MLP()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
#     num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
#     checkpoint = tf.train.Checkpoint(myAwesomeModel=model)      # 实例化Checkpoint，设置保存对象为model
#     for batch_index in range(1, num_batches+1):
#         X, y = data_loader.get_batch(args.batch_size)
#         with tf.GradientTape() as tape:
#             y_pred = model(X)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#             loss = tf.reduce_mean(loss)
#             print("batch %d: loss %f" % (batch_index, loss.numpy()))
#         grads = tape.gradient(loss, model.variables)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#         if batch_index % 100 == 0:                              # 每隔100个Batch保存一次
#             path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
#             print("model saved to %s" % path)
#
#
# def test():
#     model_to_be_restored = MLP()
#     # 实例化Checkpoint，设置恢复对象为新建立的模型model_to_be_restored
#     checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
#     checkpoint.restore(tf.train.latest_checkpoint('./save'))    # 从文件恢复模型参数
#     y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
#     print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))
#
#
# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     if args.mode == 'test':
#         test()



# # 3.2 TensorBoard: 训练过程可视化
# import tensorflow as tf
# from zh.model.mnist.mlp import MLP
# from zh.model.utils import MNISTLoader
#
# num_batches = 1000
# batch_size = 50
# learning_rate = 0.001
# log_dir = 'tensorboard'
# model = MLP()
# data_loader = MNISTLoader()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# summary_writer = tf.summary.create_file_writer(log_dir)     # 实例化记录器
# tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
# for batch_index in range(num_batches):
#     X, y = data_loader.get_batch(batch_size)
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#         loss = tf.reduce_mean(loss)
#         print("batch %d: loss %f" % (batch_index, loss.numpy()))
#         with summary_writer.as_default():                           # 指定记录器
#             tf.summary.scalar("loss", loss, step=batch_index)       # 将当前损失函数的值写入记录器
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
# with summary_writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)    # 保存Trace信息到文件（可选）



# # 3.3 tf.data ：数据集的构建与预处理
# import tensorflow as tf
# import numpy as np
#
# X = tf.constant([2013, 2014, 2015, 2016, 2017])
# Y = tf.constant([12000, 14000, 15000, 16500, 17500])
#
# # 也可以使用NumPy数组，效果相同
# # X = np.array([2013, 2014, 2015, 2016, 2017])
# # Y = np.array([12000, 14000, 15000, 16500, 17500])
#
# dataset = tf.data.Dataset.from_tensor_slices((X, Y))
#
# for x, y in dataset:
#     print(x.numpy(), y.numpy())
#
# import matplotlib.pyplot as plt
#
# (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
# train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
# mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
#
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()
#
# def rot90(image, label):
#     image = tf.image.rot90(image)
#     return image, label
#
# mnist_dataset = mnist_dataset.map(rot90)
#
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()
#
# mnist_dataset = mnist_dataset.batch(4)
#
# for images, labels in mnist_dataset:    # image: [4, 28, 28, 1], labels: [4]
#     fig, axs = plt.subplots(1, 4)
#     for i in range(4):
#         axs[i].set_title(labels.numpy()[i])
#         axs[i].imshow(images.numpy()[i, :, :, 0])
#     plt.show()
#
# mnist_dataset = mnist_dataset.shuffle(buffer_size=10000).batch(4)
#
# for images, labels in mnist_dataset:
#     fig, axs = plt.subplots(1, 4)
#     for i in range(4):
#         axs[i].set_title(labels.numpy()[i])
#         axs[i].imshow(images.numpy()[i, :, :, 0])
#     plt.show()

# # 实例：cats_vs_dogs 图像分类 1
# import tensorflow as tf
# import os
#
# num_epochs = 10
# batch_size = 32
# learning_rate = 0.001
# data_dir = 'C:/Users/Steve/tensorflow_datasets/cats-vs-dogs/'
# train_cats_dir = data_dir + '/train/cats/'
# train_dogs_dir = data_dir + '/train/dogs/'
# test_cats_dir = data_dir + '/valid/cats/'
# test_dogs_dir = data_dir + '/valid/dogs/'
#
# def _decode_and_resize(filename, label):
#     image_string = tf.io.read_file(filename)            # 读取原始文件
#     image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
#     image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
#     return image_resized, label
#
# if __name__ == '__main__':
#     # 构建训练数据集
#     train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
#     train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
#     train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
#     train_labels = tf.concat([
#         tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
#         tf.ones(train_dog_filenames.shape, dtype=tf.int32)],
#         axis=-1)
#
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
#     train_dataset = train_dataset.map(
#         map_func=_decode_and_resize,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
#     train_dataset = train_dataset.shuffle(buffer_size=23000)
#     train_dataset = train_dataset.batch(batch_size)
#     train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(32, 5, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(2, activation='softmax')
#     ])
#
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=tf.keras.losses.sparse_categorical_crossentropy,
#         metrics=[tf.keras.metrics.sparse_categorical_accuracy]
#     )
#
#     model.fit(train_dataset, epochs=num_epochs)
# # 构建测试数据集
# test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
# test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
# test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
# test_labels = tf.concat([
#     tf.zeros(test_cat_filenames.shape, dtype=tf.int32),
#     tf.ones(test_dog_filenames.shape, dtype=tf.int32)],
#     axis=-1)
#
# test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
# test_dataset = test_dataset.map(_decode_and_resize)
# test_dataset = test_dataset.batch(batch_size)
#
# print(model.metrics_names)
# print(model.evaluate(test_dataset))

# # 实例：cats vs dogs 将数据集存储为 TFRecord 文件
# import tensorflow as tf
# import os
#
# data_dir = 'C:/Users/Steve/tensorflow_datasets/cats-vs-dogs/'
# train_cats_dir = data_dir + '/train/cats/'
# train_dogs_dir = data_dir + '/train/dogs/'
# tfrecord_file = data_dir + '/train/train.tfrecords'
#
# train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
# train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
# train_filenames = train_cat_filenames + train_dog_filenames
# train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)  # 将 cat 类的标签设为0，dog 类的标签设为1
#
# with tf.io.TFRecordWriter(tfrecord_file) as writer:
#     for filename, label in zip(train_filenames, train_labels):
#         image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
#         feature = {                             # 建立 tf.train.Feature 字典
#             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
#             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
#         }
#         example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
#         writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
#
# raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件
#
# feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
#     'image': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.int64),
# }
#
# def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
#     feature_dict = tf.io.parse_single_example(example_string, feature_description)
#     feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
#     return feature_dict['image'], feature_dict['label']
#
# dataset = raw_dataset.map(_parse_example)
#
# import matplotlib.pyplot as plt
#
# for image, label in dataset:
#     plt.title('cat' if label == 0 else 'dog')
#     plt.imshow(image.numpy())
#     plt.show()


# # tf.function ：图执行模式 *
# import tensorflow as tf
# import time
# from zh.model.mnist.cnn import CNN
# from zh.model.utils import MNISTLoader
#
# num_batches = 1000
# batch_size = 50
# learning_rate = 0.001
# data_loader = MNISTLoader()
# model = CNN()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
# @tf.function
# def train_one_step(X, y):
#     with tf.GradientTape() as tape:
#         y_pred = model(X)
#         loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
#         loss = tf.reduce_mean(loss)
#         # 注意这里使用了TensorFlow内置的tf.print()。@tf.function不支持Python内置的print方法
#         tf.print("loss", loss)
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#
# start_time = time.time()
# for batch_index in range(num_batches):
#     X, y = data_loader.get_batch(batch_size)
#     train_one_step(X, y)
# end_time = time.time()
# print(end_time - start_time)
