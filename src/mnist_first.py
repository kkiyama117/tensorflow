from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def mnist_test():
    # mnist object
    mnist = input_data.read_data_sets("data/", one_hot=True)
    # 入力データ定義
    x = tf.placeholder(tf.float32, [None, 784])
    # 入力画像のログを定義
    img = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image("input_data", img, 10)

    with tf.name_scope("hidden"):
        # 入力層 -> 中間層
        # 重み w
        # 初期値 shape を自動生成. stddev は標準偏差
        w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1),
                          name="w1")
        # バイアス b
        # 0 で初期化
        b_1 = tf.Variable(tf.zeros([64]), name="b1")
        # 出力 h (w*x + b)をreLUで活性化
        h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

        # logging
        tf.summary.histogram("w_1", w_1)

    with tf.name_scope("output"):
        # 中間層 -> 出力層
        w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
        b_2 = tf.Variable(tf.zeros([10]), name="b2")
        # SoftMaxで活性化
        out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

    # 誤差を測定
    y = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope("loss"):
        # 誤差関数
        # 二乗誤差を取って, 全ての平均を取る.
        # テンソルの平均値, 第二引数無しの時, 全構成要素の平均を取ります.
        # 第二引数を指定すると軸方向を指定.
        loss = tf.reduce_mean(tf.square(y - out))

    # 訓練
    with tf.name_scope("train"):
        # 確率的勾配降下法
        # 引数は学習率
        # minimizeでアップデート
        # minimize = compute_gradients -> apply_gradients
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # 評価
    with tf.name_scope("accuracy"):
        # 第二引数 は reduce_mean 同様軸方向
        # 1 -> 行方向
        # 結果はバッチサイズと等しい一階テンソル
        correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # init
    init = tf.global_variables_initializer()

    # logging
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    # RUN
    with tf.Session() as session:
        session.run(init)
        # テストデータのロード
        # テスト用の画像データ
        test_images = mnist.test.images
        # テスト用の全正解データ
        test_labels = mnist.test.labels
        summary_writer = tf.summary.FileWriter("logs", session.graph)

        for i in range(1000):
            step = i + 1
            # 全訓練データの取得
            # 訓練用の入力データ, 正解データを取得(ミニバッチ数を指定)
            train_images, train_labels = mnist.train.next_batch(50)
            session.run(train_step,
                        feed_dict={x: train_images, y: train_labels})

            if step % 10 == 0:
                acc_val = session.run(accuracy,
                                      feed_dict={x: test_images,
                                                 y: test_labels})
                print("STEP %d: accuracy = %.3f" % (step, acc_val))
                summary_str = session.run(summary_op,
                                          feed_dict={x: test_images,
                                                     y: test_labels})
                summary_writer.add_summary(summary_str, step)
