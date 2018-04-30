import tensorflow as tf

a = tf.constant(3, name="const1")
b = tf.Variable(0, name="val1")
# add a,b
add = tf.add(a, b)
# bに結果をassign
assign = tf.assign(b, add)
# 実行時決定の変数
c = tf.placeholder(tf.int32, name="input")
# assign した結果とcを掛け算
mul = tf.multiply(assign, c)
# 各変数の初期化
init = tf.global_variables_initializer()

with tf.Session() as session:
    # 初期化
    session.run(init)
    for i in range(3):
        # 掛け算までのループを実行
        print(session.run(mul, feed_dict={c: 3}))
    for i in range(3):
        # add まで実行
        # 必要な部分以外実行されない
        # bは引き継いでいる
        print(session.run(add))
with tf.Session() as session:
    session.run(init)
    for i in range(3):
        print(session.run(add))
