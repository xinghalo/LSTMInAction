from keras.models import Sequential
from keras.layers import Dense, Activation

# 创建线性模型
model = Sequential()

# 添加各种层
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 模型选择器
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 开始训练
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 模型评估
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# 模型应用
classes = model.predict(x_test, batch_size=128)