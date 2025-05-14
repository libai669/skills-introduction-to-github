#李雅桐，物联网202402，202406942
import numpy as np


def SimpleNeuralNetwork(input_size, hidden_size, output_size):
    pass


class SimpleNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #随机初始化权重和偏置
        self.weights1 = np.random.randn(self.input_size,self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size,self.output_size)
        self.bias1 = np.random.randn(self.hidden_size)
        self.bias2 = np.random.randn(self.output_size)

    def tanh(self,x):
        return (1-np.exp(-x))/(1+np.exp(-x))
    def tanh_derivative(self,x):
        return 1-x**2

    def feedforward(self,inputs):
        self.input_layer=inputs
        self.hidden_layer_input = np.dot(self.input_layer,self.weights1)+self.bias1
        self.hidden_layer_output = self.tanh(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output,self.weights2)+self.bias2
        self.output_layer_output = self.tanh(self.output_layer_input)

        return self.output_layer_output
    def train(self,inputs,expected_output,epochs=10000,learning_rate=0.1):
        ##训练神经网络
        for epoch in range(epochs):
            #前向传播
            output = self.feedforward(inputs)

            #计算误差
            error = expected_output - output
            output_delta=error*self.tanh_derivative(output)

            #反向求误差梯度
            hidden_layer_error = output_delta.dot(self.weights2.T)
            hidden_layer_delta = hidden_layer_error*self.tanh_derivative(self.hidden_layer_output)

            #更新权重和偏置
            self.weights2 += learning_rate*self.hidden_layer_output.T.dot(output_delta)
            self.bias2 += np.sum(output_delta,axis=0)*learning_rate
            self.weights1 += learning_rate*self.input_layer.T.dot(hidden_layer_delta)
            self.bias1 += np.sum(hidden_layer_delta,axis=0)*learning_rate

            if epoch % 1000 == 0:#每1000次输入一次误差
                print(f'Epoch{epoch}-Error: {np.mean(np.abs(error))}')

    def predict(self,inputs):
        #使用训练好的神经进行预测
        return self.feedforward(inputs)
#实例化神经网络
nn = SimpleNeuralNetwork(input_size=2,hidden_size=3 ,output_size=1)

#创建一个训练数据
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

#训练神经网络
nn.train(inputs,expected_output)
#预测
outputs = nn.predict(inputs)
print("\npredictions after training:")
print(outputs)
