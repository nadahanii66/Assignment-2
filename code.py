class NeuralNetwork:
    def __init__(self):
        self.weights = [
            [0.15, 0.20],  # w1, w2ِ
            [0.25, 0.30],  # w3, w4
            [0.40, 0.45],  # w5, w6
            [0.50, 0.55]   # w7, w8
        ]
        self.biases = [0.35, 0.60]  
        
        
        self.inputs = [0.05, 0.10]
        self.target = [0.01, 0.99]
        
    def sigmoid(self, x):
        return 1 / (1 + (2.71828 ** -x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self):
        self.hidden_inputs = [
            sum(self.inputs[i] * self.weights[j][i] for i in range(2)) + self.biases[0]
            for j in range(2)
        ]
        self.hidden_outputs = [self.sigmoid(x) for x in self.hidden_inputs]
        
        self.final_inputs = [
            sum(self.hidden_outputs[i] * self.weights[j + 2][i] for i in range(2)) + self.biases[1]
            for j in range(2)
        ]
        self.final_outputs = [self.sigmoid(x) for x in self.final_inputs]
        
        return self.final_outputs
    
    def backward_propagation(self, learning_rate=0.5):
        output_errors = [self.target[i] - self.final_outputs[i] for i in range(2)]
        output_gradients = [output_errors[i] * self.sigmoid_derivative(self.final_outputs[i]) for i in range(2)]
        
        hidden_errors = [
            sum(output_gradients[j] * self.weights[j + 2][i] for j in range(2))
            for i in range(2)
        ]
        hidden_gradients = [hidden_errors[i] * self.sigmoid_derivative(self.hidden_outputs[i]) for i in range(2)]
        
        for j in range(2):
            for i in range(2):
                self.weights[j + 2][i] += learning_rate * output_gradients[j] * self.hidden_outputs[i]
                self.weights[j][i] += learning_rate * hidden_gradients[j] * self.inputs[i]
        
        self.biases[1] += learning_rate * sum(output_gradients)
        self.biases[0] += learning_rate * sum(hidden_gradients)
        
        return output_errors

def run():
    nn = NeuralNetwork()
    print("Forward Propagation Output:", nn.forward_propagation())
    print("Backward Propagation Error:", nn.backward_propagation())

run()
