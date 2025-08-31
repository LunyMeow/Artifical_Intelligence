import torch
import torch.nn.functional as F

print("GPU" if torch.cuda.is_available() else "CPU" + " kullanılıyor...")
class Neuron:
    next_id = 0

    def __init__(self, default_value: float = 0.0, activation_type=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.value = torch.tensor(default_value, dtype=torch.float32, device=device, requires_grad=False)
        self.id = Neuron.next_id
        Neuron.next_id += 1
        self.activation_type = activation_type
        self.output = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.weightedSum = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.device = device

    def activation(self, x):
        if self.activation_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_type == "tanh":
            return torch.tanh(x)
        elif self.activation_type == "linear":
            return x
        elif self.activation_type == "doubleSigmoid":
            return 3 * torch.sigmoid(x) - 1
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def activation_derivative(self):
        if self.activation_type == "sigmoid":
            return self.output * (1 - self.output)
        elif self.activation_type == "tanh":
            return 1 - self.output ** 2
        elif self.activation_type == "linear":
            return torch.tensor(1.0, device=self.device)
        elif self.activation_type == "doubleSigmoid":
            scaled_output = (self.output + 1) / 3
            return 3 * (scaled_output * (1 - scaled_output))
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def calculate_weighted_sum(self, layers, connections):
        weighted_sum = torch.tensor(0.0, device=self.device)
        bias_sum = torch.tensor(0.0, device=self.device)

        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == self.id:
                        weighted_sum += prev_neuron.value * conn.weight
                        bias_sum += conn.bias

        self.weightedSum = weighted_sum + bias_sum
        self.value = self.activation(self.weightedSum)
        self.output = self.value
        return self.value


class Connection:
    def __init__(self, weight=0.0, connectedToArg=[0, 0], bias=0.1, device="cpu"):
        self.weight = torch.tensor(weight, dtype=torch.float32, device=device, requires_grad=False)
        self.connectedTo = connectedToArg
        self.bias = torch.tensor(bias, dtype=torch.float32, device=device, requires_grad=False)
        self.device = device

    def update_weight(self, learning_rate, delta):
        self.weight += learning_rate * delta
        self.bias += learning_rate * delta * 0.1
