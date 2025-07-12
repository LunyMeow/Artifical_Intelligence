class Neuron:
    next_id = 0  # Global olarak artan ID deÄŸeri
    
    def __init__(self, default_value:float=0.0, activation_type=None):
        self.value = default_value

        self.id = Neuron.next_id  # Otomatik ID ata
        Neuron.next_id += 1  # Sonraki nÃ¶ron iÃ§in ID artÄ±r
        self.activation_type = activation_type
        self.output = 0.0  # Ã‡Ä±ktÄ± deÄŸeri, aktivasyon fonksiyonundan sonra hesaplanacak
        self.weightedSum = 0

    def activation(self, x):
        if self.activation_type == 'sigmoid':
            x = np.clip(x, -500, 500)  # x'i -500 ile 500 arasÄ±nda tut
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'linear':
        # identity fonksiyon, girdiÄŸi olduÄŸu gibi geÃ§irir
            return x
        elif self.activation_type == "doubleSigmoid":
            x = np.clip(x, -500, 500)
            base_sigmoid = 1 / (1 + np.exp(-x))
            return 3 * base_sigmoid - 1  # AralÄ±k: -1 ila 2


        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def activation_derivative(self):
        if self.activation_type == 'sigmoid':
            # Ã‡Ä±ktÄ±yÄ± 0.01-0.99 aralÄ±ÄŸÄ±na sÄ±kÄ±ÅŸtÄ±r
            safe_output = np.clip(self.output, 0.01, 0.99)
            return safe_output * (1 - safe_output)  # f'(x) = f(x)(1 - f(x))
        elif self.activation_type == 'tanh':
            return 1 - self.output ** 2  # f'(x) = 1 - f(x)^2
        elif self.activation_type == 'linear':
            return 1  # f(x) = x â†’ f'(x) = 1
        elif self.activation_type == "doubleSigmoid":
            scaled_output = (self.output + 1) / 3  # -1 ila 2 â†’ 0 ila 1
            scaled_output = np.clip(scaled_output, 0.01, 0.99)
            return 3 * (scaled_output * (1 - scaled_output))


        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def calculate_weighted_sum(self, layers, connections):
        weighted_sum = 0
        bias_sum = 0  # Bias toplamÄ±
        
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == self.id:
                        
                        weighted_sum += prev_neuron.value * conn.weight
                        bias_sum += conn.bias  # BaÄŸlantÄ± bias'larÄ±nÄ± topla
        
        self.weightedSum = weighted_sum + bias_sum  # Bias'Ä± ekle
        self.value = self.activation(self.weightedSum)
        self.output = self.value
        return self.value




class Connection:
    def __init__(self, weight=0, connectedToArg=[0, 0], bias=0.1):  # VarsayÄ±lan bias=0.1
        self.weight = weight
        self.connectedTo = connectedToArg
        self.bias = bias  # Bias parametresi eklendi
    
    def update_weight(self, learning_rate, delta):
        self.weight += learning_rate * delta
        self.bias += learning_rate * delta * 0.1  # Bias da gÃ¼ncelleniyor


