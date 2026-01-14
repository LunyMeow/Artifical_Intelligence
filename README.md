# 🧠 Adaptive Neural Cortex AI - C++ Implementation

**Demo**: https://artifical-intelligence.onrender.com/

**Credentials**: user: `admin` | pass: `admin1234`

---

## 🚀 Project Overview

This project presents an artificial intelligence system that can dynamically reorganize its own neural structure. Through the **CorticalColumn** class, the AI can optimize its architecture during the learning process, making it more flexible and adaptive compared to traditional fixed-structure models.

The system is implemented in **C++** for high performance and includes both native compilation and **WebAssembly (WASM)** support for web deployment.

---

## 🧩 Key Features

### Dynamic Neural Architecture
- **Self-Adjusting Layers**: The CorticalColumn class allows the AI to adjust neuron and layer counts based on learning progress
- **Pyramid Structure Optimization**: Automatically maintains optimal layer sizes following a pyramid pattern
- **Inactive Neuron Removal**: Identifies and removes underperforming neurons to improve efficiency
- **Adaptive Learning Rate**: Dynamically adjusts learning rate based on error slope analysis

### Multiple Activation Functions
- Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softplus, Linear
- Configurable per model

### Advanced Training Features
- **Error Monitoring**: Tracks training progress and automatically adjusts model capacity
- **Model Checkpointing**: Save and load trained models with embeddings
- **CSV-based Training**: Easy dataset management with automatic format validation
- **Progress Visualization**: Real-time training progress bars

### Embedding Support
- **Word Embeddings**: 50-dimensional word vectors for natural language processing
- **Command Embeddings**: Specialized embeddings for command prediction
- **Dual Tokenization Modes**: 
  - WORD mode: Traditional word-level tokenization
  - SUBWORD mode: Character n-gram based tokenization

### Deployment Options
- **Native C++**: High-performance desktop application
- **WebAssembly**: Browser-based inference without server requirements
- **Interactive CLI**: Command-line interface for training and testing

---

## 🗂️ Project Structure

```
Artifical_Intelligence/
├── Buildv[latest].cpp          # Main neural network implementation
├── LLM/
│   └── Embeddings/
│       ├── command_data.csv      # Command training dataset
│       ├── embeddings.db         # Word embeddings database
│       └── embeddingsForCommands.db  # Command embeddings database
├── Helpers/
│   └── network_changes_logs.py   # Training visualization tools
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies for helpers
```

---

## 🧠 CorticalColumn Class Architecture

The **CorticalColumn** class enables the AI to restructure its neural network during training:

### Network Structure
```cpp
struct Network {
    vector<vector<vector<double>>> weights;  // Layer → Neuron → Connections
    vector<vector<double>> biases;           // Layer → Neuron biases
    string activationType;                   // Activation function
    double learningRate;                     // Adaptive learning rate
    unordered_map<string, vector<float>> wordEmbeddings;
    unordered_map<string, vector<float>> commandEmbeddings;
}
```

### Dynamic Optimization Methods
- `addLayerAt(position, neuronCount)` - Insert hidden layer at specific position
- `removeLayerAt(position)` - Remove layer from network
- `optimizeLayerCount()` - Automatically adjust layer count based on input/output size
- `removeExcessNeuronsFromPyramid()` - Maintain optimal pyramid structure
- `fixLayerImbalance()` - Balance neuron distribution across layers

---

## 🛠️ Installation and Usage

### Prerequisites
- **C++ Compiler**: g++ with C++17 support or Clang
- **For native builds**: 
  - SQLite3 development libraries
  - Python 3.8+ (for helper scripts)
- **For WASM builds**:
  - Emscripten SDK

### Native Compilation

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libsqlite3-dev

# Compile
g++ -o build Buildv1_3_2.cpp ByteBPE/ByteBPETokenizer.cpp -std=c++17 -lsqlite3 -I./include

# Run interactive mode
./neural_cortex
```

### WebAssembly Compilation

```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Compile to WASM
emcc \
  Buildv1_3_2.cpp \
  ByteBPE/ByteBPETokenizer.cpp \
  -O3 \
  -std=c++17 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s EXPORT_NAME=createModule \
  -s EXPORTED_FUNCTIONS="['_load_user_model','_run_inference']" \
  -s EXPORTED_RUNTIME_METHODS="['FS','ccall','cwrap']" \
  -s ASSERTIONS=2 \
  -s SAFE_HEAP=1 \
  -s STACK_OVERFLOW_CHECK=2 \
  -o web/model.js \
  -I./include


# Test WASM build
./neural_cortex --wasm-test
```

---

## 📋 Interactive Commands

```bash
> help                          # Show all commands
> train [error] [epoch]        # Train from CSV (optional: target error, max epochs)
> generate <sentence>          # Predict command from natural language
> save <filename>              # Save model to file
> load <filename>              # Load model from file
> print                        # Display network architecture
> graph                        # Visualize training progress (requires Python)
> test                         # Run XOR test
> terminal <cmd>               # Execute system command (native only)
> exit/quit                    # Exit program
```

---

## 📊 Usage Examples

### Training a Command Prediction Model

```bash
# Start interactive mode
./neural_cortex

# Configure model (or press Enter for defaults)
Model name [command_model]: my_model
Tokenizer Mode [WORD / SUBWORD]: WORD
Training data file [LLM/Embeddings/command_data.csv]: 

# Train the model
> train 0.05 10000

# Test inference
> generate open the browser

# Save trained model
> save models/my_model.bin
```

### Programmatic Usage (C++ API)

```cpp
CorticalColumn cc;
vector<int> topology = {50, 256, 128, 50};  // Input, Hidden, Hidden, Output
cc.addModel("mymodel", topology, "tanh");

// Load embeddings
cc.models["mymodel"].wordEmbeddings = loadEmbeddings("embeddings.db");
cc.models["mymodel"].commandEmbeddings = loadEmbeddings("commands.db");

// Train
trainFromCSV(cc, "mymodel", "data.csv", 50, 50, 0.05, false, 10000);

// Inference
auto sentence_emb = sentence_embedding("open file", embeddings);
auto input = floatToDouble(sentence_emb);
auto output = cc.forward("mymodel", input);
```

### WASM Integration (JavaScript)

```javascript
// Load model
const result = Module.ccall('load_user_model', 'number', 
  ['string', 'string'], 
  ['model.bin', 'model.meta']
);

// Run inference
const command = Module.ccall('run_inference', 'string', 
  ['string'], 
  ['open the browser']
);

console.log('Predicted command:', command);
```

---

## 📈 Model Architecture Details

### Automatic Layer Optimization

The system uses logarithmic scaling to determine optimal hidden layer count:

```cpp
int layers = round(log2(inputSize/outputSize + 1) + log2(inputSize + outputSize)) / 2
```

**Complexity Levels**:
- **Low** (0.6x): Faster training, simpler problems
- **Medium** (1.0x): Balanced performance (default)
- **High** (1.5x): Complex patterns, more capacity

### Pyramid Structure

Layer sizes follow a pyramid pattern optimized for information flow:

```
Input Layer (50) → Hidden (256) → Hidden (128) → Output (50)
```

The middle layers are sized to capture complex patterns while preventing overfitting.

---

## 🔍 Monitoring and Visualization

### Training Metrics
- Real-time error slope analysis
- Automatic learning rate adjustment
- Layer activity monitoring
- Neuron contribution tracking

### Logging
All significant events are logged to `network_changes.log`:
- Layer additions/removals
- Neuron modifications
- Learning rate changes
- Training milestones

### Visualization (Python)
```bash
> graph  # Generates plots from training logs
```

---

## 🎯 Performance Considerations

- **Native C++**: ~10,000 epochs/second (depends on architecture)
- **WASM**: ~2,000 epochs/second (browser-dependent)
- **Memory**: Scales with model size (typical: 10-50 MB)
- **Inference**: < 1ms per prediction (native), < 5ms (WASM)

---

## 📌 Technical Notes

- **C++ Standard**: Requires C++17 or later
- **Dependencies**: 
  - Native: SQLite3 for embedding storage
  - WASM: No external dependencies (embeddings embedded in model)
- **Thread Safety**: Not thread-safe by default (use external synchronization)
- **Numerical Stability**: Includes NaN/Inf checks during training

---

## 🔮 Future Enhancements

- [ ] Multi-threaded training support
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Recurrent layer support (LSTM/GRU)
- [ ] Attention mechanism integration
- [ ] Automatic hyperparameter tuning
- [ ] Distributed training support

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## 📧 Contact

For questions or support, please open an issue on the project repository.
