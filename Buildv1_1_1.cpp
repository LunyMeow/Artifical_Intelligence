#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream> // Dosya işlemleri için gerekli kütüphane
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdlib>


using namespace std;

bool debug = true;
// ---------------------------
// Log yazma fonksiyonu
// ---------------------------
void log_saver(const string &message, string log_file = "network_changes.log")
{
    if (!debug)
    {
        return;
    }

    // Dosyaya ekle (append modunda aç)
    ofstream out(log_file, ios::app);
    if (out.is_open())
    {
        out << message << endl;
        out.close(); // kapatmayı unutma
    }
    else
    {
        cerr << "Log dosyasi acilamadi: " << log_file << endl;
    }
}

class CorticalColumn
{
public:
    struct Network
    {

        vector<double> errors;

        vector<vector<vector<double>>> weights;
        vector<vector<double>> biases;
        string activationType;

        double learningRate = 0.5;

        // Sigmoid ve tanh aktivasyonları
        double activate(double x)
        {
            if (activationType == "tanh")
                return tanh(x);
            return 1.0 / (1.0 + exp(-x));
        }
        double activateDerivative(double x)
        {
            if (activationType == "tanh")
                return 1.0 - tanh(x) * tanh(x);
            double s = 1.0 / (1.0 + exp(-x));
            return s * (1 - s);
        }

        int monitorNetwork()
        {
            /*
            0 : continue
            1 : slope close to zero
            */
            double slope = computeErrorSlope(200);
            if (slope < 0.0001)
            {
                if (debug)
                    cout << "Slope is close to 0 you can stop training" << endl;
                return 1;
            }
            
            return 0;
        }

        void printModelASCII(const string &name = "")
        {
            cout << "=========================\n";
            cout << "Model: " << name << " | Activation: " << activationType << "\n";
            cout << "=========================\n";

            // Katman sayısı: weights.size() + 1 (input katmanı)
            size_t numLayers = weights.size() + 1;

            // Her katmanda en fazla kaç nöron var?
            size_t maxNeurons = 0;
            // input layer nöron sayısı
            if (!weights.empty())
                maxNeurons = weights[0][0].size();
            for (auto &layer : weights)
                for (auto &neuron : layer)
                    if (neuron.size() > maxNeurons)
                        maxNeurons = neuron.size();
            for (auto &layer : weights)
                if (layer.size() > maxNeurons)
                    maxNeurons = layer.size();

            // display matrisi
            vector<vector<string>> display(maxNeurons, vector<string>(numLayers, "   "));

            // Input layer
            size_t inputNeurons = weights[0][0].size();
            for (size_t n = 0; n < inputNeurons; n++)
                display[n][0] = "[I" + to_string(n) + "]"; // I: input

            // Gizli ve çıkış katmanları
            for (size_t l = 0; l < weights.size(); l++)
                for (size_t n = 0; n < weights[l].size(); n++)
                    display[n][l + 1] = "[N" + to_string(n) + "]"; // N: neuron

            // ASCII diyagramı yazdır
            for (size_t r = 0; r < maxNeurons; r++)
            {
                for (size_t c = 0; c < numLayers; c++)
                {
                    cout << display[r][c];
                    if (c != numLayers - 1)
                        cout << "  ";
                }
                cout << "\n";
            }

            // Katmanları yazdır
            for (size_t l = 0; l < weights.size(); l++)
            {
                cout << "Layer " << l << " -> Layer " << (l + 1) << " connections:\n";
                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    cout << "   Neuron " << n << " connections: " << endl;
                    for (size_t k = 0; k < weights[l][n].size(); k++)
                    {
                        cout << "      " << n << "->" << k << ":" << weights[l][n][k] << " " << endl;
                        ;
                    }
                    cout << "     | bias: " << biases[l][n] << "\n";
                }
            }
            cout << "=========================\n";
        }

        // -----------------------------
        // Nöron Ekle
        // -----------------------------
        void addNeuron(size_t layerIndex)
        {
            if (layerIndex >= weights.size())
                return;

            size_t inputSize = weights[layerIndex][0].size();
            vector<double> newNeuronWeights;
            for (size_t i = 0; i < inputSize; i++)
                newNeuronWeights.push_back((rand() % 100) / 100.0 - 0.5);

            weights[layerIndex].push_back(newNeuronWeights);
            biases[layerIndex].push_back((rand() % 100) / 100.0 - 0.5);

            // Eğer sonraki layer varsa, yeni nörona bağlantılar ekle
            if (layerIndex + 1 < weights.size())
            {
                for (size_t n = 0; n < weights[layerIndex + 1].size(); n++)
                {
                    weights[layerIndex + 1][n].push_back((rand() % 100) / 100.0 - 0.5);
                }
            }

            log_saver("Neuron added to layer " + to_string(layerIndex));
        }

        // -----------------------------
        // Nöron Sil
        // -----------------------------
        void removeNeuron(size_t layerIndex, bool preserveWeights = false)
        {
            if (layerIndex >= weights.size() || weights[layerIndex].empty())
                return;

            // Son nöronu çıkarıyoruz
            size_t neuronIndex = weights[layerIndex].size() - 1;
            weights[layerIndex].erase(weights[layerIndex].begin() + neuronIndex);
            biases[layerIndex].erase(biases[layerIndex].begin() + neuronIndex);

            if (!preserveWeights && layerIndex + 1 < weights.size())
            {
                // Sonraki layerdaki bu nörona ait ağırlıkları da sil
                for (size_t n = 0; n < weights[layerIndex + 1].size(); n++)
                {
                    if (weights[layerIndex + 1][n].size() > neuronIndex)
                        weights[layerIndex + 1][n].erase(weights[layerIndex + 1][n].begin() + neuronIndex);
                }
            }

            log_saver("Neuron removed from layer " + to_string(layerIndex));
        }

        // Forward Pass
        vector<double> forward(const vector<double> &inputs)
        {
            vector<double> activations = inputs;
            layerInputs.clear();
            layerOutputs.clear();
            for (size_t l = 0; l < weights.size(); l++)
            {
                vector<double> next;
                vector<double> thisLayerInput;
                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    double sum = biases[l][n];
                    for (size_t w = 0; w < weights[l][n].size(); w++)
                        sum += weights[l][n][w] * activations[w];
                    thisLayerInput.push_back(sum);
                    next.push_back(activate(sum));
                }
                layerInputs.push_back(thisLayerInput);
                layerOutputs.push_back(next);
                activations = next;
            }
            return activations;
        }

        // Backpropagation
        double train(const vector<double> &inputs, const vector<double> &targets, double lr, double targetError)
        {
            vector<double> output = forward(inputs);
            vector<vector<double>> deltas(weights.size());

            // Output layer delta
            deltas.back().resize(output.size());
            double errorSum = 0.0; // Toplam hata için
            for (size_t i = 0; i < output.size(); i++)
            {
                double error = output[i] - targets[i];
                deltas.back()[i] = error * activateDerivative(layerInputs.back()[i]);
                errorSum += error * error; // Kare hata ekle
            }

            // Hidden layers delta
            for (int l = weights.size() - 2; l >= 0; l--)
            {
                deltas[l].resize(weights[l].size());
                for (size_t i = 0; i < weights[l].size(); i++)
                {
                    double sum = 0;
                    for (size_t j = 0; j < weights[l + 1].size(); j++)
                        sum += weights[l + 1][j][i] * deltas[l + 1][j];
                    deltas[l][i] = sum * activateDerivative(layerInputs[l][i]);
                }
            }

            // Ağırlık ve bias güncelle
            vector<double> layerInput;
            for (size_t l = 0; l < weights.size(); l++)
            {
                layerInput = (l == 0) ? inputs : layerOutputs[l - 1];
                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    for (size_t w = 0; w < weights[l][n].size(); w++)
                    {
                        weights[l][n][w] -= lr * deltas[l][n] * layerInput[w];
                    }
                    biases[l][n] -= lr * deltas[l][n];
                }
            }

            double error = errorSum / output.size(); // Ortalama kare hata
            return error;
        }

        double computeErrorSlope(int n = 10)
        {
            int N = errors.size();
            if (N < 2)
                return 0.0;

            // 1️⃣ Son n epoch için basit eğim
            double slopeRecent = 0.0;
            if (N >= n + 1)
            {
                slopeRecent = (errors[N - 1] - errors[N - n - 1]) / n;
            }

            // 2️⃣ Tüm epoch'lar için lineer regresyonla eğim
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            for (int i = 0; i < N; i++)
            {
                sumX += i;
                sumY += errors[i];
                sumXY += i * errors[i];
                sumXX += i * i;
            }
            double slopeOverall = (N * sumXY - sumX * sumY) / (N * sumXX - sumX * sumX);

            // cout << "Son " << n << " epoch ortalama eğimi: " << slopeRecent << endl;
            // cout << "Tüm epoch'lar için genel eğim: " << slopeOverall << endl;

            return slopeOverall; // istersen slopeRecent de dönebilirsin
        }

    private:
        vector<vector<double>> layerInputs;
        vector<vector<double>> layerOutputs;
    };

    unordered_map<string, Network> models;
    string defaultNeuronActivationType = "sigmoid";
    double learningRate = 2;

    void addModel(const string &key, const vector<int> &topology, const string &activation = "sigmoid")
    {
        Network net;
        net.activationType = activation;
        for (size_t i = 1; i < topology.size(); i++)
        {
            vector<vector<double>> layerWeights;
            vector<double> layerBiases;
            for (int n = 0; n < topology[i]; n++)
            {
                vector<double> neuronWeights;
                for (int w = 0; w < topology[i - 1]; w++)
                    neuronWeights.push_back((rand() % 100) / 100.0 - 0.5);
                layerWeights.push_back(neuronWeights);
                layerBiases.push_back((rand() % 100) / 100.0 - 0.5);
            }
            net.weights.push_back(layerWeights);
            net.biases.push_back(layerBiases);
        }
        models[key] = net;
    }

    vector<double> forward(const string &key, const vector<double> &inputs)
    {
        return models[key].forward(inputs);
    }

    double train(const string &key, const vector<double> &inputs, const vector<double> &targets, double lr = -1, double targetError = -1)
    {
        if (lr < 0)
            lr = learningRate; // Eğer kullanıcı bu fonksiyonu çağırırken lr vermezse default kullanılacak learningRate yani
        return models[key].train(inputs, targets, lr, targetError);
    }
};

void progress_bar(int current, int total = 50, int bar_length = 40)
{
    int percent = current * 100 / total;
    int filled = bar_length * percent / 100;

    cout << "\r["; // Satır başına dön
    for (int i = 0; i < filled; i++)
        cout << "#";
    for (int i = filled; i < bar_length; i++)
        cout << "-";
    cout << "] " << percent << "% (" << current << "/" << total << ")";
    cout.flush(); // Anında yazdır
}

int main()
{
    srand(time(nullptr));
    CorticalColumn cc;
    cc.addModel("xor", {2, 3, 1}, "sigmoid");

    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> Y = {{0}, {1}, {1}, {0}};

    // Eğitim
    int epochNum = 2000;
    double targetError = 0.01;
    double currentError = 1;
    // main içindeki eğitim döngüsünde
    for (int epoch = 0; epoch < epochNum; epoch++)
    {
        double epochError = 0.0;

        for (size_t i = 0; i < X.size(); i++)
        {
            double err = cc.train("xor", X[i], Y[i], 2);
            epochError += err;
        }

        epochError /= X.size();                        // epoch ortalaması
        cc.models["xor"].errors.push_back(epochError); // sadece epoch sonunda kaydet
        if (debug)
            log_saver(std::to_string(epochError));
        if (epochError < targetError)
        {
            log_saver("Stopped due targetError...");
            break;
        }

        progress_bar(epoch, epochNum);

        if (epoch + 1 % 200 == 0)
        {
            int value = cc.models["xor"].monitorNetwork();
            if (value == 1)
            {
                if (debug)
                    cout << "Stopped due closing to zero slope..." << endl;
                log_saver("Stopping due to closing zero slope");
                break;
            }
        }
    }

    cout << endl;

    // Test
    for (size_t i = 0; i < X.size(); i++)
    {
        auto out = cc.forward("xor", X[i]);
        cout << "Input: " << X[i][0] << ", " << X[i][1] << " => Output: " << out[0] << endl;
    }

    cc.models["xor"].printModelASCII("xor");

    cin.get();
    system("python3 network_changes_logs.py");
    return 0;
}
