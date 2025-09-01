#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream> // Dosya islemleri icin gerekli kutuphane
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdlib>
#include <chrono> // <--- ekle

#include <sstream>
#include <algorithm>

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

    // Dosyaya ekle (append modunda ac)
    ofstream out(log_file, ios::app);
    if (out.is_open())
    {
        out << message << endl;
        out.close(); // kapatmayi unutma
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

        int lrChanged = 0;

        double learningRate = 0.5;

        // Sigmoid ve tanh aktivasyonlari
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

        // YENİ: Piramit yapıya göre fazla nöronları sil
        // DÜZELTİLMİŞ: Piramit yapıya göre fazla nöronları sil
        // DÜZELTİLMİŞ: Piramit yapıya göre fazla nöronları sil
        void removeExcessNeuronsFromPyramid()
        {
            // weights.size() = katman sayısı - 1 (sadece bağlantılar)
            // Örnek: 4,99,2 → weights.size() = 2 (hidden→output)

            if (weights.size() < 1)
            { // En az 1 hidden layer olmalı
                cout << "Not enough layers for pyramid optimization" << endl;
                return;
            }

            // Giriş ve çıkış boyutlarını al
            size_t inputSize = (weights.size() > 0) ? weights[0][0].size() : 0;
            size_t outputSize = (weights.size() > 0) ? weights.back().size() : 0;

            if (inputSize == 0 || outputSize == 0)
            {
                cout << "Invalid network structure" << endl;
                return;
            }

            // İdeal piramit yapısını hesapla (SADECE hidden layer'lar için)
            // weights.size() = hidden layer sayısı (4,99,2 → 1 hidden layer)
            vector<size_t> idealSizes = calculatePyramidStructure(inputSize, outputSize, weights.size());

            if (debug)
            {
                cout << "Input size: " << inputSize << ", Output size: " << outputSize << endl;
                cout << "Number of hidden layers: " << weights.size() << endl;
                cout << "Ideal pyramid sizes for hidden layers: ";
                for (size_t i = 0; i < idealSizes.size(); i++)
                {
                    cout << idealSizes[i];
                    if (i < idealSizes.size() - 1)
                        cout << ", ";
                }
                cout << endl;

                cout << "Current hidden layer sizes: ";
                for (size_t i = 0; i < weights.size(); i++)
                {
                    cout << weights[i].size();
                    if (i < weights.size() - 1)
                        cout << ", ";
                }
                cout << endl;
            }

            // Tüm hidden layer'lar için fazla nöronları sil
            for (size_t i = 0; i < weights.size(); i++)
            {
                if (i >= idealSizes.size())
                {
                    if (debug)
                        cout << "Skipping layer " << i << " - no ideal size defined" << endl;
                    continue;
                }

                // Eğer mevcut boyut ideal boyuttan büyükse, fazlalığı sil
                if (weights[i].size() > idealSizes[i])
                {
                    int excess = weights[i].size() - idealSizes[i];

                    if (debug)
                    {
                        cout << "Layer " << i << " has " << excess << " excess neurons (ideal: "
                             << idealSizes[i] << ", current: " << weights[i].size() << ")" << endl;
                    }

                    // En pasif nöronları sil (maksimum 10 nöron)
                    removeMostInactiveNeurons(i, min(excess, 10));
                }
            }
        }

        // YENİ: Belirli bir katmandan en pasif nöronları sil
        void removeMostInactiveNeurons(size_t layerIndex, int numToRemove)
        {
            if (layerIndex >= weights.size() || numToRemove <= 0)
                return;

            // Nöron aktivitelerini hesapla ve sırala
            vector<pair<double, size_t>> neuronActivities;
            for (size_t n = 0; n < weights[layerIndex].size(); n++)
            {
                double activity = calculateNeuronActivity(layerIndex, n);
                neuronActivities.push_back({activity, n});
            }

            // Aktiviteye göre sırala (en pasifler başta)
            sort(neuronActivities.begin(), neuronActivities.end());

            // En pasif nöronları sil (ters sırada sil ki indeksler kaymasın)
            for (int i = numToRemove - 1; i >= 0; i--)
            {
                if (i < neuronActivities.size())
                {
                    size_t neuronIndex = neuronActivities[i].second;
                    if (debug)
                    {
                        cout << "Removing neuron " << neuronIndex << " from layer " << layerIndex
                             << " (activity: " << neuronActivities[i].first << ")" << endl;
                    }
                    removeNeuronAt(layerIndex, neuronIndex, false);
                }
            }
        }

        int monitorNetwork()
        {
            /*
            0 : continue
            1 : slope close to zero lr up
            2 : slope close to zero end train
            3 : neuron removed
            4 : neuron added
            */

            double slope = computeErrorSlope(150);

            // Eğer eğim çok küçük ama hata hala yüksek → kapasite yetmiyor olabilir
            double lastError = errors.empty() ? 9999 : errors.back();

            if (slope != 0.0 && (slope > -0.001 && slope < 0.001))
            {
                if (lrChanged <= 3)
                {
                    learningRate += 0.5;

                    if (debug)
                    {
                        cout << endl
                             << "Slope is close to 0 trying to up lr" << endl;
                        log_saver("lr changed new: " + std::to_string(learningRate));
                    }
                    lrChanged += 1;
                    return 1;
                }
                else if (lrChanged > 3)
                {
                    if (lastError > 0.24) // Hata hala yüksek → kapasiteyi arttır
                    {
                        cout << "debug" << endl;
                        removeExcessNeuronsFromPyramid();

                        // Önce katman dengesizliğini kontrol et ve düzelt
                        if (checkLayerImbalance() && fixLayerImbalance())
                        {
                            cout << "Katmanlar düzeltildi " << endl;
                            return 4; // Katman dengesizliği düzeltildi
                        }

                        // Piramit yapısını koruyarak en uygun katmana nöron ekle
                        size_t targetLayer = findOptimalLayerForAddition();
                        addNeuronToLayer(targetLayer);

                        return 4;
                    }
                    else
                    {
                        if (debug)
                            cout << endl
                                 << "Slope is close to 0 end training" << endl;
                        return 2;
                    }
                }


            }
            else if (slope != 0 && (slope > 0))
            {
                // egim artiyor hata degeri artiyor
                learningRate /= 1.2;
                if (debug)
                    cout << endl
                         << "Slope is positive errors increasing lr reduced" << endl;
                log_saver("Slope is positive errors increasing lr reduced new:" + std::to_string(learningRate));
                lrChanged += 1;
            }

            return 0;
        }

        // YENİ: Piramit yapısı için optimal katmanı bul
        // DÜZELTİLMİŞ: Piramit yapısı için optimal katmanı bul (giriş/çıkış katmanlarına dokunmaz)
        size_t findOptimalLayerForAddition()
        {
            // Sadece hidden layer'ları düşün (giriş ve çıkış katmanlarını atla)
            if (weights.size() <= 2)
            {
                // Sadece 1 hidden layer varsa onu döndür
                return 1; // Hidden layer index 1'de (0: giriş, 1: hidden, 2: çıkış)
            }

            // Giriş ve çıkış boyutlarını al
            size_t inputSize = weights[0][0].size();
            size_t outputSize = weights.back().size();

            // İdeal piramit yapısını hesapla (sadece hidden layer'lar için)
            vector<size_t> idealSizes = calculatePyramidStructure(inputSize, outputSize, weights.size() - 2);

            // Mevcut boyutlarla ideal boyutları karşılaştır, en çok eksiği olan HIDDEN katmanı bul
            size_t optimalLayer = 1; // Varsayılan olarak ilk hidden layer
            double maxDeficit = 0.0;

            // Sadece hidden layer'ları kontrol et (index 1'den weights.size()-2'ye kadar)
            for (size_t i = 1; i < weights.size() - 1; i++)
            {
                // idealSizes indeksini ayarla (hidden layer'lar 0'dan başlar)
                size_t idealIndex = i - 1;
                if (idealIndex >= idealSizes.size())
                    break;

                double deficit = static_cast<double>(idealSizes[idealIndex]) - static_cast<double>(weights[i].size());
                if (deficit > maxDeficit)
                {
                    maxDeficit = deficit;
                    optimalLayer = i;
                }
            }

            // Eğer hiç eksiği olan hidden layer yoksa, en küçük hidden layer'ı bul
            if (maxDeficit <= 0)
            {
                return findSmallestHiddenLayer();
            }

            return optimalLayer;
        }

        // YENİ: En küçük HIDDEN katmanı bul (giriş/çıkış katmanlarını atlar)
        size_t findSmallestHiddenLayer()
        {
            if (weights.size() <= 2)
            {
                return 1; // Varsayılan hidden layer index
            }

            size_t smallestLayer = 1;
            size_t minSize = weights[1].size();

            // Sadece hidden layer'ları kontrol et (index 1'den weights.size()-2'ye kadar)
            for (size_t i = 1; i < weights.size() - 1; i++)
            {
                if (weights[i].size() < minSize)
                {
                    minSize = weights[i].size();
                    smallestLayer = i;
                }
            }
            return smallestLayer;
        }

        // DÜZELTİLMİŞ: Piramit yapısını hesapla (sadece hidden layer'lar için)
        vector<size_t> calculatePyramidStructure(size_t inputSize, size_t outputSize, size_t numHiddenLayers)
        {
            vector<size_t> pyramidSizes;

            if (numHiddenLayers == 0)
                return pyramidSizes;

            if (numHiddenLayers == 1)
            {
                // 4,3,2 → hidden: 3
                size_t middle = max((inputSize + outputSize) / 2, (size_t)2); // En az 2 nöron
                pyramidSizes.push_back(middle);
            }
            else if (numHiddenLayers == 2)
            {
                // 128,216,32,10 → hidden: 216,32
                size_t peak = static_cast<size_t>(sqrt(inputSize * outputSize) * 1.2); // 1.5 → 1.2
                size_t middle = max((inputSize + outputSize) / 2, (size_t)2);
                pyramidSizes.push_back(peak);
                pyramidSizes.push_back(middle);
            }
            else
            {
                // 128,192,192,128 → hidden: 192,192
                for (size_t i = 0; i < numHiddenLayers; i++)
                {
                    double ratio = static_cast<double>(i + 1) / static_cast<double>(numHiddenLayers + 1);
                    double size = inputSize + (outputSize - inputSize) * ratio;

                    if (i == numHiddenLayers / 2)
                    {
                        size *= 1.3; // 1.5 → 1.3
                    }
                    pyramidSizes.push_back(max(static_cast<size_t>(size), (size_t)2)); // En az 2 nöron
                }
            }

            if (debug)
            {
                cout << "Pyramid sizes for " << numHiddenLayers << " hidden layers: ";
                for (size_t i = 0; i < pyramidSizes.size(); i++)
                {
                    cout << pyramidSizes[i];
                    if (i < pyramidSizes.size() - 1)
                        cout << ", ";
                }
                cout << endl;
            }

            return pyramidSizes;
        }
        // YENİ: Sadece gerçekten pasif nöronları kaldır
        void removeOnlyTrulyInactiveNeurons(double threshold = 2)
        {
            // Tüm katmanlarda dolaş ve gerçekten pasif olan nöronları kaldır
            for (size_t layerIndex = 0; layerIndex < weights.size(); layerIndex++)
            {
                // Ters sırayla gitmek daha güvenli
                for (int n = weights[layerIndex].size() - 1; n >= 0; n--)
                {
                    double neuronActivity = calculateNeuronActivity(layerIndex, n);

                    if (neuronActivity < threshold)
                    {
                        if (debug)
                        {
                            cout << "Removing truly inactive neuron " << n << " from layer " << layerIndex
                                 << " (activity: " << neuronActivity << ")" << endl;
                        }
                        removeNeuronAt(layerIndex, n, false);
                    }
                }
            }
        }

        // YENİ: Nöron aktivitesini hesapla (daha karmaşık metrik)
        // Nöron aktivite hesaplamasını iyileştir
        double calculateNeuronActivity(size_t layerIndex, size_t neuronIndex)
        {
            if (layerIndex >= weights.size() || neuronIndex >= weights[layerIndex].size())
            {
                return 1.0;
            }

            // 1. Ağırlık varyasyonunu hesapla (standart sapma benzeri)
            double sum = 0.0;
            double sumSq = 0.0;
            for (double w : weights[layerIndex][neuronIndex])
            {
                sum += w;
                sumSq += w * w;
            }
            double mean = sum / weights[layerIndex][neuronIndex].size();
            double variance = (sumSq / weights[layerIndex][neuronIndex].size()) - (mean * mean);

            // 2. Bias'ın etkisi
            double biasEffect = fabs(biases[layerIndex][neuronIndex]);

            // 3. Aktivasyon = varyasyon * bias (nöronun "ilginçliği")
            return sqrt(fabs(variance)) * (1.0 + biasEffect);
        }

        bool fixLayerImbalance()
        {
            if (!checkLayerImbalance())
            {
                cout << "WTFFF" << endl;
                return false;
            }
            vector<size_t> idealSizes = calculatePyramidStructure(
                weights[0][0].size(),
                weights.back().size(),
                weights.size() - 2); // ← Sadece hidden layer'lar

            bool changed = false;

            // Sadece hidden layer'ları düzelt (index 1'den weights.size()-2'ye kadar)
            for (size_t i = 1; i < weights.size() - 1; i++)
            {
                size_t idealIndex = i - 1;
                if (idealIndex >= idealSizes.size())
                    break;

                int deficit = idealSizes[idealIndex] - weights[i].size();

                if (deficit > 0)
                {
                    for (int j = 0; j < min(deficit, 3); j++)
                    { // Maksimum 3 nöron
                        addNeuronToLayer(i);
                        changed = true;
                    }
                }
            }

            return changed;
        }

        // DEĞİŞTİRİLDİ: Katman dengesizliği kontrolü (piramit yapıya göre)
        bool checkLayerImbalance(double threshold = 1.5)
        {
            if (weights.size() < 2)
            {
                log_saver("Something went wrong");
                return false;
            }

            vector<size_t> idealSizes = calculatePyramidStructure(
                weights[0][0].size(),
                weights.back().size(),
                weights.size());

            for (size_t i = 0; i < weights.size(); i++)
            {
                double ratio = static_cast<double>(idealSizes[i]) /
                               static_cast<double>(max(1, (int)weights[i].size()));

                if (ratio > threshold || ratio < 1.0 / threshold)
                {
                    return true;
                }
            }

            return false;
        }

        void addNeuronToLayer(size_t layerIndex)
        {
            // GİRİŞ ve ÇIKIŞ katmanlarına eklemeyi ENGELLE
            if (layerIndex >= weights.size() || layerIndex == 0 || layerIndex == weights.size() - 1)
            {
                log_saver("ERROR: Cannot add neurons to input or output layers!");
                return;
            }
            if (layerIndex >= weights.size())
            {
                log_saver("Something went wrong");
                return;
            }
            // Eğer layer boşsa 1 nöron ekle ve weights ve bias'ı başlat
            if (weights[layerIndex].empty())
            {
                size_t inputSize = (layerIndex == 0) ? layerInputs.size() : weights[layerIndex - 1].size();
                vector<double> newNeuronWeights(inputSize);
                for (size_t i = 0; i < inputSize; i++)
                    newNeuronWeights[i] = ((rand() % 100) / 100.0 - 0.5);

                weights[layerIndex].push_back(newNeuronWeights);
                biases[layerIndex].push_back(((rand() % 100) / 100.0 - 0.5));
            }
            else
            {
                size_t inputSize = weights[layerIndex][0].size();
                vector<double> newNeuronWeights(inputSize);
                for (size_t i = 0; i < inputSize; i++)
                    newNeuronWeights[i] = ((rand() % 100) / 100.0 - 0.5);

                weights[layerIndex].push_back(newNeuronWeights);
                biases[layerIndex].push_back(((rand() % 100) / 100.0 - 0.5));
            }

            // Sonraki layer varsa ve nöron sayısı > 0 ise bağlantıları güncelle
            if (layerIndex + 1 < weights.size() && !weights[layerIndex + 1].empty()) // ADDED: Check if next layer is not empty
            {
                for (size_t n = 0; n < weights[layerIndex + 1].size(); n++)
                {
                    weights[layerIndex + 1][n].push_back(((rand() % 100) / 100.0 - 0.5));
                }
            }
            // Eğer bir sonraki layer tamamen boşsa, onu da initialize et
            else if (layerIndex + 1 < weights.size() && weights[layerIndex + 1].empty())
            {
                // Initialize the next layer with at least one neuron
                size_t inputSize = weights[layerIndex].size(); // Input size is now the number of neurons we just added to
                vector<double> newNeuronWeights(inputSize);
                for (size_t i = 0; i < inputSize; i++)
                    newNeuronWeights[i] = ((rand() % 100) / 100.0 - 0.5);

                weights[layerIndex + 1].push_back(newNeuronWeights);
                biases[layerIndex + 1].push_back(((rand() % 100) / 100.0 - 0.5));
            }
            // Clear cached values since network structure changed
            layerInputs.clear();
            layerOutputs.clear();

            log_saver("Neuron added to layer " + to_string(layerIndex));
        }
        // -----------------------------
        // Islevsiz Noron Kontrol ve Kaldir
        // -----------------------------
        void removeInactiveNeurons(size_t layerIndex, double threshold = 0.05, bool preserveWeights = false)
        {
            if (layerIndex >= weights.size())
            {
                log_saver("Something went wrong");
                return;
            }
            // Ters sirayla gitmek daha guvenli
            for (int n = weights[layerIndex].size() - 1; n >= 0; n--)
            {
                double sumAbsWeights = 0.0;
                for (double w : weights[layerIndex][n])
                    sumAbsWeights += fabs(w);

                double avgWeight = sumAbsWeights / weights[layerIndex][n].size();

                if (avgWeight < threshold)
                {
                    if (debug)
                        cout << "Removing inactive neuron " << n << " from layer " << layerIndex
                             << " (avg weight: " << avgWeight << ")" << endl;

                    removeNeuronAt(layerIndex, n, preserveWeights);
                }
            }
        }

        void printModelASCII(const string &name = "")
        {
            cout << "=========================\n";
            cout << "Model: " << name << " | Activation: " << activationType << "\n";
            cout << "=========================\n";

            // Katman sayisi: weights.size() + 1 (input katmani)
            size_t numLayers = weights.size() + 1;

            // Her katmanda en fazla kac noron var?
            size_t maxNeurons = 0;
            // input layer noron sayisi
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

            // Gizli ve cikis katmanlari
            for (size_t l = 0; l < weights.size(); l++)
                for (size_t n = 0; n < weights[l].size(); n++)
                    display[n][l + 1] = "[N" + to_string(n) + "]"; // N: neuron

            // ASCII diyagrami yazdir
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

            // Katmanlari yazdir
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
        // Noron Sil
        // -----------------------------

        void removeNeuron(size_t layerIndex, bool preserveWeights = false)
        {
            if (layerIndex >= weights.size() || weights[layerIndex].empty())
            {
                cout << "Sething is not right" << endl;
                log_saver("Something is not right");
                return;
            }

            // Son noronu cikariyoruz
            size_t neuronIndex = weights[layerIndex].size() - 1;
            weights[layerIndex].erase(weights[layerIndex].begin() + neuronIndex);
            biases[layerIndex].erase(biases[layerIndex].begin() + neuronIndex);

            if (!preserveWeights && layerIndex + 1 < weights.size())
            {
                // Sonraki layerdaki bu norona ait agirliklari da sil
                for (size_t n = 0; n < weights[layerIndex + 1].size(); n++)
                {
                    if (weights[layerIndex + 1][n].size() > neuronIndex)
                        weights[layerIndex + 1][n].erase(weights[layerIndex + 1][n].begin() + neuronIndex);
                }
            }

            // Clear cached values since network structure changed
            layerInputs.clear();
            layerOutputs.clear();

            log_saver("Neuron removed from layer " + to_string(layerIndex));
        }

        // -----------------------------
        // Index ile Noron Sil
        // -----------------------------
        void removeNeuronAt(size_t layerIndex, size_t neuronIndex, bool preserveWeights = false)
        {
            if (layerIndex >= weights.size() || neuronIndex >= weights[layerIndex].size())
            {
                log_saver("Something is not right again");
                return;
            }
            weights[layerIndex].erase(weights[layerIndex].begin() + neuronIndex);
            biases[layerIndex].erase(biases[layerIndex].begin() + neuronIndex);

            if (!preserveWeights && layerIndex + 1 < weights.size())
            {
                for (size_t n = 0; n < weights[layerIndex + 1].size(); n++)
                {
                    if (weights[layerIndex + 1][n].size() > neuronIndex)
                        weights[layerIndex + 1][n].erase(weights[layerIndex + 1][n].begin() + neuronIndex);
                }
            }

            // Clear cached values since network structure changed
            layerInputs.clear();
            layerOutputs.clear();

            log_saver("Spesificed Neuron removed from layer " + to_string(layerIndex) + ", index " + to_string(neuronIndex));
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
        double train(const vector<double> &inputs, const vector<double> &targets, double targetError)
        {
            vector<double> output = forward(inputs);
            vector<vector<double>> deltas(weights.size());

            // Output layer delta
            deltas.back().resize(output.size());
            double errorSum = 0.0; // Toplam hata icin
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

            // Agirlik ve bias guncelle
            vector<double> layerInput;
            for (size_t l = 0; l < weights.size(); l++)
            {
                layerInput = (l == 0) ? inputs : layerOutputs[l - 1];
                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    for (size_t w = 0; w < weights[l][n].size(); w++)
                    {
                        weights[l][n][w] -= learningRate * deltas[l][n] * layerInput[w];
                    }
                    biases[l][n] -= learningRate * deltas[l][n];
                }
            }

            double error = errorSum / output.size(); // Ortalama kare hata
            return error;
        }

        double computeErrorSlope(int n = 10)
        {
            int N = errors.size();
            if (N < n)
                return 0.0;

            // 1️⃣ Son n epoch icin basit egim
            double slopeRecent = 0.0;
            if (N >= n + 1)
            {
                slopeRecent = (errors[N - 1] - errors[N - n - 1]) / n;
            }

            // 2️⃣ Tum epoch'lar icin lineer regresyonla egim
            double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            for (int i = 0; i < N; i++)
            {
                sumX += i;
                sumY += errors[i];
                sumXY += i * errors[i];
                sumXX += i * i;
            }
            double slopeOverall = (N * sumXY - sumX * sumY) / (N * sumXX - sumX * sumX);

            // cout << "Son " << n << " epoch ortalama egimi: " << slopeRecent << endl;
            // cout << "Tum epoch'lar icin genel egim: " << slopeOverall << endl;

            return slopeOverall; // istersen slopeRecent de donebilirsin
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

    double train(const string &key, const vector<double> &inputs, const vector<double> &targets, double targetError = -1)
    {
        // if (lr < 0)
        //     lr = learningRate; // Eger kullanici bu fonksiyonu cagirirken lr vermezse default kullanilacak learningRate yani
        return models[key].train(inputs, targets, targetError);
    }
};

void progress_bar(int current, int total = 50, int bar_length = 40)
{
    int percent = current * 100 / total;
    int filled = bar_length * percent / 100;

    cout << "\r["; // Satir basina don
    for (int i = 0; i < filled; i++)
        cout << "#";
    for (int i = filled; i < bar_length; i++)
        cout << "-";
    cout << "] " << percent << "% (" << current << "/" << total << ")";
    cout.flush(); // Aninda yazdir
}

struct DatasetInfo
{
    int inputSize;
    int outputSize;
};

// CSV basligina bakip input/output sayisini otomatik belirle
DatasetInfo getDatasetInfo(const string &filename)
{
    ifstream file(filename);
    string line;
    DatasetInfo info = {0, 0};

    if (getline(file, line))
    {
        stringstream ss(line);
        string cell;
        vector<string> headers;
        while (getline(ss, cell, ','))
            headers.push_back(cell);

        for (auto &h : headers)
        {
            string lower;
            lower.resize(h.size());
            transform(h.begin(), h.end(), lower.begin(), ::tolower);
            if (lower.find("input") != string::npos)
                info.inputSize++;
            else if (lower.find("target") != string::npos)
                info.outputSize++;
        }
    }
    return info;
}

// CSV satirini okuyup X/Y vektorlerine cevirir
bool readCSVLine(ifstream &file, vector<double> &x, vector<double> &y, int inputSize, int outputSize)
{
    string line;
    if (!getline(file, line))
        return false;
    if (line.empty())
        return readCSVLine(file, x, y, inputSize, outputSize);

    stringstream ss(line);
    string cell;
    vector<double> row;
    while (getline(ss, cell, ','))
        row.push_back(stod(cell));

    if (row.size() != inputSize + outputSize)
        return false;

    x.assign(row.begin(), row.begin() + inputSize);
    y.assign(row.begin() + inputSize, row.end());
    return true;
}

void testModelFromCSV(CorticalColumn &cc, const string &modelKey, const string &testFile)
{
    ifstream file(testFile);
    if (!file.is_open())
    {
        cerr << "Test dosyasi acilamadi: " << testFile << endl;
        return;
    }

    string header;
    getline(file, header); // basligi atla

    int inputSize = 0, outputSize = 0;
    stringstream ss(header);
    string cell;
    vector<string> headers;
    while (getline(ss, cell, ','))
        headers.push_back(cell);
    for (auto &h : headers)
    {
        string lower;
        lower.resize(h.size());
        transform(h.begin(), h.end(), lower.begin(), ::tolower);
        if (lower.find("input") != string::npos)
            inputSize++;
        else if (lower.find("target") != string::npos)
            outputSize++;
    }

    vector<double> x, y;
    int totalSamples = 0;
    double totalAccuracy = 0.0;

    while (readCSVLine(file, x, y, inputSize, outputSize))
    {
        auto out = cc.forward(modelKey, x);

        cout << "Input: ";
        for (double v : x)
            cout << v << " ";
        cout << " => Output: ";
        for (double v : out)
            cout << v << " ";
        cout << " | Target: ";
        for (double v : y)
            cout << v << " ";
        cout << endl;

        // Ortalama dogruluk: her cikisin hedefe yakinligi
        double sampleAccuracy = 0.0;
        for (size_t i = 0; i < y.size(); i++)
        {
            double diff = abs(out[i] - y[i]);
            sampleAccuracy += (1.0 - diff); // 1 fark sifirsa tam dogru, 0 fark 1 olursa tamamen yanlis
        }
        sampleAccuracy /= y.size(); // cikis basina ortalama
        totalAccuracy += sampleAccuracy;
        totalSamples++;
    }

    double finalAccuracy = (totalSamples > 0) ? (totalAccuracy / totalSamples * 100.0) : 0.0;
    cout << "\nModel Ortalama Dogruluk: " << finalAccuracy << "% (" << totalSamples << " ornek uzerinden)" << endl;
}

void trainFromCSV(CorticalColumn &cc, const string &modelKey, const string &csvFile,
                  int inputSize, int outputSize, double targetError, bool isNewData = false)
{
    string message = isNewData ? "Yeni ornek eklendi, yeniden egitim basliyor..."
                               : "CSV'deki tum verilerle yeniden egitim basliyor...";
    cout << message << endl;

    double avgError = 1;
    int maxEpoch = 500;

    for (int epoch = 0; epoch < maxEpoch; epoch++)
    {
        ifstream file(csvFile);
        if (!file.is_open())
        {
            cerr << "CSV dosyasi acilamadi: " << csvFile << endl;
            break;
        }

        string header;
        getline(file, header);

        vector<double> x, y;
        double totalError = 0.0;
        int count = 0;

        while (readCSVLine(file, x, y, inputSize, outputSize))
        {
            totalError += cc.train(modelKey, x, y, targetError);
            count++;
        }

        avgError = totalError / max(1, count);

        progress_bar(epoch, maxEpoch);

        // Model reboot mekanizması
        if ((epoch + 1) % 20 == 0)
        {
            int monitor = cc.models[modelKey].monitorNetwork();
            if (monitor == 4) // neuron added - reboot needed
            {
                // ÖNCE: Mevcut hatayı ve öğrenme oranını kaydet
                double currentLR = cc.models[modelKey].learningRate;
                vector<double> currentErrors = cc.models[modelKey].errors;

                // Modeli tamamen yeniden oluştur
                vector<int> currentTopology;
                currentTopology.push_back(cc.models[modelKey].weights[0][0].size()); // input size

                for (const auto &layer : cc.models[modelKey].weights)
                {
                    currentTopology.push_back(layer.size());
                }

                // ESKİ modeli sil ve YENİSİNİ oluştur
                cc.models.erase(modelKey);
                cc.addModel(modelKey, currentTopology, "sigmoid");

                // Öğrenme oranını ve hata geçmişini geri yükle
                cc.models[modelKey].learningRate = currentLR;
                cc.models[modelKey].errors = currentErrors;

                cout << "Model rebooted with new topology: ";
                for (size_t i = 0; i < currentTopology.size(); i++)
                {
                    cout << currentTopology[i];
                    if (i < currentTopology.size() - 1)
                        cout << ",";
                }
                cout << endl;
                log_saver("Model rebooted");

                epoch = -1; // restart training (next iteration will be epoch 0)
                continue;   // bu epoch'u atla
            }
        }

        // Hata hedefine ulaşıldı mı kontrol et
        if ((isNewData && avgError < targetError) || (!isNewData && avgError < targetError))
        {
            log_saver("Stopped due to target error...");
            break;
        }

        if (debug)
        {
            cc.models[modelKey].errors.push_back(avgError);
            log_saver(to_string(avgError));
        }
    }

    cout << "Egitim tamamlandi. Ortalama hata: " << avgError << endl;
}

// Kullanicidan surekli veri al, CSV'ye yaz ve modeli egit
void interactiveTraining(CorticalColumn &cc, const string &modelKey, const string &csvFile,
                         int inputSize, int outputSize, double targetError = 0.05)
{
    cout << "\n==== Etkilesimli Egitim Modu ====\n";
    cout << "Format: input1,input2,...,inputN - output1,output2,...,outputM\n";
    cout << "Eger '-' koymazsan sadece input forward edilir.\n";
    cout << "Cikmak icin 'exit' yaz.\n\n";

    while (true)
    {
        cout << "> ";
        string line;
        getline(cin, line);

        if (line == "exit" || line == "quit")
            break;
        else if (line == "graph")
        {
            system("python3 Helpers/network_changes_logs.py");
            continue;
        }
        else if (line == "train")
        {
            line = "train";
        }
        else if (line == "print")
        {
            cc.models[modelKey].printModelASCII();
            continue;
        }
        if (line.empty())
            continue;

        // Input ve outputlari ayir
        vector<double> inputs, targets;
        size_t dashPos = line.find('-');
        string inputsPart = (dashPos != string::npos) ? line.substr(0, dashPos) : line;
        string targetsPart = (dashPos != string::npos) ? line.substr(dashPos + 1) : "";

        // Inputlari parse et
        stringstream ssIn(inputsPart);
        string token;
        if (line != "train")
        {
            while (getline(ssIn, token, ','))
            {
                try
                {
                    inputs.push_back(stod(token));
                }
                catch (...)
                {
                    cerr << "Gecersiz input: " << token << endl;
                }
            }

            // Target varsa parse et
            if (!targetsPart.empty())
            {
                stringstream ssOut(targetsPart);
                while (getline(ssOut, token, ','))
                {
                    try
                    {
                        targets.push_back(stod(token));
                    }
                    catch (...)
                    {
                        cerr << "Gecersiz target: " << token << endl;
                    }
                }
            }

            if (inputs.size() != (size_t)inputSize)
            {
                cerr << "Beklenen input boyutu: " << inputSize << ", ama girdin: " << inputs.size() << endl;
                continue;
            }

            if (!targets.empty() && targets.size() != (size_t)outputSize)
            {
                cerr << "Beklenen output boyutu: " << outputSize << ", ama girdin: " << targets.size() << endl;
                continue;
            }
        }

        // CSV’ye kaydet
        if (!targets.empty() && line != "train")
        {
            bool fileExists = ifstream(csvFile).good();
            ofstream out(csvFile, ios::app);
            if (!out.is_open())
            {
                cerr << "CSV dosyasi acilamadi: " << csvFile << endl;
                continue;
            }

            // Ilk defa aciliyorsa header yaz
            if (!fileExists)
            {
                for (int i = 0; i < inputSize; i++)
                    out << "input" << i << ",";
                for (int i = 0; i < outputSize; i++)
                {
                    out << "target" << i;
                    if (i != outputSize - 1)
                        out << ",";
                }
                out << "\n";
            }

            // Satir yaz
            for (size_t i = 0; i < inputs.size(); i++)
                out << inputs[i] << ",";
            for (size_t i = 0; i < targets.size(); i++)
            {
                out << targets[i];
                if (i != targets.size() - 1)
                    out << ",";
            }
            out << "\n";
            out.close();
        }

        // Ana kodda kullanım:
        if (line == "train")
        {
            trainFromCSV(cc, modelKey, csvFile, inputSize, outputSize, targetError, false);
            continue;
        }

        // Eger target verilmisse egit
        if (!targets.empty())
        {
            trainFromCSV(cc, modelKey, csvFile, inputSize, outputSize, targetError, true);
        }

        // Forward sonucu goster
        auto result = cc.forward(modelKey, inputs);
        cout << "Model ciktisi: ";
        for (double v : result)
            cout << v << " ";
        cout << endl;

        // Eger target vardiysa dogruluk kontrol et
        if (!targets.empty())
        {
            cout << "Beklenen cikti: ";
            for (double v : targets)
                cout << v << " ";
            cout << endl;
        }
    }
}

/*
int main()
{
    srand(time(nullptr));
    CorticalColumn cc;

    string filename = "parity_problem.csv";
    DatasetInfo info = getDatasetInfo(filename);

    // Modeli CSV'ye gore otomatik boyutlandir
    vector<int> layers = {info.inputSize, 24, 6, info.outputSize}; // gizli noronlar
    cc.addModel("parity", layers, "sigmoid");

    int epochNum = 2000;
    double targetError = 0.001;

    // Baslangic zamani
    auto startTime = chrono::high_resolution_clock::now();

    double lastEpochError = 0.0;

    for (int epoch = 0; epoch < epochNum; epoch++)
    {
        double epochError = 0.0;
        int lineCount = 0;

        ifstream file(filename);
        string header;
        getline(file, header); // basligi atla

        vector<double> x, y;
        while (readCSVLine(file, x, y, info.inputSize, info.outputSize))
        {
            double err = cc.train("parity", x, y);
            epochError += err;
            lineCount++;
        }

        epochError /= lineCount;
        lastEpochError = epochError;
        cc.models["parity"].errors.push_back(epochError);

        if (debug)
            log_saver(to_string(epochError));

        progress_bar(epoch, epochNum);

        if (epochError < targetError)
        {
            log_saver("Stopped due targetError...");
            break;
        }

        if ((epoch + 1) % 150 == 0)
        {
            int monitor = cc.models["parity"].monitorNetwork();
            if (monitor == 2)
            {
                log_saver("Stopped due closing to zero slope...");
                break;
            }
        }
    }

    // Bitis zamani
    auto endTime = chrono::high_resolution_clock::now();
    auto durationMs = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    double durationSec = durationMs / 1000.0;

    cout << endl;
    cout << "==== Egitim Tamamlandi ====" << endl;

    if (durationSec >= 1.0)
        cout << "Gecen sure: " << durationSec << " saniye" << endl;
    else
        cout << "Gecen sure: " << durationMs << " milisaniye" << endl;

    cout << "Son epoch hatasi: " << lastEpochError << endl
         << endl;

    cout << "==== Test Basliyor ====" << endl;
    string testFile = "parity_problem.csv"; // test icin ayri CSV
    testModelFromCSV(cc, "parity", testFile);


    system("python3 Helpers/network_changes_logs.py");

    cout << "\n==== Kullanicidan Girdi Al ====" << endl;
    cout << "Virgulle ayrilmis input degerlerini gir (or: 0.5,1,0,1): ";

    string inputLine;
    getline(cin, inputLine); // once cin.get() kaldirman lazim, onun yerine burasi calisacak

    stringstream ss(inputLine);
    string token;
    vector<double> userInput;

    while (getline(ss, token, ','))
    {
        try
        {
            userInput.push_back(stod(token));
        }
        catch (...)
        {
            cerr << "Gecersiz sayi girdin: " << token << endl;
        }
    }

    if (!userInput.empty())
    {
        auto result = cc.forward("parity", userInput);

        cout << "Model ciktisi: ";
        for (double v : result)
            cout << v << " ";
        cout << endl;
    }
    else
    {
        cout << "Hic gecerli input girmedin!" << endl;
    }
    cin.get();
    return 0;
}*/

int main()
{
    srand(time(nullptr));
    CorticalColumn cc;

    // Baslangic modelini olustur (4 giris, 2 cikis)
    vector<int> layers = {4, 99, 2};
    cc.addModel("parity", layers, "sigmoid");

    // Etkilesimli mod baslat
    interactiveTraining(cc, "parity", "data.csv", layers[0], layers.back(), 0.01);

    return 0;
}
