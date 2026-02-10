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
#include <memory>
#include <sstream>
#include <algorithm>
#include <set>

#ifdef __EMSCRIPTEN__
#include <cstdio>
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...) cout
#endif

#include "ByteBPE/ByteBPETokenizer.h"

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

/* =========================
   TOKENIZER
========================= */
enum class TokenizerMode
{
    WORD,    // mevcut (default)
    SUBWORD, // char n-gram
    BPE
};

struct SetupConfig
{
    string modelName;
    string modelFile;
    string csvFile;
    double targetError;
    int maxEpoch;
    vector<int> layers;
    bool csvAvailable;
    TokenizerMode mode;
};

enum class RunMode
{
    CLI,
    SERVICE // web / api / arka plan
};

bool debugLog = debug;
string bpe_model_path = "LLM/Embeddings/bpe_tokenizer.json"; // Düzelteceğin zaman Network yapısının içinde de bu sabit var onu da değiştir

TokenizerMode mode = TokenizerMode::BPE;

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

        double learningRate = 0.01;
        int modelChangedSelf = 0;

        TokenizerMode mode;
        unique_ptr<ByteBPETokenizer> bpe_tokenizer;

        unordered_map<string, vector<float>> wordEmbeddings;
        unordered_map<string, vector<float>> commandEmbeddings;

        // Default constructor
        Network() = default;

        // Move constructor
        Network(Network &&other) noexcept = default;

        // Move assignment operator
        Network &operator=(Network &&other) noexcept = default;

        // Delete copy operations
        Network(const Network &) = delete;
        Network &operator=(const Network &) = delete;

        // ===============================
        // Aktivasyon Fonksiyonları
        // ===============================
        double activate(double x)
        {
            if (activationType == "tanh")
                return tanh(x);

            if (activationType == "relu")
                return x > 0.0 ? x : 0.0;

            if (activationType == "leaky_relu")
                return x > 0.0 ? x : 0.01 * x;

            if (activationType == "elu")
                return x >= 0.0 ? x : 0.01 * (exp(x) - 1.0);

            if (activationType == "softplus")
                return log(1.0 + exp(x));

            if (activationType == "linear")
                return x;

            // default: sigmoid
            return 1.0 / (1.0 + exp(-x));
        }

        double activateDerivative(double x)
        {
            if (activationType == "tanh")
            {
                double t = tanh(x);
                return 1.0 - t * t;
            }

            if (activationType == "relu")
                return x > 0.0 ? 1.0 : 0.0;

            if (activationType == "leaky_relu")
                return x > 0.0 ? 1.0 : 0.01;

            if (activationType == "elu")
                return x >= 0.0 ? 1.0 : 0.01 * exp(x);

            if (activationType == "softplus")
                return 1.0 / (1.0 + exp(-x)); // sigmoid(x)

            if (activationType == "linear")
                return 1.0;

            // default: sigmoid
            double s = 1.0 / (1.0 + exp(-x));
            return s * (1.0 - s);
        }

        // Network struct içine eklenecek fonksiyonlar:

        // Belirli bir pozisyona hidden layer ekle
        bool addLayerAt(size_t position, size_t neuronCount = 0)
        {
            // position: 1 = ilk hidden layer, 2 = ikinci hidden layer, vs.
            // 0 veya weights.size() üzerinde olamaz (input/output katmanları korunur)

            if (position == 0 || position > weights.size())
            {
                if (debug)
                    cout << "ERROR: Cannot add layer at position " << position
                         << " (input/output layers are protected)" << endl;
                return false;
            }

            // Önceki ve sonraki katman boyutlarını al
            size_t prevLayerSize = (position == 1) ? weights[0][0].size() : weights[position - 2].size();
            size_t nextLayerSize = (position == weights.size()) ? weights.back().size() : weights[position - 1].size();

            // Eğer nöron sayısı belirtilmemişse, ortalamasını al
            if (neuronCount == 0)
            {
                neuronCount = (prevLayerSize + nextLayerSize) / 2;
                if (neuronCount < 2)
                    neuronCount = 2;
            }

            if (debug)
            {
                cout << "Adding layer at position " << position
                     << " with " << neuronCount << " neurons" << endl;
                cout << "Previous layer size: " << prevLayerSize
                     << ", Next layer size: " << nextLayerSize << endl;
            }

            // Yeni katman için ağırlıklar ve bias'ları oluştur
            vector<vector<double>> newWeights;
            vector<double> newBiases;

            // Yeni katmanın her nöronu için önceki katmandan gelen bağlantılar
            for (size_t n = 0; n < neuronCount; n++)
            {
                vector<double> neuronWeights(prevLayerSize);
                for (size_t w = 0; w < prevLayerSize; w++)
                    neuronWeights[w] = ((rand() % 100) / 100.0 - 0.5);

                newWeights.push_back(neuronWeights);
                newBiases.push_back(((rand() % 100) / 100.0 - 0.5));
            }

            // Yeni katmanı ekle
            weights.insert(weights.begin() + position - 1, newWeights);
            biases.insert(biases.begin() + position - 1, newBiases);

            // Sonraki katmanın ağırlıklarını güncelle (yeni katmandan gelen bağlantılar)
            if (position <= weights.size() - 1)
            {
                size_t nextLayerIndex = position; // weights array'inde sonraki katman
                for (size_t n = 0; n < weights[nextLayerIndex].size(); n++)
                {
                    weights[nextLayerIndex][n].clear();
                    for (size_t w = 0; w < neuronCount; w++)
                        weights[nextLayerIndex][n].push_back(((rand() % 100) / 100.0 - 0.5));
                }
            }

            // Cache temizle
            layerInputs.clear();
            layerOutputs.clear();

            if (debug)
                cout << "Layer successfully added at position " << position << endl;

            log_saver("Layer added at position " + to_string(position) +
                      " with " + to_string(neuronCount) + " neurons");

            return true;
        }

        // En son hidden layer'a yeni katman ekle
        bool addLayer(size_t neuronCount = 0)
        {
            // Çıkış katmanından önce ekle
            return addLayerAt(weights.size(), neuronCount);
        }

        // Belirli pozisyondaki hidden layer'ı sil
        bool removeLayerAt(size_t position)
        {
            // position: 1 = ilk hidden layer, 2 = ikinci hidden layer, vs.
            // En az 1 hidden layer kalmalı

            if (position == 0 || position > weights.size() - 1)
            {
                if (debug)
                    cout << "ERROR: Invalid layer position " << position << endl;
                return false;
            }

            if (weights.size() <= 2)
            {
                if (debug)
                    cout << "ERROR: Cannot remove layer - minimum 1 hidden layer required" << endl;
                return false;
            }

            if (debug)
            {
                cout << "Removing layer at position " << position << endl;
                cout << "Layer size: " << weights[position - 1].size() << " neurons" << endl;
            }

            // Katmanı sil
            weights.erase(weights.begin() + position - 1);
            biases.erase(biases.begin() + position - 1);

            // Sonraki katmanın ağırlıklarını güncelle
            if (position < weights.size() + 1) // Silinen katmandan sonra bir katman varsa
            {
                size_t prevLayerSize = (position == 1) ? weights[0][0].size() : weights[position - 2].size();
                size_t nextLayerIndex = position - 1;

                // Sonraki katmanın her nöronunun ağırlıklarını yeniden boyutlandır
                for (size_t n = 0; n < weights[nextLayerIndex].size(); n++)
                {
                    weights[nextLayerIndex][n].clear();
                    for (size_t w = 0; w < prevLayerSize; w++)
                        weights[nextLayerIndex][n].push_back(((rand() % 100) / 100.0 - 0.5));
                }
            }

            // Cache temizle
            layerInputs.clear();
            layerOutputs.clear();

            if (debug)
                cout << "Layer successfully removed from position " << position << endl;

            log_saver("Layer removed from position " + to_string(position));

            return true;
        }

        // En pasif (en az etkili) hidden layer'ı bul ve sil
        bool removeMostInactiveLayer()
        {
            if (weights.size() <= 2)
            {
                if (debug)
                    cout << "Cannot remove layer - minimum 1 hidden layer required" << endl;
                return false;
            }

            // Her hidden layer'ın toplam aktivitesini hesapla
            vector<pair<double, size_t>> layerActivities;

            for (size_t l = 1; l < weights.size(); l++) // Hidden layer'ları kontrol et
            {
                double totalActivity = 0.0;

                // Katmandaki tüm nöronların aktivitelerini topla
                for (size_t n = 0; n < weights[l - 1].size(); n++)
                {
                    totalActivity += calculateNeuronActivity(l, n);
                }

                // Ortalama aktivite
                double avgActivity = totalActivity / weights[l - 1].size();
                layerActivities.push_back({avgActivity, l});
            }

            // En düşük aktiviteye sahip katmanı bul
            auto minLayer = min_element(layerActivities.begin(), layerActivities.end());

            if (debug)
            {
                cout << "Most inactive layer: " << minLayer->second
                     << " (activity: " << minLayer->first << ")" << endl;
            }

            return removeLayerAt(minLayer->second);
        }

        // Katman sayısını otomatik optimize et
        bool optimizeLayerCount()
        {
            // Mevcut katman yapısını analiz et
            size_t inputSize = weights[0][0].size();
            size_t outputSize = weights.back().size();
            size_t currentHiddenCount = weights.size() - 1;

            // İdeal katman sayısını hesapla
            int idealHiddenCount = hiddenLayers_logImproved(inputSize, outputSize, "medium");

            if (debug)
            {
                cout << "Current hidden layers: " << currentHiddenCount << endl;
                cout << "Ideal hidden layers: " << idealHiddenCount << endl;
            }

            bool changed = false;

            // Fazla katman varsa sil
            while (currentHiddenCount > idealHiddenCount && currentHiddenCount > 1)
            {
                if (removeMostInactiveLayer())
                {
                    currentHiddenCount--;
                    changed = true;
                }
                else
                {
                    break;
                }
            }

            // Eksik katman varsa ekle
            while (currentHiddenCount < idealHiddenCount)
            {
                // Optimal pozisyonu bul (en küçük katmandan sonra)
                size_t optimalPos = findSmallestHiddenLayer() + 1;

                if (addLayerAt(optimalPos))
                {
                    currentHiddenCount++;
                    changed = true;
                }
                else
                {
                    break;
                }
            }

            return changed;
        }

        // Katman bilgilerini yazdır
        void printLayerInfo()
        {
            cout << "\n==== Layer Information ====" << endl;
            cout << "Total layers (including input/output): " << weights.size() + 1 << endl;
            cout << "Hidden layers: " << weights.size() - 1 << endl;
            cout << "\nLayer details:" << endl;

            // Input layer
            cout << "Layer 0 (Input): " << weights[0][0].size() << " neurons" << endl;

            // Hidden layers
            for (size_t l = 0; l < weights.size() - 1; l++)
            {
                double avgActivity = 0.0;
                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    avgActivity += calculateNeuronActivity(l + 1, n);
                }
                avgActivity /= weights[l].size();

                cout << "Layer " << (l + 1) << " (Hidden): " << weights[l].size()
                     << " neurons, avg activity: " << avgActivity << endl;
            }

            // Output layer
            cout << "Layer " << weights.size() << " (Output): " << weights.back().size() << " neurons" << endl;
            cout << "==========================\n"
                 << endl;
        }

        // YENI: Piramit yapiya gore fazla noronlari sil
        // DUZELTILMIS: Piramit yapiya gore fazla noronlari sil
        // DUZELTILMIS: Piramit yapiya gore fazla noronlari sil
        bool removeExcessNeuronsFromPyramid()
        {
            // weights.size() = katman sayisi - 1 (sadece baglantilar)
            // Ornek: 4,99,2 → weights.size() = 2 (hidden→output)

            if (weights.size() < 1)
            { // En az 1 hidden layer olmali
                cout << "Not enough layers for pyramid optimization" << endl;
                return false;
            }

            // Giris ve cikis boyutlarini al
            size_t inputSize = (weights.size() > 0) ? weights[0][0].size() : 0;
            size_t outputSize = (weights.size() > 0) ? weights.back().size() : 0;

            if (inputSize == 0 || outputSize == 0)
            {
                cout << "Invalid network structure" << endl;
                return false;
            }

            // Ideal piramit yapisini hesapla (SADECE hidden layer'lar icin)
            // weights.size() = hidden layer sayisi (4,99,2 → 1 hidden layer)
            vector<size_t> idealSizes = calculatePyramidStructure(inputSize, outputSize, weights.size() - 1);

            if (debug)
            {
                cout << "Input size: " << inputSize << ", Output size: " << outputSize << endl;
                cout << "Number of hidden layers: " << weights.size() - 1 << endl;
                cout << "Ideal pyramid sizes for hidden layers: ";
                for (size_t i = 0; i < idealSizes.size(); i++)
                {
                    cout << idealSizes[i];
                    if (i < idealSizes.size() - 1)
                        cout << ", ";
                }
                cout << endl;

                cout << "Current hidden layer sizes: ";
                for (size_t i = 0; i < weights.size() - 1; i++)
                {
                    cout << weights[i].size();
                    if (i < weights.size() - 1)
                        cout << ", ";
                }
                cout << endl;
            }
            cout << "Debug1 " << weights.size() << endl;
            // Tum hidden layer'lar icin fazla noronlari sil
            bool changed = false;
            for (size_t i = 1; i < weights.size(); i++)
            {
                cout << "Debug2 " << idealSizes.size() << endl;
                if (i > idealSizes.size())
                {
                    if (debug)
                        cout << "Skipping layer " << i << " - no ideal size defined " << idealSizes.size() << endl;
                    continue;
                }
                // cout << "Debug3 > " << weights[1][1].size() << " > " << idealSizes[0] << endl;

                // Eger mevcut boyut ideal boyuttan buyukse, fazlaligi sil
                cout << "Debug 2.1 " << weights[i - 1].size() << endl;
                cout << "Debug 2.2 " << idealSizes[i - 1] << endl;
                cout << "Debug 2.3 " << i << endl;
                if (weights[i - 1].size() > idealSizes[i - 1])
                {
                    cout << "Debug4" << endl;

                    int excess = weights[i - 1].size() - idealSizes[i - 1];
                    cout << "Debug4.1" << endl;

                    if (debug)
                    {
                        cout << "Layer " << i << " has " << excess << " excess neurons (ideal: "
                             << idealSizes[i - 1] << ", current: " << weights[i - 1].size() << ")" << endl;
                    }

                    // En pasif noronlari sil (maksimum 10 noron)
                    changed = removeMostInactiveNeurons(i, min(excess, 10));
                }
                else
                {
                    cout << "Debug4.2" << endl;
                }
                cout << "Debug5" << endl;
            }
            cout << "Debug6" << endl;
            return changed;
        }

        // YENI: Belirli bir katmandan en pasif noronlari sil
        bool removeMostInactiveNeurons(size_t layerIndex, int numToRemove)
        {
            bool changed = false;
            if (layerIndex >= weights.size() || numToRemove <= 0)
                return changed;

            // Noron aktivitelerini hesapla ve sirala
            vector<pair<double, size_t>> neuronActivities;
            for (size_t n = 0; n < weights[layerIndex].size(); n++)
            {
                double activity = calculateNeuronActivity(layerIndex, n);
                neuronActivities.push_back({activity, n});
            }

            // Aktiviteye gore sirala (en pasifler basta)
            sort(neuronActivities.begin(), neuronActivities.end());

            // En pasif noronlari sil (ters sirada sil ki indeksler kaymasin)
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
                    changed = removeNeuronAt(layerIndex, neuronIndex, false);
                }
            }
            return changed;
        }

        int monitorNetwork(double targetError = 0.2)
        {
            /*
            0 : continue
            1 : slope close to zero lr up
            2 : slope close to zero end train
            3 : neuron removed
            4 : neuron added
            5 : layers changed
            */

            double slope = computeErrorSlope(75);

            // Eger egim cok kucuk ama hata hala yuksek → kapasite yetmiyor olabilir
            double lastError = errors.empty() ? 9999 : errors.back();

            if (slope != 0.0 && (abs(slope) < 0.001))
            {
                if (lrChanged <= 3)
                {
                    learningRate += 0.3;

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
                    if (lastError > targetError * 2) // Hata hala yuksek → kapasiteyi arttir
                    {

                        // 1️⃣ Optimal katman sayısını hesapla (logImproved)
                        int newHiddenCount = hiddenLayers_logImproved(weights[0][0].size(), weights.back().size(), "low");

                        int currentHiddenCount = weights.size() - 1; // hidden layer sayısı
                        if (newHiddenCount > currentHiddenCount)
                        {
                            // 2️⃣ Eksik katmanları ekle
                            int layersToAdd = newHiddenCount - currentHiddenCount;
                            for (int i = 0; i < layersToAdd; i++)
                            {
                                // Yeni hidden layer oluştur
                                vector<vector<double>> newWeights;
                                vector<double> newBiases;

                                size_t prevLayerSize = (weights.empty()) ? 0 : weights.back().size();
                                size_t nextLayerSize = (weights.empty()) ? 0 : weights.back().size();

                                // Basit initialization: her neuron inputSize kadar rastgele ağırlık alır
                                int neuronsInNewLayer = (prevLayerSize + nextLayerSize) / 2;
                                if (neuronsInNewLayer < 2)
                                    neuronsInNewLayer = 2;

                                for (size_t n = 0; n < neuronsInNewLayer; n++)
                                {
                                    vector<double> neuronWeights(prevLayerSize);
                                    for (size_t w = 0; w < prevLayerSize; w++)
                                        neuronWeights[w] = ((rand() % 100) / 100.0 - 0.5);
                                    newWeights.push_back(neuronWeights);
                                    newBiases.push_back(((rand() % 100) / 100.0 - 0.5));
                                }

                                // Yeni layer'i ağırlık ve bias listesine ekle
                                weights.insert(weights.end() - 1, newWeights); // çıkış layer önüne ekle
                                biases.insert(biases.end() - 1, newBiases);
                            }

                            if (debug)
                                cout << "[DEBUG] Added " << layersToAdd << " hidden layer(s) dynamically" << endl;

                            return 4; // katman eklendi
                        }

                        // SADECE gercekten pasif noronlari kaldir (rastgele katmanlardan DEGIL)
                        if (fixLayerImbalance())
                        {
                            return 4;
                        }

                        if (removeExcessNeuronsFromPyramid()) //|| removeOnlyTrulyInactiveNeurons())
                        {
                            return 4;
                        }
                        if (optimizeLayerCount())
                        {
                            return 5;
                        }

                        //// Piramit yapisini koruyarak en uygun katmana noron ekle
                        // size_t targetLayer = findOptimalLayerForAddition();
                        // log_saver("Optimize edilebilecek katman :" + targetLayer);
                        // if (addNeuronToLayer(targetLayer))
                        //{
                        //     cout << "Neuron added" << endl;
                        //     return 4;
                        // }

                        return 2;
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
                learningRate -= 0.02;
                if (debug)
                    cout << endl
                         << "Slope is positive errors increasing lr reduced" << endl;
                log_saver("Slope is positive errors increasing lr reduced new:" + std::to_string(learningRate));
                lrChanged += 1;
            }

            return 0;
        }

        // 2. Logaritmik yaklasim
        int hiddenLayers_logImproved(int inputSize, int outputSize, const string &level = "medium")
        {
            double ratio = (double)inputSize / (double)outputSize;
            double complexity = log2(ratio + 1) + log2(inputSize + outputSize);

            // Temel katman sayısı
            int layers = (int)round(complexity / 2.0);

            // Seviye ayarı
            double multiplier = 1.0;
            if (level == "low")
                multiplier = 0.6;
            else if (level == "medium")
                multiplier = 1.0;
            else if (level == "high")
                multiplier = 1.5;

            layers = (int)round(layers * multiplier);

            // 1–12 arası sınırla
            if (layers < 1)
                layers = 1;
            if (layers > 13)
                layers = 12;

            return layers;
        }

        // YENI: Piramit yapisi icin optimal katmani bul
        // DUZELTILMIS: Piramit yapisi icin optimal katmani bul (giris/cikis katmanlarina dokunmaz)
        size_t findOptimalLayerForAddition()
        {
            // Sadece hidden layer'lari dusun (giris ve cikis katmanlarini atla)
            if (weights.size() <= 2)
            {
                // Sadece 1 hidden layer varsa onu dondur
                return 1; // Hidden layer index 1'de (0: giris, 1: hidden, 2: cikis)
            }

            // Giris ve cikis boyutlarini al
            size_t inputSize = weights[0][0].size();
            size_t outputSize = weights.back().size();

            // Ideal piramit yapisini hesapla (sadece hidden layer'lar icin)
            vector<size_t> idealSizes = calculatePyramidStructure(inputSize, outputSize, weights.size() - 1);

            // Mevcut boyutlarla ideal boyutlari karsilastir, en cok eksigi olan HIDDEN katmani bul
            size_t optimalLayer = 1; // Varsayilan olarak ilk hidden layer
            double maxDeficit = 0.0;

            // Sadece hidden layer'lari kontrol et (index 1'den weights.size()-2'ye kadar)
            for (size_t i = 1; i < weights.size() - 1; i++)
            {
                // idealSizes indeksini ayarla (hidden layer'lar 0'dan baslar)
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

            // Eger hic eksigi olan hidden layer yoksa, en kucuk hidden layer'i bul
            if (maxDeficit <= 0)
            {
                return findSmallestHiddenLayer();
            }

            return optimalLayer;
        }

        // YENI: En kucuk HIDDEN katmani bul (giris/cikis katmanlarini atlar)
        size_t findSmallestHiddenLayer()
        {
            if (weights.size() <= 2)
            {
                return 1; // Varsayilan hidden layer index
            }

            size_t smallestLayer = 1;
            size_t minSize = weights[1].size();

            // Sadece hidden layer'lari kontrol et (index 1'den weights.size()-2'ye kadar)
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

        // DUZELTILMIS: Piramit yapisini hesapla (sadece hidden layer'lar icin)
        vector<size_t> calculatePyramidStructure(size_t inputSize, size_t outputSize, size_t numHiddenLayers)
        {
            vector<size_t> pyramidSizes;

            if (numHiddenLayers == 0)
            {
                cout << "Debug8" << endl;
                return pyramidSizes;
            }
            if (numHiddenLayers == 1)
            {
                // 4,3,2 → hidden: 3
                size_t middle = max((inputSize + outputSize) / 2, (size_t)2); // En az 2 noron
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
                    pyramidSizes.push_back(max(static_cast<size_t>(size), (size_t)2)); // En az 2 noron
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
        // YENI: Sadece gercekten pasif noronlari kaldir
        bool removeOnlyTrulyInactiveNeurons(double threshold = 0.3)
        {
            // Tum katmanlarda dolas ve gercekten pasif olan noronlari kaldir
            for (size_t layerIndex = 1; layerIndex < weights.size(); layerIndex++)
            {
                // Ters sirayla gitmek daha guvenli
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
                        return removeNeuronAt(layerIndex, n, false);

                        printModelASCII("", true);
                    }
                }
            }
            return false;
        }

        // YENI: Noron aktivitesini hesapla (daha karmasik metrik)
        // Noron aktivite hesaplamasini iyilestir
        double calculateNeuronActivity(size_t layerIndex, size_t neuronIndex)
        {
            if (layerIndex >= weights.size() || neuronIndex >= weights[layerIndex].size())
            {
                return 1.0;
            }

            // 1. Agirlik varyasyonunu hesapla (standart sapma benzeri)
            double sum = 0.0;
            double sumSq = 0.0;
            for (double w : weights[layerIndex][neuronIndex])
            {
                sum += w;
                sumSq += w * w;
            }
            double mean = sum / weights[layerIndex][neuronIndex].size();
            double variance = (sumSq / weights[layerIndex][neuronIndex].size()) - (mean * mean);

            // 2. Bias'in etkisi
            double biasEffect = fabs(biases[layerIndex][neuronIndex]);

            // 3. Aktivasyon = varyasyon * bias (noronun "ilgincligi")
            return sqrt(fabs(variance)) * (1.0 + biasEffect);
        }

        bool fixLayerImbalance()
        {
            if (!checkLayerImbalance())
            {
                if (debug)
                    cout << "[DEBUG] fixLayerImbalance: No imbalance detected -> returning false" << endl;
                return false;
            }

            vector<size_t> idealSizes = calculatePyramidStructure(
                weights[0][0].size(),
                weights.back().size(),
                weights.size() - 1); // sadece hidden layer'lar

            if (debug)
            {
                cout << "[DEBUG] fixLayerImbalance: Input=" << weights[0][0].size()
                     << " Output=" << weights.back().size()
                     << " HiddenCount=" << (weights.size() - 1) << endl;

                cout << "[DEBUG] idealSizes = [ ";
                for (auto v : idealSizes)
                    cout << v << " ";
                cout << "]" << endl;
            }

            bool changed = false;

            // Hidden katmanları düzelt
            for (size_t i = 1; i <= weights.size() - 1; i++)
            {
                size_t idealIndex = i - 1;
                if (idealIndex >= idealSizes.size())
                {
                    if (debug)
                        cout << "[DEBUG] BREAK: idealIndex=" << idealIndex
                             << " >= idealSizes.size()=" << idealSizes.size() << endl;
                    break;
                }

                int deficit = static_cast<int>(idealSizes[idealIndex]) -
                              static_cast<int>(weights[i - 1].size());

                if (debug)
                {
                    cout << "[DEBUG] Layer " << i
                         << " -> ideal=" << idealSizes[idealIndex]
                         << " actual=" << weights[i].size()
                         << " deficit=" << deficit << endl;
                }

                if (deficit > 0)
                {
                    int toAdd = min(deficit, 3);
                    if (debug)
                        cout << "[DEBUG] Adding " << toAdd
                             << " neuron(s) to hidden layer " << i << endl;

                    for (int j = 0; j < toAdd; j++)
                    {
                        bool result = addNeuronToLayer(i);
                        changed = changed || result;

                        if (debug)
                        {
                            cout << "   [DEBUG] addNeuronToLayer("
                                 << i << ") -> " << (result ? "success" : "failed")
                                 << " | newSize=" << weights[i].size() << endl;
                        }
                    }
                }
                else if (deficit < 0)
                {
                    if (debug)
                        cout << "[DEBUG] Layer " << i
                             << " has surplus neurons (" << -deficit << ")" << endl;
                    // Burada removeNeuronFromLayer eklenebilir
                }
            }

            if (debug)
                cout << "[DEBUG] fixLayerImbalance finished -> changed=" << (changed ? "true" : "false") << endl;

            return changed;
        }

        bool checkLayerImbalance(double threshold = 1.5)
        {
            if (weights.size() < 2)
            {
                log_saver("Something went wrong: weights.size() < 2");
                return false;
            }

            // Input size, output size ve hidden count → ideal yapıyı hesapla
            vector<size_t> idealSizes = calculatePyramidStructure(
                weights[0][0].size(),
                weights.back().size(),
                weights.size() - 1);

            if (debug)
            {
                cout << "[DEBUG] Input size = " << weights[0][0].size()
                     << " Output size = " << weights.back().size()
                     << " Hidden count = " << (weights.size() - 1) << endl;

                cout << "[DEBUG] idealSizes = [ ";
                for (auto v : idealSizes)
                    cout << v << " ";
                cout << "] (len=" << idealSizes.size() << ")" << endl;
            }

            for (size_t i = 1; i <= weights.size() - 1; i++) // sadece hidden
            {
                size_t idealIndex = i - 1; // hidden index → idealSizes index
                cout << "nice  i:" << i << endl;
                if (idealIndex >= idealSizes.size())
                {
                    if (debug)
                        cout << "[DEBUG] BREAK: idealIndex=" << idealIndex
                             << " >= idealSizes.size()=" << idealSizes.size() << endl;
                    break;
                }

                size_t actualSize = weights[i - 1].size();
                size_t idealSize = idealSizes[idealIndex];

                double ratio = static_cast<double>(idealSize) /
                               static_cast<double>(max(1, (int)actualSize));

                if (debug)
                {
                    cout << "[DEBUG] Layer " << i
                         << " -> ideal=" << idealSize
                         << " actual=" << actualSize
                         << " ratio=" << ratio
                         << " threshold=" << threshold << endl;
                }

                if (ratio >= threshold || ratio < 1.0 / threshold)
                {
                    if (debug)
                        cout << "[DEBUG] ❌ Imbalance detected at hidden layer " << i << endl;
                    return true;
                }
            }

            if (debug)
                cout << "[DEBUG] ✅ All hidden layers within balance" << endl;

            return false;
        }

        bool addNeuronToLayer(size_t layerIndex)
        {
            if (debug)
                cout << "Debug adding 0 layerIndex:" << layerIndex << endl;

            // layerIndex: 0 = ilk hidden katman, 1 = ikinci hidden katman, vs.
            // weights.size()-1 = çıkış katmanı indeksi

            if (layerIndex >= weights.size())
            {
                if (debug)
                    cout << "ERROR: Layer index " << layerIndex
                         << " corresponds to output layer or is invalid!" << endl;
                return false;
            }

            if (debug)
                cout << "Adding neuron to HIDDEN layer " << layerIndex << endl;

            // Yeni nöron ağırlıkları (önceki katmandan gelen bağlantılar)
            size_t inputSize = weights[layerIndex - 1][0].size();
            vector<double> newNeuronWeights(inputSize);
            for (size_t i = 0; i < inputSize; i++)
                newNeuronWeights[i] = ((rand() % 100) / 100.0 - 0.5);

            // Hidden katmana nöron ekle
            weights[layerIndex - 1].push_back(newNeuronWeights);
            biases[layerIndex - 1].push_back(((rand() % 100) / 100.0 - 0.5));

            // Sonraki katmana (hidden→hidden veya hidden→output) yeni bağlantılar ekle
            for (size_t n = 0; n < weights[layerIndex].size(); n++)
            {
                weights[layerIndex][n].push_back(((rand() % 100) / 100.0 - 0.5));
            }

            // Cache temizle
            layerInputs.clear();
            layerOutputs.clear();

            cout << "Neuron added to hidden layer " << layerIndex << endl;
            log_saver("Neuron added to hidden layer :" + to_string(layerIndex));
            return true;
        }

        // -----------------------------
        // Islevsiz Noron Kontrol ve Kaldir
        // -----------------------------

        void printModelASCII(const string &name = "", bool onlyModel = false)
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
            if (onlyModel == false)
            {

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
            }

            cout << "\n";

            // Katman basina noron sayilarini yazdir
            cout << "Layer neuron counts: ";
            cout << "[" << weights[0][0].size() << "] "; // Input layer

            for (size_t l = 0; l < weights.size(); l++)
            {
                cout << "[" << weights[l].size() << "] ";
            }
            cout << "\n";

            cout << "=========================\n";
        }

        // -----------------------------
        // Index ile Noron Sil
        // -----------------------------
        bool removeNeuronAt(size_t layerIndex, size_t neuronIndex, bool preserveWeights = false)
        {
            bool changed = false;
            if (layerIndex == 0)
            {
                cout << "Cant delete input neuron At" << endl;
                return changed;
            }
            if (layerIndex >= weights.size() + 1) // cikis katmani ve gecersiz index
            {
                cout << "Cant delete output neuron At" << endl;
                return changed;
            }

            size_t weightLayer = layerIndex - 1; // weights giris-cikis kaymasi
            if (neuronIndex >= weights[weightLayer].size())
            {
                log_saver("Something is not right again" + to_string(neuronIndex) + " >= " + to_string(weights[weightLayer].size()));
                return changed;
            }

            // Gizli katmandan noronu sil
            weights[weightLayer].erase(weights[weightLayer].begin() + neuronIndex);
            biases[weightLayer].erase(biases[weightLayer].begin() + neuronIndex);
            // Sonraki layer'in agirliklarini guncelle
            if (!preserveWeights && layerIndex < weights.size())
            {
                for (size_t n = 0; n < weights[layerIndex].size(); n++)
                {
                    if (weights[layerIndex][n].size() > neuronIndex)
                        weights[layerIndex][n].erase(weights[layerIndex][n].begin() + neuronIndex);
                }
            }
            changed = true;

            // Cache temizle
            layerInputs.clear();
            layerOutputs.clear();

            log_saver("Specified Neuron removed from layer " + to_string(layerIndex) + ", index " + to_string(neuronIndex));
            return changed;
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

        double train(const vector<double> &inputs, const vector<double> &targets, double targetError)
        {
            // -----------------------------
            // Model Kontrolu (Egitime baslamadan once)
            // -----------------------------

            if (weights.empty() || biases.empty())
            {
                if (debug)
                    cout << "[ERROR] Hata: Model bos. weights veya biases yok!" << endl;
                return -1;
            }

            if (weights.size() != biases.size())
            {
                if (debug)
                    cout << "[ERROR] Hata: weights ve biases katman sayilari eslesmiyor!" << endl;
                return -1;
            }

            if (inputs.empty())
            {
                if (debug)
                    cout << "[ERROR] Hata: Girdi vektoru bos!" << endl;
                return -1;
            }

            if (targets.empty())
            {
                if (debug)
                    cout << "[ERROR] Hata: Hedef vektoru bos!" << endl;
                return -1;
            }
            vector<double> output = forward(inputs); // <- layerOutputs burada dolacak

            if (layerOutputs.size() != weights.size())
            {
                if (debug)
                {
                    cout << "[ERROR] Hata: layerOutputs boyutu weights ile eşleşmiyor!" << endl;
                    cout << "  layerOutputs.size() = " << layerOutputs.size() << endl;
                    cout << "  weights.size()      = " << weights.size() << endl;

                    for (size_t i = 0; i < layerOutputs.size(); i++)
                    {
                        cout << "  layerOutputs[" << i << "].size() = " << layerOutputs[i].size() << endl;
                    }
                    for (size_t i = 0; i < weights.size(); i++)
                    {
                        cout << "  weights[" << i << "].size() = " << weights[i].size() << endl;
                    }
                }
                return -1;
            }

            if (layerInputs.size() != weights.size())
            {
                if (debug)
                    cout << "[ERROR] Hata: layerInputs boyutu weights ile eslesmiyor!" << endl;
                return -1;
            }

            for (size_t l = 0; l < weights.size(); l++)
            {
                if (weights[l].size() != biases[l].size())
                {
                    if (debug)
                        cout << "[ERROR] Hata: Layer " << l << " weights ve biases boyutlari eslesmiyor!" << endl;
                    return -1;
                }

                for (size_t n = 0; n < weights[l].size(); n++)
                {
                    size_t expectedInputSize = (l == 0) ? inputs.size() : weights[l - 1].size();
                    if (weights[l][n].size() != expectedInputSize)
                    {
                        if (debug)
                        {
                            cout << "[ERROR] Hata: Layer " << l << ", Neuron " << n
                                 << " agirlik boyutu beklenen (" << expectedInputSize
                                 << ") ile eslesmiyor! (" << weights[l][n].size() << ")" << endl;
                        }
                        return -1;
                    }
                }
            }

            if (targets.size() != weights.back().size())
            {
                if (debug)
                    cout << "[ERROR] Hata: Hedef boyutu output katmani boyutu ile eslesmiyor! "
                         << targets.size() << " vs " << weights.back().size() << endl;
                return -1;
            }

            // -----------------------------
            // Backpropagation
            // -----------------------------

            vector<vector<double>> deltas(weights.size());

            // Output layer delta
            deltas.back().resize(output.size());
            double errorSum = 0.0;
            for (size_t i = 0; i < output.size(); i++)
            {
                double error = output[i] - targets[i];
                deltas.back()[i] = error * activateDerivative(layerInputs.back()[i]);
                errorSum += error * error;
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

            double error = errorSum / output.size();

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

        bool saveToFile(const string &filename, const string &modelKey = "")
        {
            ofstream file(filename, ios::binary);
            if (!file.is_open())
            {
                if (debug)
                    cout << "[ERROR] Dosya acilamadi: " << filename << endl;
                return false;
            }

            // 0. Model key'ini kaydet
            uint64_t keyLen = static_cast<uint64_t>(modelKey.length());
            file.write(reinterpret_cast<const char *>(&keyLen), sizeof(keyLen));
            if (keyLen > 0)
                file.write(modelKey.c_str(), static_cast<std::streamsize>(keyLen));

            // 1. Aktivasyon tipini kaydet
            uint64_t typeLen = static_cast<uint64_t>(activationType.length());
            file.write(reinterpret_cast<const char *>(&typeLen), sizeof(typeLen));
            if (typeLen > 0)
                file.write(activationType.c_str(), static_cast<std::streamsize>(typeLen));

            // 1.5 TokenizerMode'u kaydet
            int modeInt = static_cast<int>(mode);
            file.write(reinterpret_cast<const char *>(&modeInt), sizeof(modeInt));

            // 2. Learning rate ve diğer parametreleri kaydet
            file.write(reinterpret_cast<const char *>(&learningRate), sizeof(learningRate));
            file.write(reinterpret_cast<const char *>(&lrChanged), sizeof(lrChanged));
            file.write(reinterpret_cast<const char *>(&modelChangedSelf), sizeof(modelChangedSelf));

            // 3. Errors vector'ünü kaydet
            uint64_t errSize = static_cast<uint64_t>(errors.size());
            file.write(reinterpret_cast<const char *>(&errSize), sizeof(errSize));
            if (errSize > 0)
                file.write(reinterpret_cast<const char *>(errors.data()), static_cast<std::streamsize>(errSize * sizeof(double)));

            // 4. Weights yapısını kaydet
            uint64_t numLayers = static_cast<uint64_t>(weights.size());
            file.write(reinterpret_cast<const char *>(&numLayers), sizeof(numLayers));

            for (const auto &layer : weights)
            {
                uint64_t numNeurons = static_cast<uint64_t>(layer.size());
                file.write(reinterpret_cast<const char *>(&numNeurons), sizeof(numNeurons));

                for (const auto &neuron : layer)
                {
                    uint64_t numWeights = static_cast<uint64_t>(neuron.size());
                    file.write(reinterpret_cast<const char *>(&numWeights), sizeof(numWeights));
                    if (numWeights > 0)
                        file.write(reinterpret_cast<const char *>(neuron.data()), static_cast<std::streamsize>(numWeights * sizeof(double)));
                }
            }

            // 5. Biases yapısını kaydet
            uint64_t numBiasLayers = static_cast<uint64_t>(biases.size());
            file.write(reinterpret_cast<const char *>(&numBiasLayers), sizeof(numBiasLayers));

            for (const auto &layer : biases)
            {
                uint64_t numBiases = static_cast<uint64_t>(layer.size());
                file.write(reinterpret_cast<const char *>(&numBiases), sizeof(numBiases));
                if (numBiases > 0)
                    file.write(reinterpret_cast<const char *>(layer.data()), static_cast<std::streamsize>(numBiases * sizeof(double)));
            }

            // 🆕 6. WORD EMBEDDINGS KAYDET
            uint64_t wordEmbSize = static_cast<uint64_t>(wordEmbeddings.size());
            file.write(reinterpret_cast<const char *>(&wordEmbSize), sizeof(wordEmbSize));

            for (const auto &[word, vec] : wordEmbeddings)
            {
                // Kelime uzunluğu ve kelimeyi kaydet
                uint64_t wordLen = static_cast<uint64_t>(word.length());
                file.write(reinterpret_cast<const char *>(&wordLen), sizeof(wordLen));
                if (wordLen > 0)
                    file.write(word.c_str(), static_cast<std::streamsize>(wordLen));

                // Vector'ü kaydet
                uint64_t vecSize = static_cast<uint64_t>(vec.size());
                file.write(reinterpret_cast<const char *>(&vecSize), sizeof(vecSize));
                if (vecSize > 0)
                    file.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(vecSize * sizeof(float)));
            }

            // 🆕 7. COMMAND EMBEDDINGS KAYDET
            uint64_t cmdEmbSize = static_cast<uint64_t>(commandEmbeddings.size());
            file.write(reinterpret_cast<const char *>(&cmdEmbSize), sizeof(cmdEmbSize));

            for (const auto &[cmd, vec] : commandEmbeddings)
            {
                uint64_t cmdLen = static_cast<uint64_t>(cmd.length());
                file.write(reinterpret_cast<const char *>(&cmdLen), sizeof(cmdLen));
                if (cmdLen > 0)
                    file.write(cmd.c_str(), static_cast<std::streamsize>(cmdLen));

                uint64_t vecSize = static_cast<uint64_t>(vec.size());
                file.write(reinterpret_cast<const char *>(&vecSize), sizeof(vecSize));
                if (vecSize > 0)
                    file.write(reinterpret_cast<const char *>(vec.data()), static_cast<std::streamsize>(vecSize * sizeof(float)));
            }

            bool hasBPE = (bpe_tokenizer != nullptr);
            file.write(reinterpret_cast<const char *>(&hasBPE), sizeof(hasBPE));
            if (hasBPE)
            {
                // BPE model dosya yolunu kaydet
                uint64_t pathLen = static_cast<uint64_t>(bpe_model_path.length());
                file.write(reinterpret_cast<const char *>(&pathLen), sizeof(pathLen));
                if (pathLen > 0)
                    file.write(bpe_model_path.c_str(), static_cast<std::streamsize>(pathLen));
            }

            file.close();
            if (debug)
                cout << "[INFO] Model + embeddings basariyla kaydedildi: " << filename << endl;
            return true;
        }

        string loadedModelKey; // Yüklenen model key'ini sakla

        bool loadFromFile(const string &filename, const char *bpeFilePath = nullptr)
        {
            ifstream file(filename, ios::binary);
            if (!file.is_open())
            {
                if (debug)
                    cout << "[ERROR] Dosya acilamadi: " << filename << endl;
                return false;
            }

            // 0. Model key'ini oku
            uint64_t keyLen;
            if (!file.read(reinterpret_cast<char *>(&keyLen), sizeof(keyLen)))
            {
                cerr << "[ERROR] Failed to read model key length" << endl;
                return false;
            }
            if (keyLen > 0)
            {
                loadedModelKey.resize(static_cast<size_t>(keyLen));
                if (!file.read(&loadedModelKey[0], static_cast<std::streamsize>(keyLen)))
                {
                    cerr << "[ERROR] Failed to read model key" << endl;
                    return false;
                }
            }

            // 1. Aktivasyon tipini oku
            uint64_t typeLen;
            if (!file.read(reinterpret_cast<char *>(&typeLen), sizeof(typeLen)))
            {
                cerr << "[ERROR] Failed to read activation type length" << endl;
                return false;
            }
            if (typeLen > 0)
            {
                activationType.resize(static_cast<size_t>(typeLen));
                if (!file.read(&activationType[0], static_cast<std::streamsize>(typeLen)))
                {
                    cerr << "[ERROR] Failed to read activation type" << endl;
                    return false;
                }
            }

            // 1.5 TokenizerMode'u oku
            int modeInt;
            file.read(reinterpret_cast<char *>(&modeInt), sizeof(modeInt));
            mode = static_cast<TokenizerMode>(modeInt);

            // 2. Learning rate ve diğer parametreleri oku
            if (!file.read(reinterpret_cast<char *>(&learningRate), sizeof(learningRate)))
            {
                cerr << "[ERROR] Failed to read learningRate" << endl;
                return false;
            }
            if (!file.read(reinterpret_cast<char *>(&lrChanged), sizeof(lrChanged)))
            {
                cerr << "[ERROR] Failed to read lrChanged" << endl;
                return false;
            }
            if (!file.read(reinterpret_cast<char *>(&modelChangedSelf), sizeof(modelChangedSelf)))
            {
                cerr << "[ERROR] Failed to read modelChangedSelf" << endl;
                return false;
            }

            // 3. Errors vector'ünü oku
            uint64_t errSize;
            if (!file.read(reinterpret_cast<char *>(&errSize), sizeof(errSize)))
            {
                cerr << "[ERROR] Failed to read errors size" << endl;
                return false;
            }
            if (errSize > 0)
            {
                errors.resize(static_cast<size_t>(errSize));
                if (!file.read(reinterpret_cast<char *>(errors.data()), static_cast<std::streamsize>(errSize * sizeof(double))))
                {
                    cerr << "[ERROR] Failed to read errors data" << endl;
                    return false;
                }
            }

            // 4. Weights yapısını oku
            uint64_t numLayers;
            if (!file.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers)))
            {
                cerr << "[ERROR] Failed to read numLayers" << endl;
                return false;
            }
            if (numLayers > 10000)
            {
                cerr << "[ERROR] numLayers too large: " << numLayers << endl;
                return false;
            }
            weights.resize(static_cast<size_t>(numLayers));

            for (auto &layer : weights)
            {
                uint64_t numNeurons;
                if (!file.read(reinterpret_cast<char *>(&numNeurons), sizeof(numNeurons)))
                {
                    cerr << "[ERROR] Failed to read numNeurons" << endl;
                    return false;
                }

                if (numNeurons > 10000)
                { // güvenlik sınırı
                    cout << "[WARNING] Suspicious numNeurons: " << numNeurons << endl;
                }

                layer.resize(numNeurons);

                for (auto &neuron : layer)
                {
                    uint64_t numWeights;
                    if (!file.read(reinterpret_cast<char *>(&numWeights), sizeof(numWeights)))
                    {
                        cerr << "[ERROR] Failed to read numWeights" << endl;
                        return false;
                    }

                    if (numWeights > 10000)
                    { // güvenlik sınırı
                        cerr << "[ERROR] Suspicious numWeights: " << numWeights << endl;
                        return false;
                    }

                    neuron.resize(numWeights);

                    if (!file.read(reinterpret_cast<char *>(neuron.data()), numWeights * sizeof(double)))
                    {
                        cerr << "[ERROR] Failed to read neuron weights" << endl;
                        return false;
                    }
                }
            }

            // 5. Biases yapısını oku
            uint64_t numBiasLayers;
            if (!file.read(reinterpret_cast<char *>(&numBiasLayers), sizeof(numBiasLayers)))
            {
                cerr << "[ERROR] Failed to read numBiasLayers" << endl;
                return false;
            }
            if (numBiasLayers > 10000)
            {
                cerr << "[ERROR] numBiasLayers too large: " << numBiasLayers << endl;
                return false;
            }
            biases.resize(static_cast<size_t>(numBiasLayers));

            for (auto &layer : biases)
            {
                uint64_t numBiases;
                if (!file.read(reinterpret_cast<char *>(&numBiases), sizeof(numBiases)))
                {
                    cerr << "[ERROR] Failed to read numBiases" << endl;
                    return false;
                }
                if (numBiases > 1000000)
                {
                    cerr << "[ERROR] numBiases suspicious: " << numBiases << endl;
                    return false;
                }
                layer.resize(static_cast<size_t>(numBiases));
                if (numBiases > 0)
                {
                    if (!file.read(reinterpret_cast<char *>(layer.data()), static_cast<std::streamsize>(numBiases * sizeof(double))))
                    {
                        cerr << "[ERROR] Failed to read bias data" << endl;
                        return false;
                    }
                }
            }

            // 🆕 6. WORD EMBEDDINGS OKU
            uint64_t wordEmbSize;
            if (!file.read(reinterpret_cast<char *>(&wordEmbSize), sizeof(wordEmbSize)))
            {
                cerr << "[ERROR] Failed to read wordEmbSize" << endl;
                return false;
            }

            wordEmbeddings.clear();
            for (uint64_t i = 0; i < wordEmbSize; i++)
            {
                // Kelimeyi oku
                uint64_t wordLen;
                if (!file.read(reinterpret_cast<char *>(&wordLen), sizeof(wordLen)))
                {
                    cerr << "[ERROR] Failed to read wordLen" << endl;
                    return false;
                }
                if (wordLen > 10000)
                {
                    cerr << "[ERROR] wordLen suspicious: " << wordLen << endl;
                    return false;
                }
                string word(static_cast<size_t>(wordLen), ' ');
                if (wordLen > 0)
                {
                    if (!file.read(&word[0], static_cast<std::streamsize>(wordLen)))
                    {
                        cerr << "[ERROR] Failed to read word data" << endl;
                        return false;
                    }
                }

                // Vector'ü oku
                uint64_t vecSize;
                if (!file.read(reinterpret_cast<char *>(&vecSize), sizeof(vecSize)))
                {
                    cerr << "[ERROR] Failed to read vecSize" << endl;
                    return false;
                }
                if (vecSize > 1000000)
                {
                    cerr << "[ERROR] vecSize suspicious: " << vecSize << endl;
                    return false;
                }
                vector<float> vec(static_cast<size_t>(vecSize));
                if (vecSize > 0)
                {
                    if (!file.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(vecSize * sizeof(float))))
                    {
                        cerr << "[ERROR] Failed to read word vector data" << endl;
                        return false;
                    }
                }

                wordEmbeddings[word] = vec;
            }

            // 🆕 7. COMMAND EMBEDDINGS OKU
            uint64_t cmdEmbSize;
            if (!file.read(reinterpret_cast<char *>(&cmdEmbSize), sizeof(cmdEmbSize)))
            {
                cerr << "[ERROR] Failed to read cmdEmbSize" << endl;
                return false;
            }

            commandEmbeddings.clear();
            for (uint64_t i = 0; i < cmdEmbSize; i++)
            {
                uint64_t cmdLen;
                if (!file.read(reinterpret_cast<char *>(&cmdLen), sizeof(cmdLen)))
                {
                    cerr << "[ERROR] Failed to read cmdLen" << endl;
                    return false;
                }
                if (cmdLen > 10000)
                {
                    cerr << "[ERROR] cmdLen suspicious: " << cmdLen << endl;
                    return false;
                }
                string cmd(static_cast<size_t>(cmdLen), ' ');
                if (cmdLen > 0)
                {
                    if (!file.read(&cmd[0], static_cast<std::streamsize>(cmdLen)))
                    {
                        cerr << "[ERROR] Failed to read cmd data" << endl;
                        return false;
                    }
                }

                uint64_t vecSize;
                if (!file.read(reinterpret_cast<char *>(&vecSize), sizeof(vecSize)))
                {
                    cerr << "[ERROR] Failed to read cmd vecSize" << endl;
                    return false;
                }
                if (vecSize > 1000000)
                {
                    cerr << "[ERROR] cmd vecSize suspicious: " << vecSize << endl;
                    return false;
                }
                vector<float> vec(static_cast<size_t>(vecSize));
                if (vecSize > 0)
                {
                    if (!file.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(vecSize * sizeof(float))))
                    {
                        cerr << "[ERROR] Failed to read cmd vector data" << endl;
                        return false;
                    }
                }

                commandEmbeddings[cmd] = vec;
            }

            // Cache'leri temizle
            layerInputs.clear();
            layerOutputs.clear();

            // ✅ Fonksiyon parametresini öncelikli kullan, yoksa binary'den oku
            bool hasBPE = false;
            if (!file.read(reinterpret_cast<char *>(&hasBPE), sizeof(hasBPE)))
            {
                cerr << "[ERROR] Failed to read hasBPE flag" << endl;
                return false;
            }

            if (hasBPE)
            {
                string loadedBpePath;

                // 1️⃣ Önce parametreyi kontrol et
                if (bpeFilePath != nullptr && bpeFilePath[0] != '\0')
                {
                    // ✅ Parametre verilmişse onu kullan
                    loadedBpePath = bpeFilePath;
                    if (debugLog)
                        cout << "[loadFromFile] BPE path from parameter: " << loadedBpePath << endl;

                    // Binary'den path bilgisini oku ama atla (uyumluluk için)
                    uint64_t pathLen;
                    if (!file.read(reinterpret_cast<char *>(&pathLen), sizeof(pathLen)))
                    {
                        cerr << "[ERROR] Failed to read BPE path length" << endl;
                        return false;
                    }

                    if (pathLen > 0 && pathLen < 20000)
                    {
                        // Binary'deki path'i atla
                        file.seekg(pathLen, std::ios::cur);
                    }
                }
                else
                {
                    // ❌ Parametre yoksa binary'den oku (eski davranış)
                    uint64_t pathLen;
                    if (!file.read(reinterpret_cast<char *>(&pathLen), sizeof(pathLen)))
                    {
                        cerr << "[ERROR] Failed to read BPE path length" << endl;
                        return false;
                    }

                    if (pathLen > 20000)
                    {
                        cerr << "[ERROR] BPE path length suspicious: " << pathLen << endl;
                        return false;
                    }

                    loadedBpePath.resize(pathLen > 0 ? static_cast<size_t>(pathLen) : 0);
                    if (pathLen > 0)
                    {
                        if (!file.read(&loadedBpePath[0], static_cast<std::streamsize>(pathLen)))
                        {
                            cerr << "[ERROR] Failed to read BPE path" << endl;
                            return false;
                        }
                    }

                    if (debugLog)
                        cout << "[loadFromFile] BPE path from binary: " << loadedBpePath << endl;
                }

                // 2️⃣ BPE tokenizer'ı yükle (her iki durumda da)
                if (bpe_tokenizer != nullptr)
                {
                    bpe_tokenizer.reset();
                    bpe_tokenizer = nullptr;
                }

                bpe_tokenizer = std::make_unique<ByteBPETokenizer>();
                if (bpe_tokenizer != nullptr)
                {
                    bpe_tokenizer->debugLog = debugLog;
                    bool ok = bpe_tokenizer->load(loadedBpePath);
                    if (!ok)
                    {
                        cerr << "[ERROR] BPE tokenizer failed to load: " << loadedBpePath << endl;
                        bpe_tokenizer.reset();
                        bpe_tokenizer = nullptr;
                        hasBPE = false;
                    }
                    else
                    {
                        // Verify tokenizer actually populated vocab
                        try
                        {
                            auto &idmap = bpe_tokenizer->get_id_to_token();
                            if (idmap.empty())
                            {
                                cerr << "[ERROR] BPE tokenizer loaded but vocabulary is empty" << endl;
                                bpe_tokenizer.reset();
                                bpe_tokenizer = nullptr;
                                hasBPE = false;
                            }
                            else
                            {
                                cout << "[loadFromFile] [INFO] BPE tokenizer yuklendi: " << loadedBpePath
                                     << " (vocab size: " << idmap.size() << ")" << endl;
                            }
                        }
                        catch (...)
                        {
                            cerr << "[ERROR] Exception while validating bpe_tokenizer" << endl;
                            bpe_tokenizer.reset();
                            bpe_tokenizer = nullptr;
                            hasBPE = false;
                        }
                    }
                }
            }

            file.close();
            if (debug)
                cout << "[INFO] Model + embeddings basariyla yuklendi: " << filename
                     << " (Words: " << wordEmbeddings.size()
                     << ", Commands: " << commandEmbeddings.size() << ")" << endl;

            return true;
        }

        ~Network()
        {
            if (bpe_tokenizer != nullptr)
            {
                bpe_tokenizer.reset();
                bpe_tokenizer = nullptr;
            }
        }

    private:
        vector<vector<double>> layerInputs;
        vector<vector<double>> layerOutputs;
    };

    unordered_map<string, Network> models;
    string defaultNeuronActivationType = "tanh";
    double learningRate = 0.02;

    void addModel(const string &key, const vector<int> &topology, const string &activation = "tanh")
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
        models[key] = std::move(net);
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

    bool saveModel(const string &modelKey, const string &filename)
    {
        auto it = models.find(modelKey);
        if (it == models.end())
        {
            if (debug)
                cout << "[ERROR] Model bulunamadi: " << modelKey << endl;
            return false;
        }

        // Model key'ini binary dosyasına kaydet
        return it->second.saveToFile(filename, modelKey);
    }

    // Model yükleme fonksiyonu
    bool loadModel(const string &filename, const string &newModelKey = "", const string &bpeJsonPath = bpe_model_path)
    {
        Network net;
        if (debug)
        {
            cout << "[CorticalColumn] [loadModel] BPE json path:" << bpeJsonPath << "\n";
        }
        if (!net.loadFromFile(filename, bpeJsonPath.c_str()))
        {
            return false;
        }

        // Eğer key verilmemişse, dosyadan okunan key'i kullan
        string modelKey = newModelKey.empty() ? net.loadedModelKey : newModelKey;
        if (modelKey.empty())
        {
            modelKey = "loaded_model";
        }

        models[modelKey] = std::move(net);
        if (debug)
            cout << "[INFO] Model '" << modelKey << "' olarak yuklendi" << endl;
        return true;
    }
};

void progress_bar(int current, int total = 50, int bar_length = 40, double extra = 0)
{
    int percent = current * 100 / total;
    int filled = bar_length * percent / 100;

    cout << "\r["; // Satir basina don
    for (int i = 0; i < filled; i++)
        cout << "#";
    for (int i = filled; i < bar_length; i++)
        cout << "-";
    cout << "] " << percent << "% (" << current << "/" << total << ") " << extra;
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

    // Boş satırları atla
    if (line.empty())
        return readCSVLine(file, x, y, inputSize, outputSize);

    // Satırı temizle (başındaki/sonundaki boşluklar)
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);

    if (line.empty())
        return readCSVLine(file, x, y, inputSize, outputSize);

    stringstream ss(line);
    string cell;
    vector<double> row;

    while (getline(ss, cell, ','))
    {
        // Hücreyi temizle
        cell.erase(0, cell.find_first_not_of(" \t\r\n"));
        cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

        if (cell.empty())
        {
            if (debug)
                cerr << "[WARNING] Boş hücre bulundu, 0 olarak ekleniyor" << endl;
            row.push_back(0.0);
            continue;
        }

        try
        {
            row.push_back(stod(cell));
        }
        catch (const invalid_argument &e)
        {
            if (debug)
            {
                cerr << "[ERROR] Geçersiz sayı: '" << cell << "' (satır: " << line << ")" << endl;
                cerr << "         Hata: " << e.what() << endl;
            }
            return false;
        }
        catch (const out_of_range &e)
        {
            if (debug)
                cerr << "[ERROR] Sayı aralık dışı: '" << cell << "'" << endl;
            return false;
        }
    }

    if (row.size() != (size_t)(inputSize + outputSize))
    {
        if (debug)
        {
            cerr << "[WARNING] Satır boyutu uyumsuz. Beklenen: " << (inputSize + outputSize)
                 << ", Bulunan: " << row.size() << endl;
            cerr << "          Satır: " << line << endl;
        }
        return false;
    }

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

void ensureCSVHeader(
    const string &csvFile,
    int inputSize,
    int outputSize)
{
    // Beklenen header'ı oluştur (1'den başlar)
    string expectedHeader;

    for (int i = 1; i <= inputSize; i++)
    {
        expectedHeader += "input" + to_string(i) + ",";
    }

    for (int i = 1; i <= outputSize; i++)
    {
        expectedHeader += "target" + to_string(i);
        if (i != outputSize)
            expectedHeader += ",";
    }

    // Dosya var mı?
    ifstream in(csvFile);
    if (!in.good())
    {
        // Dosya yok → sadece header yaz
        ofstream out(csvFile);
        out << expectedHeader << "\n";
        return;
    }

    // İlk satırı oku
    string firstLine;
    getline(in, firstLine);
    in.close();

    // Header zaten doğruysa çık
    if (firstLine == expectedHeader)
        return;

    // Header yoksa → dosyayı yeniden yaz
    vector<string> lines;
    lines.reserve(1024); // küçük performans iyileştirmesi

    lines.push_back(expectedHeader);

    ifstream in2(csvFile);
    string line;
    while (getline(in2, line))
    {
        lines.push_back(line);
    }
    in2.close();

    ofstream out(csvFile, ios::trunc);
    for (const auto &l : lines)
        out << l << "\n";
}

void trainFromCSV(CorticalColumn &cc, const string &modelKey, const string &csvFile,
                  int inputSize, int outputSize, double targetError, bool isNewData = false, int maxEpoch = 1000)
{
    string message = isNewData ? "Yeni ornek eklendi, yeniden egitim basliyor..."
                               : "CSV'deki tum verilerle yeniden egitim basliyor...";
    cout << message << endl;

    // CSV dosyasını kontrol et
    ifstream testFile(csvFile);
    if (!testFile.is_open())
    {
        cerr << "[ERROR] CSV dosyasi bulunamadi: " << csvFile << endl;
        return;
    }
    testFile.close();

    // CSV içeriğini kontrol et
    ifstream checkFile(csvFile);
    string firstLine;
    getline(checkFile, firstLine); // Header

    if (!getline(checkFile, firstLine))
    {
        cerr << "[ERROR] CSV dosyasi bos (sadece header var)!" << endl;
        cerr << "[COZUM] Önce veri oluşturun:" << endl;
        cerr << "  1. python3 createEmbeddings.py" << endl;
        cerr << "  2. python3 commandCreator.py" << endl;
        checkFile.close();
        return;
    }
    checkFile.close();

    double avgError = 1;
    int validSamples = 0;

    for (int epoch = 0; epoch < maxEpoch; epoch++)
    {
        ifstream file(csvFile);
        if (!file.is_open())
        {
            cerr << "[ERROR] CSV dosyasi acilamadi: " << csvFile << endl;
            break;
        }

        // Header'ı kontrol et ve geçir
        string header;
        if (!getline(file, header))
        {
            cerr << "[ERROR] CSV header okunamadi!" << endl;
            file.close();
            break;
        }

        vector<double> x, y;
        double totalError = 0.0;
        int count = 0;
        int lineNum = 1; // Header = 0, data = 1'den başlar

        while (readCSVLine(file, x, y, inputSize, outputSize))
        {
            lineNum++;

            // Vektörleri kontrol et
            if (x.empty() || y.empty())
            {
                if (debug)
                    cerr << "[WARNING] Satır " << lineNum << " atlandı (boş vektör)" << endl;
                continue;
            }

            try
            {
                double error = cc.train(modelKey, x, y, targetError);

                // NaN veya inf kontrolü
                if (isnan(error) || isinf(error))
                {
                    if (debug)
                        cerr << "[WARNING] Satır " << lineNum << " - Geçersiz hata değeri: " << error << endl;
                    continue;
                }

                totalError += error;
                count++;
            }
            catch (const exception &e)
            {
                cerr << "[ERROR] Satır " << lineNum << " eğitim hatası: " << e.what() << endl;
                continue;
            }
        }

        file.close();

        if (count == 0)
        {
            cerr << "\n[ERROR] Hiç geçerli veri işlenemedi!" << endl;
            cerr << "[DEBUG] CSV formatını kontrol edin:" << endl;
            cerr << "  - İlk satır: input1,input2,...,target1,target2,..." << endl;
            cerr << "  - Sonraki satırlar: sayısal değerler (virgülle ayrılmış)" << endl;
            cerr << "  - Beklenen: " << inputSize << " input + " << outputSize << " output = "
                 << (inputSize + outputSize) << " sütun" << endl;
            break;
        }

        avgError = totalError / count;
        validSamples = count;

        progress_bar(epoch, maxEpoch, 40, avgError);

        // Model reboot mekanizmasi
        if ((epoch + 1) % 20 == 0)
        {
            int monitor = cc.models[modelKey].monitorNetwork(targetError);
            if (monitor == 4) // neuron added - reboot needed
            {
                // ONCE: Mevcut hatayi ve ogrenme oranini kaydet
                double currentLR = cc.models[modelKey].learningRate;
                vector<double> currentErrors = cc.models[modelKey].errors;

                // Eski topology'yi sakla
                vector<int> oldTopology;
                oldTopology.push_back(cc.models[modelKey].weights[0][0].size()); // input size
                for (const auto &layer : cc.models[modelKey].weights)
                {
                    oldTopology.push_back(layer.size());
                }

                // Simdi yeni topology'yi olustur
                vector<int> currentTopology;
                currentTopology.push_back(cc.models[modelKey].weights[0][0].size()); // input size
                for (const auto &layer : cc.models[modelKey].weights)
                {
                    currentTopology.push_back(layer.size());
                }

                cc.models.erase(modelKey);
                cc.addModel(modelKey, currentTopology, "tanh");

                // Ogrenme oranini ve hata gecmisini geri yukle
                cc.models[modelKey].learningRate = currentLR / 2;
                cc.models[modelKey].errors = currentErrors;

                if (debug)
                {
                    cout << "\n[INFO] Model rebooted: ";
                    for (size_t i = 0; i < currentTopology.size(); i++)
                    {
                        cout << currentTopology[i];
                        if (i < currentTopology.size() - 1)
                            cout << ",";
                    }
                    cout << endl;
                }

                log_saver("Model rebooted");

                epoch = -1; // restart training
                continue;
            }
            else if (monitor == 2)
            {
                cout << "\n[INFO] Stopping (slope close to 0)" << endl;
                break;
            }
        }

        // Hata hedefine ulasildi mi kontrol et
        if (avgError < targetError)
        {
            cout << "\n[INFO] Target error reached!" << endl;
            log_saver("Stopped due to target error...");
            break;
        }

        if (debug && epoch % 10 == 0)
        {
            cc.models[modelKey].errors.push_back(avgError);
            log_saver(to_string(avgError));
        }
    }

    cout << "\n\n[INFO] Egitim tamamlandi." << endl;
    cout << "  Ortalama hata: " << avgError << endl;
    cout << "  İşlenen örnek sayısı: " << validSamples << endl;
}

int load_embeddings_from_meta(
    const string &metaFile,
    unordered_map<string, vector<float>> &wordEmb,
    unordered_map<string, vector<float>> &commandEmb)
{
    ifstream in(metaFile);
    if (!in.is_open())
    {
#ifdef __EMSCRIPTEN__
        printf("[ERROR] Meta file not found: %s\n", metaFile.c_str());
#endif
        return 1;
    }
    string line;

    enum Section
    {
        NONE,
        WORDS,
        COMMANDS
    };
    Section section = NONE;

    while (getline(in, line))
    {
        if (line.empty())
            continue;

        if (line == "[WORD_EMBEDDINGS]")
        {
            section = WORDS;
            continue;
        }
        if (line == "[COMMAND_EMBEDDINGS]")
        {
            section = COMMANDS;
            continue;
        }
        if (line[0] == '[')
        {
            section = NONE;
            continue;
        }

        auto pos = line.find('=');
        if (pos == string::npos)
            continue;

        string key = line.substr(0, pos);
        string vecStr = line.substr(pos + 1);

        vector<float> vec;
        stringstream ss(vecStr);
        string num;

        // FIXED: Parse all numbers from the stringstream
        while (getline(ss, num, ','))
        {
            try
            {
                float v = stof(num);
                vec.push_back(v);
            }
            catch (...)
            {
                // Skip invalid numbers
                continue;
            }
        }

        if (section == WORDS)
            wordEmb[key] = vec;
        else if (section == COMMANDS)
            commandEmb[key] = vec;
    }
    return 0;
}

const int EMB_SIZE = 50;
#ifndef __EMSCRIPTEN__

#include <sqlite3.h>

unordered_map<string, vector<float>> load_embeddings_sqlite(
    const string &db_path,
    int EMB_SIZE)
{
    unordered_map<string, vector<float>> embeddings;
    sqlite3 *db;

    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
    {
        cerr << "[ERROR] SQLite acilamadi: " << sqlite3_errmsg(db) << endl;
        return embeddings;
    }

    const char *sql = "SELECT word, vector FROM embeddings;";
    sqlite3_stmt *stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        cerr << "[ERROR] SQL prepare hatasi\n";
        sqlite3_close(db);
        return embeddings;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        string word = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
        string vec_str = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));

        vector<float> vec;
        vec.reserve(EMB_SIZE);

        stringstream ss(vec_str);
        string val;
        while (getline(ss, val, ','))
        {
            vec.push_back(stof(val));
        }

        if ((int)vec.size() == EMB_SIZE)
        {
            embeddings[word] = vec;
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return embeddings;
}
#endif

vector<string> tokenize(
    const string &sentence,
    TokenizerMode mode = TokenizerMode::BPE,
    int subword_n = 3,
    ByteBPETokenizer *bpe_tokenizer = nullptr // ← EKLE
)
{
    vector<string> tokens;

    if (mode == TokenizerMode::WORD)
    {
        size_t i = 0, n = sentence.size();

        // tahmini kapasite (opsiyonel ama faydalı)
        tokens.reserve(n / 5);

        while (i < n)
        {
            while (i < n && sentence[i] == ' ')
                i++;
            size_t start = i;
            while (i < n && sentence[i] != ' ')
                i++;
            if (start < i)
                tokens.emplace_back(sentence.substr(start, i - start));
        }
        return tokens;
    }

    /* =========================
       SUBWORD TOKENIZER
       (char n-gram)
    ========================= */
    if (mode == TokenizerMode::SUBWORD)
    {
        for (size_t i = 0; i + subword_n <= sentence.size(); i++)
        {
            // boşluk içeren n-gram’leri atla
            bool valid = true;
            for (int j = 0; j < subword_n; j++)
            {
                if (sentence[i + j] == ' ')
                {
                    valid = false;
                    break;
                }
            }
            if (valid)
                tokens.emplace_back(sentence.substr(i, subword_n));
        }
        return tokens;
    }

    /* =========================
   BPE TOKENIZER
========================= */
    if (mode == TokenizerMode::BPE)
    {
        if (!bpe_tokenizer)
        {
            cerr << "[ERROR] BPE tokenizer verilmedi!\n";
            return tokens;
        }

        // Kelimelere ayır
        vector<string> words;
        size_t i = 0, n = sentence.size();
        while (i < n)
        {
            while (i < n && sentence[i] == ' ')
                i++;
            size_t start = i;
            while (i < n && sentence[i] != ' ')
                i++;
            if (start < i)
                words.push_back(sentence.substr(start, i - start));
        }

        // Her kelimeyi BPE ile encode et
        for (const auto &word : words)
        {
            // Özel token kontrolü
            if (word.size() > 2 && word[0] == '<' && word[word.size() - 1] == '>')
            {
                tokens.push_back(word);
                continue;
            }

            // Encode da hata var .:::::::
            //  BPE encode
            auto ids = bpe_tokenizer->encode(word);

            // IDs'i token string'lerine dönüştür
            if (ids.empty())
            {
                // Fallback: kelimenin kendisini token olarak kullan
                tokens.push_back(word);
                continue;
            }

            for (int id : ids)
            {
                auto decoded = bpe_tokenizer->decode({id});
                if (!decoded.empty())
                {
                    tokens.push_back(decoded);
                }
            }
        }

        return tokens;
    }

    return tokens;
}

/* =========================
   SENTENCE EMBEDDING
========================= */
vector<float> sentence_embedding(
    const string &sentence,
    unordered_map<string, vector<float>> &emb,
    TokenizerMode mode = TokenizerMode::BPE,
    int subword_n = 3,
    ByteBPETokenizer *bpe_tokenizer = nullptr // ← EKLE
)
{
    vector<float> result(EMB_SIZE, 0.0f);
    if (debugLog)
    {
        if (mode == TokenizerMode::BPE)
        {
            cout << "[sentence_embedding] mode: BPE";
        }
        else if (mode == TokenizerMode::SUBWORD)
        {
            cout << "\n[sentence_embedding] mode subword\n";
        }
        else if (mode == TokenizerMode::WORD)
        {
            cout << "\n[sentence_embedding] mode subword\n";
        }
        else
        {
            cout << "\n[sentence_embedding] mode unknown\n";
        }
    }

    auto tokens = tokenize(sentence, mode, subword_n, bpe_tokenizer); // ← GÜNCELLE

    if (debugLog)
    {

        if (mode == TokenizerMode::WORD)
        {
            cout << "[Sentence Embedding] Mode : " << "WORD \n";
        }
        else if (mode == TokenizerMode::SUBWORD)
            cout << "[Sentence Embedding] Mode : " << "SUBWORD \n";
        else if (mode == TokenizerMode::BPE)
            cout << "[Sentence Embedding] Mode : " << "BPE \n";
        cout << "[Sentence Embedding] ayrılmış tokenler:";
        for (auto i : tokens)
        {
            cout << i << ",";
        }
        cout << "\n";
    }

    int count = 0;
    for (auto &w : tokens)
    {
        auto it = emb.find(w);
        if (it == emb.end())
            continue;

        for (int i = 0; i < EMB_SIZE; i++)
            result[i] += it->second[i];

        count++;
    }

    if (count == 0)
        return result;

    float inv = 1.0f / count;
    for (float &v : result)
        v *= inv;

    return result;
}

// En yakın kelimeyi embedding'den bul
string findClosestWord(
    const vector<float> &target,
    unordered_map<string, vector<float>> &embeddings,
    const set<string> &excludeWords = {})
{
    string bestWord;
    float bestSim = -1e9;

    for (const auto &[word, emb] : embeddings)
    {
        // Exclude edilmiş kelimeleri atla
        if (excludeWords.count(word))
            continue;

        // Cosine similarity hesapla
        float dotProduct = 0.0f;
        float normA = 0.0f, normB = 0.0f;

        for (int i = 0; i < EMB_SIZE; i++)
        {
            dotProduct += target[i] * emb[i];
            normA += target[i] * target[i];
            normB += emb[i] * emb[i];
        }

        float similarity = dotProduct / (sqrt(normA) * sqrt(normB) + 1e-8f);

        if (similarity > bestSim)
        {
            bestSim = similarity;
            bestWord = word;
        }
    }

    return bestWord;
}

// Float vector'ü double vector'e dönüştür
vector<double> floatToDouble(const vector<float> &fvec)
{
    return vector<double>(fvec.begin(), fvec.end());
}

// Double vector'ü float vector'e dönüştür
vector<float> doubleToFloat(const vector<double> &dvec)
{
    return vector<float>(dvec.begin(), dvec.end());
}

// Text generation fonksiyonu
string generateText(
    CorticalColumn &cc,
    const string &modelKey,
    const string &prompt,
    unordered_map<string, vector<float>> &embeddings,
    int maxWords = 20)
{
    // Prompt'u embedding'e çevir
    auto promptEmb = sentence_embedding(prompt, embeddings, mode, 3, cc.models[modelKey].bpe_tokenizer.get());
    auto input = floatToDouble(promptEmb);

    string generatedText = prompt;
    set<string> recentWords; // Son kullanılan kelimeleri takip et

    for (int i = 0; i < maxWords; i++)
    {
        // Forward pass
        auto output = cc.forward(modelKey, input);
        auto outputFloat = doubleToFloat(output);

        // En yakın kelimeyi bul (son 5 kelimeyi exclude et)
        string nextWord = findClosestWord(outputFloat, embeddings, recentWords);

        if (nextWord.empty())
            break;

        generatedText += " " + nextWord;

        // Recent words'ü güncelle (sliding window)
        recentWords.insert(nextWord);
        if (recentWords.size() > 5)
        {
            recentWords.erase(recentWords.begin());
        }

        // Çıktıyı yeni input olarak kullan
        input = output;
    }

    return generatedText;
}

void helpPage()
{
    cout << "\n==== Etkilesimli Egitim Modu ====\n";
    cout << "Komutlar:\n";
    cout << "  exit/quit           - Cikis\n";
    cout << "  help                - Komutlar\n";
    cout << "  test                - Test XOR\n";

    cout << "  train [error] [ep]  - CSV'den egitim\n";
    cout << "  save <filename>     - Modeli kaydet\n";
    cout << "  load <filename>     - Model yukle\n";
    cout << "  generate <prompt>   - Text uret (embeddings gerekli)\n";
    cout << "  print               - Model yapisini goster\n";
    cout << "  graph               - Grafik olustur\n";
    cout << "  terminal <cmd>      - Terminal komutu calistir\n\n";
}

bool validateCSV(const string &csvFile, int expectedInputSize, int expectedOutputSize)
{
    ifstream file(csvFile);
    if (!file.is_open())
    {
        cerr << "[CSV ERROR] Dosya açılamadı\n"
             << "  Dosya: " << csvFile << endl;
        return false;
    }

    string line;
    int lineNum = 0;
    int expectedCols = expectedInputSize + expectedOutputSize;

    while (getline(file, line))
    {
        lineNum++;

        if (line.empty())
            continue;

        int commaCount = count(line.begin(), line.end(), ',');
        int foundCols = commaCount + 1;

        // =========================
        // HEADER KONTROLÜ
        // =========================
        if (lineNum == 1)
        {
            if (foundCols != expectedCols)
            {
                cerr << "\n[CSV HEADER ERROR]\n"
                     << "  Dosya            : " << csvFile << "\n"
                     << "  Satır            : 1 (HEADER)\n"
                     << "  Beklenen sütun   : " << expectedCols
                     << " (Input=" << expectedInputSize
                     << ", Target=" << expectedOutputSize << ")\n"
                     << "  Bulunan sütun    : " << foundCols << "\n"
                     << "  Beklenen virgül : " << (expectedCols - 1) << "\n"
                     << "  Bulunan virgül  : " << commaCount << "\n"
                     << "  Header içeriği  : " << line.substr(0, 150) << "\n";
                file.close();
                return false;
            }
            continue;
        }

        // =========================
        // SATIR SÜTUN SAYISI
        // =========================
        if (foundCols != expectedCols)
        {
            cerr << "\n[CSV ROW COLUMN ERROR]\n"
                 << "  Dosya            : " << csvFile << "\n"
                 << "  Satır            : " << lineNum << "\n"
                 << "  Beklenen sütun   : " << expectedCols
                 << " (Input=" << expectedInputSize
                 << ", Target=" << expectedOutputSize << ")\n"
                 << "  Bulunan sütun    : " << foundCols << "\n"
                 << "  Satır içeriği   : " << line.substr(0, 150) << "\n";
            file.close();
            return false;
        }

        // =========================
        // SAYISAL KONTROL (ilk 3 veri satırı)
        // =========================
        if (lineNum <= 4)
        {
            stringstream ss(line);
            string cell;
            int colNum = 0;

            while (getline(ss, cell, ','))
            {
                colNum++;

                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

                try
                {
                    stod(cell);
                }
                catch (...)
                {
                    cerr << "\n[CSV VALUE ERROR]\n"
                         << "  Dosya      : " << csvFile << "\n"
                         << "  Satır      : " << lineNum << "\n"
                         << "  Sütun      : " << colNum
                         << (colNum <= expectedInputSize ? " (INPUT)" : " (TARGET)") << "\n"
                         << "  Değer      : '" << cell << "'\n"
                         << "  Satır örn. : " << line.substr(0, 150) << "\n";
                    file.close();
                    return false;
                }
            }
        }
    }

    file.close();

    if (lineNum < 2)
    {
        cerr << "\n[CSV ERROR] Dosya sadece header içeriyor, veri yok!\n"
             << "  Dosya: " << csvFile << endl;
        return false;
    }

    if (debug)
    {
        cout << "[CSV OK] Doğrulama başarılı\n"
             << "  Dosya        : " << csvFile << "\n"
             << "  Veri satırı  : " << (lineNum - 1) << "\n"
             << "  Input boyutu : " << expectedInputSize << "\n"
             << "  Target boyutu: " << expectedOutputSize << "\n";
    }

    return true;
}

// main()'den önce ekleyin
void testXOR()
{
    CorticalColumn cc;
    vector<int> layers = {2, 4, 1};
    cc.addModel("xor", layers, "tanh");
    cc.models["xor"].learningRate = 0.1;

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    cout << "[XOR TEST]\n";
    for (int epoch = 0; epoch < 5000; epoch++)
    {
        double totalError = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            totalError += cc.train("xor", inputs[i], targets[i], 0.01);
        }
        if (epoch % 500 == 0)
        {
            cout << "Epoch " << epoch << " Error: " << (totalError / 4) << endl;
        }
    }

    cout << "\nResults:\n";
    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto out = cc.forward("xor", inputs[i]);
        cout << inputs[i][0] << " XOR " << inputs[i][1]
             << " = " << out[0] << " (expected: " << targets[i][0] << ")\n";
    }
}

struct InteractiveState
{
    CorticalColumn *cc;
    string modelKey;
    string csvFile;
    int inputSize;
    int outputSize;
    double targetError;
    int maxEpoch;
    TokenizerMode mode;

    unordered_map<string, vector<float>> embeddings;
    unordered_map<string, vector<float>> embeddingsForCommands;
};

// All Commands:
#ifndef __EMSCRIPTEN__
// Load embeddings
void loadEmbeddingsDB(InteractiveState &state)
{
    if (ifstream("LLM/Embeddings/embeddings.db").good())
    {
        state.embeddings =
            load_embeddings_sqlite("LLM/Embeddings/embeddings.db", EMB_SIZE);
        cout << "[INFO] Embeddings yuklendi ("
             << state.embeddings.size() << ")\n";
    }
    else
        cout << "[ERROR] embeddings.db bulunamadi\n";

    if (ifstream("LLM/Embeddings/embeddingsForCommands.db").good())
    {
        state.embeddingsForCommands =
            load_embeddings_sqlite("LLM/Embeddings/embeddingsForCommands.db", EMB_SIZE);
        cout << "[INFO] Command embeddings yuklendi ("
             << state.embeddingsForCommands.size() << ")\n";
    }
    else
        cout << "[ERROR] embeddingsForCommands.db bulunamadi\n";
}
#endif
void cmdHelp()
{
    helpPage();
}

// save
void cmdSave(InteractiveState &state, const string &filename)
{
    if (filename.empty())
    {
        cout << "[ERROR] save <filename>\n";
        return;
    }

    if (state.cc->saveModel(state.modelKey, filename))
        cout << "[INFO] Model kaydedildi: " << filename << endl;
    else
        cout << "[ERROR] Model kaydedilemedi\n";
}

// load
void cmdLoad(InteractiveState &state, const string &filename)
{
    if (filename.empty())
    {
        cout << "[ERROR] load <filename>\n";
        return;
    }

    if (state.cc->loadModel(filename, state.modelKey))
        cout << "[INFO] Model yuklendi: " << filename << endl;
    else
        cout << "[ERROR] Model yuklenemedi\n";
}

// generate
void cmdGenerate(InteractiveState &state, const string &sentence)
{
    if (sentence.empty())
    {
        cout << "[ERROR] generate <cumle>\n";
        return;
    }

    if (debugLog)
    {
        if (state.cc->models[state.modelKey].mode == TokenizerMode::WORD)
        {
            cout << "[cmdGenerate] Mode: WORD\n";
        }
        else if (state.cc->models[state.modelKey].mode == TokenizerMode::SUBWORD)
        {
            cout << "[cmdGenerate] Mode: SUBWORD\n";
        }
        else if (state.cc->models[state.modelKey].mode == TokenizerMode::BPE)
        {
            cout << "[cmdGenerate] Mode: BPE\n";
        }
    }
    // ✅ DOĞRU KONTROL
    if (state.cc->models[state.modelKey].bpe_tokenizer == nullptr)
    {
        cout << "[cmdGenerate] [ERROR] modelin bpe tokenizeri nullptr\n";
        return; // ❗ Buradan çık
    }

    auto emb = sentence_embedding(
        sentence,
        state.cc->models[state.modelKey].wordEmbeddings,
        state.cc->models[state.modelKey].mode,
        3,                                                   // subword_n
        state.cc->models[state.modelKey].bpe_tokenizer.get() // ← EKLE
    );

    auto input = floatToDouble(emb);

    auto output = state.cc->forward(state.modelKey, input);

    if (debugLog)
    {

        if (!output.empty())
        {
            double sum = 0.0;
            for (double v : output)
                sum += v;

            double mean = sum / output.size();

            cout << "[cmdGenerate] Output size: " << output.size() << endl;
            cout << "[cmdGenerate] Output mean: " << mean << endl;
        }
        else
        {
            cout << "[cmdGenerate] Output vector is empty" << endl;
        }
    }

    auto outFloat = doubleToFloat(output);

    string cmd = findClosestWord(outFloat, state.cc->models[state.modelKey]
                                               .commandEmbeddings);

    cout << "\n[KOMUT TAHMİNİ]\n";
    cout << "Girdi: " << sentence << "\n";
    cout << "Tahmin: " << cmd << "\n\n";
}

void generateForWasmTest(CorticalColumn &cc, const string &sentence, string modelKey)
{
    if (sentence.empty())
    {
        cout << "[ERROR] generate <cumle>\n";
        return;
    }

    auto emb = sentence_embedding(sentence, cc.models[modelKey].wordEmbeddings, cc.models[modelKey].mode, 3, cc.models[modelKey].bpe_tokenizer.get());

    auto input = floatToDouble(emb);

    auto output = cc.forward(modelKey, input);
    auto outFloat = doubleToFloat(output);

    string cmd = findClosestWord(outFloat, cc.models[modelKey]
                                               .commandEmbeddings);

    cout << "\n[KOMUT TAHMİNİ]\n";
    cout << "Girdi: " << sentence << "\n";
    cout << "Tahmin: " << cmd << "\n\n";
}

// train
void cmdTrain(InteractiveState &state)
{
    if (!validateCSV(
            state.csvFile,
            state.inputSize,
            state.outputSize))
    {
        cout << "[ERROR] CSV dogrulamasi basarisiz\n";
        return;
    }

    trainFromCSV(
        *state.cc,
        state.modelKey,
        state.csvFile,
        state.inputSize,
        state.outputSize,
        state.targetError,
        false,
        state.maxEpoch);
}

// print
void cmdPrint(InteractiveState &state)
{
    state.cc->models[state.modelKey]
        .printModelASCII(state.modelKey, true);
}

// graph (⚠️ WASM’DA KAPATILABİLİR)
void cmdGraph()
{
#ifndef __EMSCRIPTEN__
    system("python3 Helpers/network_changes_logs.py");
#else
    cout << "[DISABLED] graph komutu WASM ortaminda kapali\n";
#endif
}

// terminal (⚠️ GÜVENLİK NEDENİYLE)
void cmdTerminal(const string &cmd)
{
#ifndef __EMSCRIPTEN__
    system(cmd.c_str());
#else
    cout << "[BLOCKED] terminal komutu web ortaminda yasak\n";
#endif
}

static vector<string> split(const string &s)
{
    vector<string> tokens;
    istringstream iss(s);
    string tok;
    while (iss >> tok)
        tokens.push_back(tok);
    return tokens;
}

void interactiveTraining(InteractiveState &state, const string &line)
{
    if (line == "help")
        cmdHelp();
    else if (line.rfind("train", 0) == 0)
    {
#ifndef __EMSCRIPTEN__
        vector<string> args = split(line);

        // Varsayılanlar
        double oldTarget = state.targetError;
        int oldEpoch = state.maxEpoch;

        if (args.size() >= 2)
        {
            try
            {
                state.targetError = stod(args[1]);
            }
            catch (...)
            {
                cout << "[ERROR] Invalid targetError\n";
                state.targetError = oldTarget;
            }
        }

        if (args.size() >= 3)
        {
            try
            {
                state.maxEpoch = stoi(args[2]);
            }
            catch (...)
            {
                cout << "[ERROR] Invalid maxEpoch\n";
                state.maxEpoch = oldEpoch;
            }
        }

        cout << "[TRAIN] targetError=" << state.targetError
             << " maxEpoch=" << state.maxEpoch << "\n";

        cmdTrain(state);
#endif
    }

    else if (line == "print")
        cmdPrint(state);
    else if (line == "graph")
    {
#ifndef __EMSCRIPTEN__
        cmdGraph();
#endif
    }
    else if (line.rfind("save", 0) == 0)
    {
#ifndef __EMSCRIPTEN__
        string filename;

        if (line.size() > 4)
        {
            // "save " sonrası
            filename = line.substr(4);

            // Baştaki boşlukları temizle
            filename.erase(0, filename.find_first_not_of(" "));
        }

        cmdSave(state, filename);
#endif
    }

    else if (line.rfind("load", 0) == 0)
        cmdLoad(state, line.substr(5));
    else if (line.rfind("generate", 0) == 0)
        cmdGenerate(state, line.substr(9));
    else if (line.rfind("terminal", 0) == 0)
    {
#ifndef __EMSCRIPTEN__
        cmdTerminal(line.substr(8));
#endif
    }
}

SetupConfig setup(RunMode runMode, string modelName = "command_model", string csvFile = "command_data.csv")
{
    SetupConfig config;
    srand(time(nullptr));

    if (runMode == RunMode::CLI)
    {
        cout << "\n========================================\n";
        cout << "  KOMUT TAHMİN SİSTEMİ - NEURAL MODEL\n";
        cout << "========================================\n\n";

        cout << "Model adı [command_model]: ";
        getline(cin, config.modelName);
        if (config.modelName.empty())
            config.modelName = "command_model";

        cout << "Model Tokenizer Modu [WORD / SUBWORD / BPE (default)]:";
        string cmdMode;

        getline(cin, cmdMode);
        if (cmdMode.empty())
            config.mode = TokenizerMode::BPE;
        else if (cmdMode == "SUBWORD")
            config.mode = TokenizerMode::SUBWORD;
        else if (cmdMode == "WORD") // ← EKLE
            config.mode = TokenizerMode::WORD;
        else
            config.mode = TokenizerMode::BPE;

        if (config.mode == TokenizerMode::BPE)
        {
            cout << "BPE json dosya yolunu giriniz (default:" << bpe_model_path << "):";
            cmdMode = "";
            getline(cin, cmdMode);
            if (!cmdMode.empty())
                bpe_model_path = cmdMode;
        }

        cout << "Eğitim veri dosyası [LLM/Embeddings/command_data.csv]: ";
        getline(cin, config.csvFile);
        if (config.csvFile.empty())
            config.csvFile = "LLM/Embeddings/command_data.csv";
    }
    else
    {
        // 🔒 SERVICE MODE → HER ŞEY DEFAULT
        config.modelName = modelName;
        config.csvFile = csvFile;
        config.mode = TokenizerMode::BPE;
    }

    config.modelFile = "web/user_0000/" + config.modelName;
    config.csvAvailable = ifstream(config.csvFile).good();

    config.targetError = 0.05;
    config.maxEpoch = 10000;
    config.layers = {50, 256, 128, 50};

    if (runMode == RunMode::CLI || debugLog)
    {
        cout << "\n========================================\n";
        cout << "AYARLAR:\n";
        cout << "  Model: " << config.modelName << "\n";
        cout << "  Model dosyası: " << config.modelFile << "\n";
        cout << "  CSV dosyası: " << config.csvFile << "\n";
        cout << "  Hedef hata: " << config.targetError << "\n";
        cout << "  Max epoch: " << config.maxEpoch << "\n";
        if (config.mode == TokenizerMode::WORD)
            cout << "  Tokenizer Mod: " << "WORD" << "\n";
        else if (config.mode == TokenizerMode::SUBWORD)
            cout << "  Tokenizer Mod: " << "SUBWORD" << "\n";
        else if (config.mode == TokenizerMode::BPE)
            cout << "  Tokenizer Mod: " << "BPE" << "\n";

        cout << "========================================\n\n";
    }

    return config;
}

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif
#include <cstring>
static CorticalColumn g_cc;
static InteractiveState g_state;
static bool modelLoaded = false;

extern "C"
{

#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_KEEPALIVE
#endif
    int load_user_model(const char *bin, const char *bpeJson);

#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_KEEPALIVE
#endif
    const char *run_inference(const char *input);
}

int load_user_model(const char *bin, const char *bpeJson)
{
    g_state.embeddings.clear();
    g_state.embeddingsForCommands.clear();
    modelLoaded = false;

    g_cc = CorticalColumn(); // temiz başlat (çok önemli)
    g_cc.addModel("user", {50, 256, 128, 50}, "tanh");
    cout << "[load_user_model] BPE json dosya yolu :" << bpeJson << "\n";
    if (!g_cc.loadModel(bin, "user", bpeJson))
    {

        cout << "Model couldnt loaded.";
        return 1;
    }

    g_state.cc = &g_cc;
    g_state.modelKey = "user";
    g_state.embeddings = g_cc.models[g_state.modelKey].wordEmbeddings;
    g_state.embeddingsForCommands = g_cc.models[g_state.modelKey].commandEmbeddings;

    // ✅ YENİ KOD:
        if (g_state.mode == TokenizerMode::BPE)
    {
        if (bpeJson && bpeJson[0] != '\0') // ✅ bpeJson parametresini kontrol et
        {
            if (ifstream(bpeJson).good()) // ✅ JSON dosyasını kontrol et
            {
                g_cc.models[g_state.modelKey].bpe_tokenizer = std::make_unique<ByteBPETokenizer>();
                g_cc.models[g_state.modelKey].bpe_tokenizer->load(bpeJson); // ✅ JSON yükle
                if (debugLog)
                    cout << "[INFO] BPE tokenizer yuklendi: " << bpeJson << "\n";
            }
            else
            {
                if (debugLog)
                    cerr << "[ERROR] BPE tokenizer dosyası bulunamadı: " << bpeJson << "\n";
                return 1;
            }
        }
        else
        {
            if (debugLog)
                cerr << "[ERROR] BPE tokenizer verilmedi!\n";
            return 1;
        }
    }

    modelLoaded = true;
    return 1;
}

const char *run_inference(const char *input)
{
    static string result;

    if (!modelLoaded)
    {
        result = "MODEL_NOT_LOADED";
        return result.c_str();
    }

    if (g_state.embeddings.empty() ||
        g_state.embeddingsForCommands.empty())
    {
        result = "EMBEDDINGS_NOT_LOADED " + std::to_string(g_state.embeddings.size()) + " - " + std::to_string(g_state.embeddingsForCommands.size());
        return result.c_str();
    }

    auto emb = sentence_embedding(input, g_state.embeddings, mode, 3, g_state.cc->models[g_state.modelKey].bpe_tokenizer.get());
    auto out = g_cc.forward("user", floatToDouble(emb));
    auto cmd = findClosestWord(
        doubleToFloat(out),
        g_state.embeddingsForCommands);

    result = cmd;
    return result.c_str();
}

#ifndef __EMSCRIPTEN__
int main(int argc, char *argv[]) // 🆕 Parametreleri ekledik
{

    RunMode runMode = RunMode::CLI;

    // WASM test modu
    if (argc > 1 && string(argv[1]) == "--wasm-test")
    {
        cout << "[TEST MODE] WASM\n";
        runMode = RunMode::SERVICE;
        bpe_model_path = string(argv[2]) + ".json";
        if (debugLog)
        {
            cout << "[main] BPE model dosyası " << bpe_model_path << " olarak değiştirildi.\n";
        }

        // Manuel olarak embeddings yükle ve test et
        SetupConfig config;
        if (argc == 2)
        {
            config = setup(runMode);
        }
        else if (argc == 3)
        {
            config = setup(runMode, argv[2]);
        }
        else if (argc == 4)
        {
            config = setup(runMode, argv[2], argv[3]);
        }

        CorticalColumn cc;
        cc.addModel(config.modelName, config.layers, "tanh");

        // Model yükle
        if (cc.loadModel(config.modelFile, config.modelName, bpe_model_path))
        {
            cout << "[OK] Model yuklendi\n";
            cout << "  Word embeddings: "
                 << cc.models[config.modelName].wordEmbeddings.size() << "\n";
            cout << "  Command embeddings: "
                 << cc.models[config.modelName].commandEmbeddings.size() << "\n";

            // Test inference
            if (!cc.models[config.modelName].wordEmbeddings.empty())
            {
                auto testEmb = cc.models[config.modelName]
                                   .wordEmbeddings.begin()
                                   ->second;
                auto input = floatToDouble(testEmb);
                auto output = cc.forward(config.modelName, input);
                cout << "[TEST] Forward pass OK (output size: "
                     << output.size() << ")\n";

                generateForWasmTest(cc, "dosya kopyala", config.modelName);
            }
        }
        else
        {
            cout << "[ERROR] Model yuklenemedi!\n";
        }

        return 0;
    }

    // CLI çalıştırmak istersen:
    // RunMode mode = RunMode::CLI;
    SetupConfig config = setup(runMode);

    CorticalColumn cc;

    cc.addModel(
        config.modelName,
        config.layers,
        "tanh");

    // ✅ BPE TOKENIZER YÜKLE (eğer BPE modu seçildiyse)

    if (config.mode == TokenizerMode::BPE)
    {

        if (ifstream(bpe_model_path).good())
        {
            cc.models[config.modelName].bpe_tokenizer = std::make_unique<ByteBPETokenizer>();
            cc.models[config.modelName].bpe_tokenizer->load(bpe_model_path);
            cout << "[INFO] BPE tokenizer yuklendi: " << bpe_model_path << "\n";
        }
        else
        {
            cerr << "[ERROR] BPE model bulunamadi: " << bpe_model_path << "\n";
            return 1;
        }
    }

    if (ifstream(config.modelFile).good())
    {
        cc.loadModel(config.modelFile, config.modelName);
        cc.models[config.modelName].mode = config.mode;

        cout << "[DEBUG] hazır model yüklendi.\n";

        // ✅ 2. BPE tokenizer'ı kontrol et
        if (cc.models[config.modelName].bpe_tokenizer == nullptr && config.mode == TokenizerMode::BPE)
        {
            cc.models[config.modelName].bpe_tokenizer = std::make_unique<ByteBPETokenizer>();
            cc.models[config.modelName].bpe_tokenizer->load(bpe_model_path);
            cout << "[main] [INFO] BPE tokenizer yeniden yuklendi\n";
        }
    }
    else
    {
        // Yeni model oluşturuluyorsa BPE'yi yükle
        if (config.mode == TokenizerMode::BPE)
        {
            cc.models[config.modelName].bpe_tokenizer = std::make_unique<ByteBPETokenizer>();
            cc.models[config.modelName].bpe_tokenizer->load(bpe_model_path);
        }
    }

    InteractiveState state;
    state.cc = &cc;
    state.modelKey = config.modelName;
    state.csvFile = config.csvFile;
    state.inputSize = config.layers.front();
    state.outputSize = config.layers.back();
    state.targetError = config.targetError;
    state.maxEpoch = config.maxEpoch;
    state.mode = config.mode;

    loadEmbeddingsDB(state);

    // ✅ 2. EMBEDDINGS'LERİ MODEL'E AKTAR
    cc.models[config.modelName].wordEmbeddings = state.embeddings;
    cc.models[config.modelName].commandEmbeddings = state.embeddingsForCommands;
    cc.models[config.modelName].mode = state.mode;

    cout << "[INFO] Embeddings modele aktarildi:\n";
    cout << "  Words   : " << cc.models[config.modelName].wordEmbeddings.size() << "\n";
    cout << "  Commands: " << cc.models[config.modelName].commandEmbeddings.size() << "\n";

    // 🔴 SERVICE MODE → while yok
    if (runMode == RunMode::SERVICE)
    {
        cout << "[INFO] Model servis modunda hazir.\n";
        return 0;
    }

    // 🟢 CLI MODE
    cout << "[INFO] Interactive sistem hazir. 'help' yazabilirsiniz.\n\n";

    string line;
    while (true)
    {
        cout << "> ";
        getline(cin, line);

        if (line == "exit" || line == "quit")
            break;

        interactiveTraining(state, line);
    }

    return 0;
}
#endif