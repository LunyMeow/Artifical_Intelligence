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
#include <set>

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
        int modelChangedSelf = 0;

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

            double slope = computeErrorSlope(150);

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
                learningRate /= 1.2;
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

        bool saveToFile(const string &filename)
        {
            ofstream file(filename, ios::binary);
            if (!file.is_open())
            {
                if (debug)
                    cout << "[ERROR] Dosya acilamadi: " << filename << endl;
                return false;
            }

            // 1. Aktivasyon tipini kaydet
            size_t typeLen = activationType.length();
            file.write(reinterpret_cast<const char *>(&typeLen), sizeof(typeLen));
            file.write(activationType.c_str(), typeLen);

            // 2. Learning rate ve diğer parametreleri kaydet
            file.write(reinterpret_cast<const char *>(&learningRate), sizeof(learningRate));
            file.write(reinterpret_cast<const char *>(&lrChanged), sizeof(lrChanged));
            file.write(reinterpret_cast<const char *>(&modelChangedSelf), sizeof(modelChangedSelf));

            // 3. Errors vector'ünü kaydet
            size_t errSize = errors.size();
            file.write(reinterpret_cast<const char *>(&errSize), sizeof(errSize));
            file.write(reinterpret_cast<const char *>(errors.data()), errSize * sizeof(double));

            // 4. Weights yapısını kaydet
            size_t numLayers = weights.size();
            file.write(reinterpret_cast<const char *>(&numLayers), sizeof(numLayers));

            for (const auto &layer : weights)
            {
                size_t numNeurons = layer.size();
                file.write(reinterpret_cast<const char *>(&numNeurons), sizeof(numNeurons));

                for (const auto &neuron : layer)
                {
                    size_t numWeights = neuron.size();
                    file.write(reinterpret_cast<const char *>(&numWeights), sizeof(numWeights));
                    file.write(reinterpret_cast<const char *>(neuron.data()), numWeights * sizeof(double));
                }
            }

            // 5. Biases yapısını kaydet
            size_t numBiasLayers = biases.size();
            file.write(reinterpret_cast<const char *>(&numBiasLayers), sizeof(numBiasLayers));

            for (const auto &layer : biases)
            {
                size_t numBiases = layer.size();
                file.write(reinterpret_cast<const char *>(&numBiases), sizeof(numBiases));
                file.write(reinterpret_cast<const char *>(layer.data()), numBiases * sizeof(double));
            }

            file.close();
            if (debug)
                cout << "[INFO] Model basariyla kaydedildi: " << filename << endl;
            return true;
        }

        // Modeli dosyadan yükle
        bool loadFromFile(const string &filename)
        {
            ifstream file(filename, ios::binary);
            if (!file.is_open())
            {
                if (debug)
                    cout << "[ERROR] Dosya acilamadi: " << filename << endl;
                return false;
            }

            // 1. Aktivasyon tipini oku
            size_t typeLen;
            file.read(reinterpret_cast<char *>(&typeLen), sizeof(typeLen));
            activationType.resize(typeLen);
            file.read(&activationType[0], typeLen);

            // 2. Learning rate ve diğer parametreleri oku
            file.read(reinterpret_cast<char *>(&learningRate), sizeof(learningRate));
            file.read(reinterpret_cast<char *>(&lrChanged), sizeof(lrChanged));
            file.read(reinterpret_cast<char *>(&modelChangedSelf), sizeof(modelChangedSelf));

            // 3. Errors vector'ünü oku
            size_t errSize;
            file.read(reinterpret_cast<char *>(&errSize), sizeof(errSize));
            errors.resize(errSize);
            file.read(reinterpret_cast<char *>(errors.data()), errSize * sizeof(double));

            // 4. Weights yapısını oku
            size_t numLayers;
            file.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));
            weights.resize(numLayers);

            for (auto &layer : weights)
            {
                size_t numNeurons;
                file.read(reinterpret_cast<char *>(&numNeurons), sizeof(numNeurons));
                layer.resize(numNeurons);

                for (auto &neuron : layer)
                {
                    size_t numWeights;
                    file.read(reinterpret_cast<char *>(&numWeights), sizeof(numWeights));
                    neuron.resize(numWeights);
                    file.read(reinterpret_cast<char *>(neuron.data()), numWeights * sizeof(double));
                }
            }

            // 5. Biases yapısını oku
            size_t numBiasLayers;
            file.read(reinterpret_cast<char *>(&numBiasLayers), sizeof(numBiasLayers));
            biases.resize(numBiasLayers);

            for (auto &layer : biases)
            {
                size_t numBiases;
                file.read(reinterpret_cast<char *>(&numBiases), sizeof(numBiases));
                layer.resize(numBiases);
                file.read(reinterpret_cast<char *>(layer.data()), numBiases * sizeof(double));
            }

            // Cache'leri temizle
            layerInputs.clear();
            layerOutputs.clear();

            file.close();
            if (debug)
                cout << "[INFO] Model basariyla yuklendi: " << filename << endl;
            return true;
        }

    private:
        vector<vector<double>> layerInputs;
        vector<vector<double>> layerOutputs;
    };

    unordered_map<string, Network> models;
    string defaultNeuronActivationType = "sigmoid";
    double learningRate = 1.3;

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

    bool saveModel(const string &modelKey, const string &filename)
    {
        auto it = models.find(modelKey);
        if (it == models.end())
        {
            if (debug)
                cout << "[ERROR] Model bulunamadi: " << modelKey << endl;
            return false;
        }

        // Model dosyası adını ve key'i kaydet
        ofstream metaFile(filename + ".meta");
        if (metaFile.is_open())
        {
            metaFile << modelKey << endl;
            metaFile.close();
        }

        return it->second.saveToFile(filename);
    }

    // Model yükleme fonksiyonu
    bool loadModel(const string &filename, const string &newModelKey = "")
    {
        string modelKey = newModelKey;

        // Eğer key verilmemişse, meta dosyadan oku
        if (modelKey.empty())
        {
            ifstream metaFile(filename + ".meta");
            if (metaFile.is_open())
            {
                getline(metaFile, modelKey);
                metaFile.close();
            }
            else
            {
                modelKey = "loaded_model";
            }
        }

        Network net;
        if (!net.loadFromFile(filename))
        {
            return false;
        }

        models[modelKey] = net;
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

    double avgError = 1;

    for (int epoch = 0; epoch < maxEpoch; epoch++)
    {
        ifstream file(csvFile);
        if (!file.is_open())
        {
            cerr << "CSV dosyasi acilamadi: " << csvFile << endl;
            break;
        }
        ensureCSVHeader(csvFile, inputSize, outputSize);

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

                // Eski modeli yazdir
                cout << "Old model topology: ";
                for (size_t i = 0; i < oldTopology.size(); i++)
                {
                    cout << oldTopology[i];
                    if (i < oldTopology.size() - 1)
                        cout << ",";
                }
                cout << endl;

                // Simdi yeni topology'yi olustur
                vector<int> currentTopology;
                currentTopology.push_back(cc.models[modelKey].weights[0][0].size()); // input size
                for (const auto &layer : cc.models[modelKey].weights)
                {
                    currentTopology.push_back(layer.size());
                }

                cc.models.erase(modelKey);
                cc.addModel(modelKey, currentTopology, "sigmoid");

                // Ogrenme oranini ve hata gecmisini geri yukle
                cc.models[modelKey].learningRate = currentLR / 2;
                cc.models[modelKey].errors = currentErrors;

                cout << "Model rebooted with new topology: ";
                for (size_t i = 0; i < currentTopology.size(); i++)
                {
                    cout << currentTopology[i];
                    if (i < currentTopology.size() - 1)
                        cout << ",";
                }
                cout << endl;
                if (debug)
                {
                    cout << "Model Rebooted" << endl;
                }

                log_saver("Model rebooted");

                epoch = -1; // restart training (next iteration will be epoch 0)
                continue;   // bu epoch'u atla
            }
            else if (monitor == 2)
            {
                cout << "Stopping" << endl;
            }
        }

        // Hata hedefine ulasildi mi kontrol et
        if (avgError < targetError)
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

const int EMB_SIZE = 50;

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

/* =========================
   EMBEDDING LOAD
========================= */
unordered_map<string, vector<float>> load_embeddings(const string &file)
{
    unordered_map<string, vector<float>> emb;
    ifstream fin(file);

    string line;
    while (getline(fin, line))
    {
        stringstream ss(line);
        string word;
        ss >> word;

        vector<float> vec(EMB_SIZE);
        for (int i = 0; i < EMB_SIZE; i++)
        {
            ss >> vec[i];
        }
        emb[word] = vec;
    }
    return emb;
}

/* =========================
   TOKENIZER
========================= */
vector<string> tokenize(const string &sentence)
{
    vector<string> tokens;
    stringstream ss(sentence);
    string word;
    while (ss >> word)
    {
        tokens.push_back(word);
    }
    return tokens;
}

/* =========================
   SENTENCE EMBEDDING
========================= */
vector<float> sentence_embedding(
    const string &sentence,
    unordered_map<string, vector<float>> &emb)
{
    vector<float> result(EMB_SIZE, 0.0f);
    auto tokens = tokenize(sentence);

    int count = 0;
    for (auto &w : tokens)
    {
        auto it = emb.find(w);
        if (it == emb.end())
            continue;

        for (int i = 0; i < EMB_SIZE; i++)
        {
            result[i] += it->second[i];
        }
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
    auto promptEmb = sentence_embedding(prompt, embeddings);
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
    cout << "  train [error] [ep]  - CSV'den egitim\n";
    cout << "  save <filename>     - Modeli kaydet\n";
    cout << "  load <filename>     - Model yukle\n";
    cout << "  generate <prompt>   - Text uret (embeddings gerekli)\n";
    cout << "  print               - Model yapisini goster\n";
    cout << "  graph               - Grafik olustur\n";
    cout << "  terminal <cmd>      - Terminal komutu calistir\n\n";
}

void interactiveTraining(
    CorticalColumn &cc,
    const string &modelKey,
    const string &csvFile,
    int inputSize,
    int outputSize,
    double targetError = 0.05,
    int maxEpoch = 1000)
{

    helpPage();

    unordered_map<string, vector<float>> embeddings;

    if (ifstream("LLM/Embeddings/embeddings.db").good())
    {
        embeddings = load_embeddings_sqlite("LLM/Embeddings/embeddings.db", EMB_SIZE);
        cout << "[INFO] Embeddings yuklendi ("
             << embeddings.size() << " kelime)\n";
    }
    else
    {
        cout << "[ERROR] embeddings.db bulunamadi...\n";
    }

    while (true)
    {
        cout << "> ";
        string line;
        getline(cin, line);

        if (line == "exit" || line == "quit")
        {
            break;
        }
        else if (line == "help")
        {
            helpPage();
            continue;
        }
        // SAVE KOMUTU
        else if (line.rfind("save", 0) == 0)
        {
            stringstream ss(line);
            string cmd, filename;
            ss >> cmd >> filename;

            if (filename.empty())
            {
                cout << "[ERROR] Kullanim: save <filename>" << endl;
                continue;
            }

            if (cc.saveModel(modelKey, filename))
            {
                cout << "[INFO] Model kaydedildi: " << filename << endl;
            }
            else
            {
                cout << "[ERROR] Model kaydedilemedi!" << endl;
            }
            continue;
        }
        // LOAD KOMUTU
        else if (line.rfind("load", 0) == 0)
        {
            stringstream ss(line);
            string cmd, filename;
            ss >> cmd >> filename;

            if (filename.empty())
            {
                cout << "[ERROR] Kullanim: load <filename>" << endl;
                continue;
            }

            if (cc.loadModel(filename, modelKey))
            {
                cout << "[INFO] Model yuklendi: " << filename << endl;
            }
            else
            {
                cout << "[ERROR] Model yuklenelemedi!" << endl;
            }
            continue;
        }
        // GENERATE KOMUTU
        else if (line.rfind("generate", 0) == 0)
        {
            if (embeddings.empty())
            {
                cout << "[ERROR] Embeddings yuklenmemis! 'embeddings.txt' dosyasi gerekli." << endl;
                continue;
            }

            string restOfLine = line.substr(9); // "generate " sonrası
            restOfLine.erase(0, restOfLine.find_first_not_of(" \t"));

            if (restOfLine.empty())
            {
                cout << "[ERROR] Kullanim: generate <prompt> [-e <num>]" << endl;
                cout << "         Ornek: generate hello world" << endl;
                cout << "         Ornek: generate hello world -e 5" << endl;
                continue;
            }

            // -e parametresini kontrol et
            int maxWords = 20; // default değer
            string prompt = restOfLine;

            size_t ePos = restOfLine.rfind("-e");
            if (ePos != string::npos)
            {
                // -e bulundu, sayıyı parse et
                string beforeE = restOfLine.substr(0, ePos);
                string afterE = restOfLine.substr(ePos + 2);

                // beforeE'den boşlukları temizle
                beforeE.erase(beforeE.find_last_not_of(" \t") + 1);

                // afterE'den sayıyı al
                afterE.erase(0, afterE.find_first_not_of(" \t"));

                try
                {
                    maxWords = stoi(afterE);
                    if (maxWords <= 0 || maxWords > 1000)
                    {
                        cout << "[WARNING] Gecersiz sayi: " << afterE << ", default 20 kullaniliyor" << endl;
                        maxWords = 20;
                    }
                    prompt = beforeE;
                }
                catch (...)
                {
                    cout << "[WARNING] -e parametresi parse edilemedi, default 20 kullaniliyor" << endl;
                }
            }

            prompt.erase(0, prompt.find_first_not_of(" \t"));
            prompt.erase(prompt.find_last_not_of(" \t") + 1);

            if (prompt.empty())
            {
                cout << "[ERROR] Prompt bos olamaz!" << endl;
                continue;
            }

            cout << "\n[GENERATING TEXT...] (max " << maxWords << " words)\n";
            string generated = generateText(cc, modelKey, prompt, embeddings, maxWords);
            cout << "\nGenerated Text:\n"
                 << generated << "\n"
                 << endl;
            continue;
        }
        else if (line == "graph")
        {
            system("python3 Helpers/network_changes_logs.py");
            continue;
        }
        else if (line.rfind("terminal", 0) == 0)
        {
            string cmd = line.substr(8);
            cmd.erase(0, cmd.find_first_not_of(" \t"));

            if (cmd.empty())
            {
                cout << "[ERROR] terminal komutu bos olamaz" << endl;
                continue;
            }

            cout << "[TERMINAL] calistiriliyor: " << cmd << endl;
            system(cmd.c_str());
            continue;
        }
        else if (line.rfind("train", 0) == 0)
        {
            stringstream ss(line);
            string cmd;
            ss >> cmd;

            double newTargetError;
            int newMaxEpoch;

            if (ss >> newTargetError)
            {
                targetError = newTargetError;
                cout << "[INFO] targetError guncellendi: " << targetError << endl;

                if (ss >> newMaxEpoch)
                {
                    maxEpoch = newMaxEpoch;
                    cout << "[INFO] maxEpoch guncellendi: " << maxEpoch << endl;
                }
            }

            trainFromCSV(cc, modelKey, csvFile, inputSize, outputSize, targetError, false, maxEpoch);
            continue;
        }
        else if (line == "print")
        {
            cc.models[modelKey].printModelASCII(modelKey, true);
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

        // Ana kodda kullanim:
        if (line == "train")
        {
            trainFromCSV(cc, modelKey, csvFile, inputSize, outputSize, targetError, false, maxEpoch);
            continue;
        }

        // Eger target verilmisse egit
        if (!targets.empty())
        {
            trainFromCSV(cc, modelKey, csvFile, inputSize, outputSize, targetError, true, maxEpoch);
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



// =========================
//   MAIN
//=========================
int main() {
    auto embeddings = load_embeddings("embeddings.txt");

    string sentence;
    cout << "Sentence: ";
    getline(cin, sentence);

    auto vec = sentence_embedding(sentence, embeddings);

    cout << "\nSentence Vector (first 10 dims):\n";
    for (int i = 0; i < 10; i++) {
        cout << vec[i] << " ";
    }
    cout << endl;

    return 0;
}


*/

/*


ÖRNEK KULLANIM:
> save my_model.bin              # Modeli kaydet
> load my_model.bin              # Modeli yükle
> generate hello world           # 20 kelime üret
> train 0.01 5000               # CSV'den eğitim
> print                         # Model yapısını göster
> graph                         # Grafik çiz
> terminal python test.py       # Terminal komutu

*/

int main()
{
    srand(time(nullptr));
    CorticalColumn cc;

    // Model oluştur veya yükle
    vector<int> layers = {50, 1024, 512, 128, 50};
    cc.addModel("text_gen", layers, "tanh");

    // Eğer kaydedilmiş model varsa:
    // cc.loadModel("saved_model.bin", "text_gen");

    // Interactive mode başlat
    interactiveTraining(cc, "text_gen", "data.csv", layers[0], layers.back(), 0.05);

    return 0;
}