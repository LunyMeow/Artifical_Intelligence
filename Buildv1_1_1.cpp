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
        int modelChangedSelf = 0;

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
            */

            double slope = computeErrorSlope(150);

            // Eger egim cok kucuk ama hata hala yuksek → kapasite yetmiyor olabilir
            double lastError = errors.empty() ? 9999 : errors.back();

            if (slope != 0.0 && (slope > -0.001 && slope < 0.001))
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
                        // SADECE gercekten pasif noronlari kaldir (rastgele katmanlardan DEGIL)
                        if(fixLayerImbalance()){
                            return 4;
                        }
                        
                        if (removeExcessNeuronsFromPyramid() || removeOnlyTrulyInactiveNeurons())
                        {
                            return 4;
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
                log_saver("Something is not right again" + to_string(neuronIndex )+ " >= " + to_string(weights[weightLayer].size()));
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

        progress_bar(epoch, maxEpoch, 40, avgError);

        // Model reboot mekanizmasi
        if ((epoch + 1) % 50 == 0)
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
                continue; // bu epoch'u atla
            }
            else if (monitor == 2)
            {
                cout << "Stopping" << endl;
            }
        }

        // Hata hedefine ulasildi mi kontrol et
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
    vector<int> layers = {4, 1, 2};
    cc.addModel("parity", layers, "sigmoid");

    // Etkilesimli mod baslat
    interactiveTraining(cc, "parity", "data.csv", layers[0], layers.back(), 0.02);

    return 0;
}
