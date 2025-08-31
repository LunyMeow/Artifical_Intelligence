#include <iostream>
#include <fstream> // Dosya işlemleri için gerekli kütüphane
#include <cmath>   //fmod için gerekli
#include <vector>  // genişleyen liste için
using namespace std;

string isletimSistemi()
{
#ifdef _WIN32
    return "Windows";
#elif __linux__
    return "Linux";
#elif __APPLE__
    return "MacOS";
#else
    return "Bilinmiyor";
#endif
}

bool debug = false;
string defaultNeuronActivationType = "tanh";
string defaultOutActivation = "sigmoid";

bool is_multiple(double a, double b)
{
    return fmod(a, b) == 0; // a'nın b'ye bölümünden kalan 0 ise true döner
}

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

class CorticalColumn
{
public:
    CorticalColumn()
    {
        cout << "Cortical Column sinifi olusturuldu..." << endl;
    }

    string log_file = "network_changes.log";
    double learningRate = 0.3;
    double targetError = 0;
    int maxEpochForTargetError = 1000;
    vector<int> originalNetworkModel;
    double overfit_thresold = 0.9;
    bool useDynamicModelChanges = false;
    double targetEpoch;
    double enable_dynamic_control = true;

    vector<double> neuron_health_history;

    vector<string> parts;

    vector<double> error_history;

    double neuronHealtThreshould = 0.3;
    vector<string> change_log;

    int currentEpoch = 0;
    int lr_cooldown = 0;
    int lr_cooldown_period = 20;
    int last_lr_change_epoch = -1;
    int lrChanged = 0;

    vector<double> val_error_history;




    // ---------------------------
    // Log yazma fonksiyonu
    // ---------------------------
    void log(const string &message)
    {
        if (!debug)
        {
            return;
        }
        
        // change_log'a ekle
        if (debug)
            change_log.push_back(message);

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
};

int main()
{
    cout << "Bu sistem: " << isletimSistemi() << endl;

    for (int i = 0; i < 100; i++)
    {
        progress_bar(i, 99);
    }
    cout << endl;

    CorticalColumn corticalColumn;
    corticalColumn.log("Merhaba");
    return 0;
}
