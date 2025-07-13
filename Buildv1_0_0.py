import csv
import random
import sys
import time
import traceback
from typing import Dict, List, Optional
import matplotlib

import modeltrainingprogram
import os
if os.name == 'nt':
    print("Windows sistemi")
    matplotlib.use('TkAgg')
else:
    print("Linux veya MacOS sistemi")
    matplotlib.use('Agg')  # Bu satÄ±rÄ± plt importundan Ã¶nce koy


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable




import pyqtgraph as pg
import networkx as nx
import numpy as np
from pyqtgraph.Qt import QtGui , QtCore  # Add this import
from pyqtgraph.Qt import QtWidgets


import pickle
import gzip
from collections import defaultdict


from scipy import stats
import json

import mplcursors  # Fare etkileÅŸimi iÃ§in



import signal
import json
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

from NeuronAndConnection import Neuron ,Connection



debug=False

defaultNeuronActivationType='tanh'
defaultOutActivation = 'sigmoid' 



def is_multiple(a, b):
    return a % b == 0 if b != 0 else False

def progress_bar(current, total=50, bar_length=40):
    percent = int(100 * current / total)
    filled = int(bar_length * percent // 100)
    bar = '#' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f"\r[{bar}] {percent}%  ({current}/{total})")
    sys.stdout.flush()









class CorticalColumn:
    
    def __init__(self,log_file="network_changes.log", learning_rateArg=0.3,targetError=None,
                 maxEpochForTargetError=1000,originalNetworkModel=None,
                 overfit_threshold=0.9,useDynamicModelChanges=True,targetEpoch=None):
        
        
        self.firstLearningRate = learning_rateArg
        self.learningRate = learning_rateArg
        self.neuronHealtThreshould = 0.3
        self.change_log = []  # DeÄŸiÅŸiklik loglarÄ±nÄ± tutacak liste
        self.current_epoch = 0  # Mevcut epoch bilgisi
        self.log_file = log_file
        self.log_start_time = time.time()
        # Loss history'yi de burada tutmak isterseniz:
        self.loss_history = []

        self.lr_cooldown = 0
        self.lr_cooldown_period = 20  # Increased from 10
        self.last_lr_change_epoch = -float('inf')
        self.lrChanged = 0

        self.maxEpochForTargetError=maxEpochForTargetError
        self.targetError=targetError
        if debug and self.targetError != None:print(f"Hedef hata deÄŸerine eriÅŸilememesi durumunda minimum hata deÄŸerinin sÄ±fÄ±rlanmasÄ± iÃ§in gereken epoch sayÄ±sÄ± {self.maxEpochForTargetError} deÄŸerinin katlarÄ± olarak belirlendi.")
        self.originalNetworkModel = originalNetworkModel
        self.neuron_health_history = {}  # Dictionary to store health history of neurons

                # Overfitting control
        self.overfit_threshold = overfit_threshold
        self.val_error_history = []
        self.useDynamicModelChanges = useDynamicModelChanges

        self.targetEpoch=targetEpoch

        
        self.parts={}



        

        # Log dosyasÄ±nÄ± baÅŸlat (varsa sil, yenisini oluÅŸtur)
        with open(self.log_file, 'w') as f:
            f.write("")  # BoÅŸ dosya oluÅŸtur
    

    def createPartOfAI(self,partName):
        self.parts[partName] = self.someAI(partName=partName,outerCorticalClass=self)

    def _get_timestamp(self):
        """GeÃ§erli zaman damgasÄ±nÄ± ISO formatÄ±nda dÃ¶ndÃ¼rÃ¼r"""
        return datetime.now().isoformat()
    
    def _append_to_log(self, entry):
        """Log giriÅŸini hem bellekte hem de dosyaya ekler"""
        self.change_log.append(entry)
        with open(self.log_file, 'a') as f:
            json.dump(entry, f)
            f.write("\n")

    def log_change(self, change_type, details, partName=None):
        """Değişiklikleri loglayan yardımcı fonksiyon"""
        log_entry = {
            'epoch': self.current_epoch,
            'timestamp': self._get_timestamp(),
            'elapsed_seconds': round(time.time() - self.log_start_time, 2),
            'type': change_type,
            'details': details,
            'partName': partName if partName is not None else "None"
        }

        if partName is not None and partName in self.parts:
            log_entry['network_state'] = {
                'layer_sizes': [len(layer) for layer in self.parts[partName].layers],
                'total_neurons': sum(len(layer) for layer in self.parts[partName].layers)
            }
        else:
            log_entry['network_state'] = {
                'layer_sizes': [],
                'total_neurons': 0
            }

        self._append_to_log(log_entry)


    def calculate_slope(self, error_history, start_epoch=None, end_epoch=None):
        """
        Belirtilen epoch aralÄ±ÄŸÄ±ndaki hata deÄŸerlerinin eÄŸimini hesaplar

        Parametreler:
        error_history (list): Hata deÄŸerlerinin listesi
        start_epoch (int): BaÅŸlangÄ±Ã§ epoch indeksi (None ise son %20'nin baÅŸlangÄ±cÄ±)
        end_epoch (int): BitiÅŸ epoch indeksi (None ise son epoch)

        Returns:
        float: Hata eÄŸimi (pozitif = hatalar artÄ±yor, negatif = hatalar azalÄ±yor)
        float: RÂ² deÄŸeri (eÄŸimin gÃ¼venilirliÄŸi, 1'e yakÄ±n = gÃ¼venilir)
        """
        if not error_history or len(error_history) < 2:
            return 0.0, 0.0

        # VarsayÄ±lan aralÄ±klarÄ± ayarla
        if start_epoch is None:
            start_epoch = int(len(error_history) * 0.8)  # Son %20'nin baÅŸlangÄ±cÄ±
        if end_epoch is None:
            end_epoch = len(error_history) - 1  # Son epoch

        # GeÃ§erli aralÄ±ÄŸÄ± kontrol et
        start_epoch = max(0, min(start_epoch, len(error_history)-1))
        end_epoch = max(0, min(end_epoch, len(error_history)-1))

        if start_epoch >= end_epoch:
            return 0.0, 0.0

        # SeÃ§ilen aralÄ±ktaki hata ve epoch deÄŸerlerini al
        selected_errors = error_history[start_epoch:end_epoch+1]
        epochs = list(range(start_epoch, end_epoch+1))

        # Lineer regresyon ile eÄŸim ve RÂ² deÄŸerini hesapla
        slope, intercept, r_value, _, _ = stats.linregress(epochs, selected_errors)
        r_squared = r_value**2

        return slope, r_squared

    def monitor_network(self,avg_error,partName,eveyXEpochAdaptNetwork=10):
        """
        Enhanced monitoring with plateau detection and improved learning rate adjustments.
        This function analyzes error trends to manage learning rate changes for optimal training.
        """
        # Add the current error to history
        self.loss_history.append(avg_error)
        window_size = min(70, len(self.loss_history))
        startEpoch=len(self.loss_history) - window_size
        slope, r_square = self.calculate_slope(
                self.loss_history, 
                start_epoch=startEpoch, 
                end_epoch=None
        )

        if slope<1e-6 and slope >-1e-6 and debug:
            self.log_change('its linear now',{
                'between':f"{startEpoch}-now",
                'slope':slope,
                'partName':partName
            },partName=partName)
        


        #print(is_multiple(self.current_epoch,self.maxEpochForTargetError),slope)
        if self.maxEpochForTargetError is not None:
            if is_multiple(self.current_epoch+1,self.maxEpochForTargetError) and slope > -1e-8:
                minError = np.min(self.loss_history)
                increaseValue=0.9
                if debug:
                    self.log_change('minimum Error Changed and Neural network resetted',{
                        'current target error':self.targetError,
                        'minimum Error':minError,
                        'now error':self.loss_history[-1],
                        'current lr':self.learningRate,
                        'first lr':self.firstLearningRate,
                        'increase value':increaseValue,
                        'now target value':minError/increaseValue,
                        'partName':partName

                    },partName=partName)
                self.learningRate = self.firstLearningRate
                self.targetError = minError/increaseValue
                #print("---------------------------------------------")
                if self.originalNetworkModel is not None:
                    self.parts[partName].setLayers(self.originalNetworkModel)
                return self.learningRate,self.targetError
            

        # Log debug information about error trend if needed
        if debug:
            if len(self.loss_history) % 50 == 0 :
                
                self.log_change('error_trend_analysis', {
                    'window_size': window_size,
                    'slope': slope,
                    'r_square': r_square,
                    'current_lr': self.learningRate,
                    'partName':partName
                },partName=partName)
        
        
        
        # Check if we can update learning rate (not in cooldown period)
        if self.current_epoch - self.last_lr_change_epoch >= self.lr_cooldown_period:
            # Apply a small automatic decay to prevent stagnation (0.9995^1000 â‰ˆ 0.61, so gradual)
            # This helps ensure the learning rate doesn't stay too high for too long
            
            # Get new learning rate based on loss trend
            new_lr = self.update_learning_rate(self.learningRate, self.loss_history,slopeArg=slope)
            
            # If learning rate changed significantly, reset cooldown counter

            self.learningRate = new_lr
            self.last_lr_change_epoch = self.current_epoch


        if self.current_epoch % eveyXEpochAdaptNetwork == 2 :  # Her 10 epoch'ta bir
            self.log_change('avg_error_debug', {
                'avg_error': avg_error,
                'partName':partName,

            },partName=partName)

            if self.targetError is not None:
                progress_bar(self.loss_history[1]-avg_error,total=self.loss_history[1])
            else:
                progress_bar(current=self.current_epoch,total=self.targetEpoch)
            
            if self.useDynamicModelChanges:
                self.adapt_network_structure(avg_error,partName=partName)
            

        
        return self.learningRate,None
    
    def update_learning_rate(self, current_lr, loss_history, 
                             patience=60, # Increased from 20
                             min_lr=0.1, max_lr=4.0,  # Reduced max_lr
                             factor=0.0005,  # Reduced from 0.002
                             threshold=1e-2, increase_threshold=5e-2,slopeArg=None):  # More conservative increase threshold
        """
        Updates learning rate based on loss history trends with improved stability.
        
        Parameters:
        - current_lr: Current learning rate
        - loss_history: History of loss values
        - patience: Number of epochs to consider for trend analysis
        - min_lr/max_lr: Bounds for learning rate
        - factor: Rate of change factor for learning rate adjustments
        - threshold: Minimum improvement threshold to maintain current lr
        - increase_threshold: Threshold to consider increasing lr
        
        Returns:
        - Updated learning rate value
        """




        # Return current_lr if we don't have enough data
        if len(loss_history) < 2 * patience:
            return current_lr
        
            # Yeni: Gradyan Stabilizasyonu
        if abs(slopeArg) > 0.1:  # Ani hata deÄŸiÅŸimlerinde
            emergency_factor = 0.2 if slopeArg > 0 else 0.1
            new_lr = current_lr * (1 - emergency_factor)
            self.log_change('emergency change bc slope',{
                'slope':slopeArg,
                'emergency_factor':emergency_factor,
                'old_lr':current_lr,
                'new_lr':new_lr,
                'change':new_lr-current_lr

            })
            return max(new_lr, min_lr)
        if len(loss_history)>=100:
            if abs(loss_history[-1]-loss_history[-100]) < 0.0002 :
                new_lr=0.7
                self.log_change('slope is close to 0 lr_up',{'slope':slopeArg,'old_lr':current_lr,'new_lr':new_lr,'last 1 and last 100 error different':abs(loss_history[-1]-loss_history[-100])})
                return min(new_lr,max_lr)
        # Analyze recent and previous loss trends
        recent_losses = loss_history[-patience:]
        previous_losses = loss_history[-(2 * patience):-patience]
        
        # Yeni: Cosine Annealing Esintili Decay
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.current_epoch / 1000))
        decayed_lr = current_lr * cosine_decay
        
        # Calculate average losses
        avg_old = sum(previous_losses) / patience if previous_losses else 0
        avg_new = sum(recent_losses) / patience if recent_losses else 0
        
        # Calculate relative improvement
        improvement = (avg_old - avg_new) / avg_old if avg_old != 0 else 0
        
        # Check for error increase (negative improvement)
        if improvement < -0.01:  # More sensitive to error increases (1% worsening)
            # Error is increasing - reduce learning rate more aggressively
            reduction_factor = factor * 3  # More aggressive reduction than before
            new_lr = current_lr * (1 - reduction_factor)
            new_lr = max(new_lr, min_lr)
            
            self.log_change('lr_emergency_reduction', {
                'before_lr': current_lr,
                'new_lr': new_lr,
                'change': new_lr - current_lr,
                'reason': f"Error increasing: improvement={improvement}",
                'lrChanged': self.lrChanged
            })
            
            # Reset change counter when we make an emergency reduction
            self.lrChanged = -1
            
            if debug:print(f"Error increasing! Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")
            return new_lr
        
        # Case 1: Insufficient improvement - reduce learning rate
        if improvement < threshold:
            # Check if we're at minimum learning rate already
            if current_lr <= min_lr * 1.01:
                # At minimum, try a small increase to escape potential local minimum
                # Less aggressive than before
                small_increase_factor = 1.5  # Was 1/(factor*10) which was too large
                new_lr = current_lr * small_increase_factor
                new_lr = min(new_lr, max_lr)
                
                self.log_change('lr_small_up', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'increase_factor': small_increase_factor,
                    'reason': f'min_limit_reached, trying escape',
                    'lrChanged': self.lrChanged
                })
                
                self.lrChanged = 0  # Reset change counter
                if debug:print(f"Min limit reached. Learning rate slightly increased: {current_lr:.6f} -> {new_lr:.6f}")
                return new_lr
            else:
                # Standard reduction
                new_lr = current_lr * (1 - factor)
                new_lr = max(new_lr, min_lr)
                
                self.log_change('lr_down', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'factor': factor,
                    'reason': f"improvement: {improvement:.6f} < threshold: {threshold}",
                    'lrChanged': self.lrChanged
                })
                
                # Track consecutive decreases
                if self.lrChanged <= 0:
                    self.lrChanged -= 1
                else:
                    self.lrChanged = -1
                    
                if debug:print(f"Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
                return new_lr
        
        # Case 2: Strong improvement - consider increasing learning rate
        elif improvement > increase_threshold:
            # Check if we're at maximum learning rate already
            if current_lr >= max_lr * 0.999:
                # At maximum, try a small decrease to prevent divergence
                reduction_factor = factor * 10  # Less aggressive than before (was 50)
                new_lr = current_lr * (1 - reduction_factor)
                new_lr = max(new_lr, min_lr)
                
                self.log_change('lr_max_down', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'reduction_factor': reduction_factor,
                    'reason': f'max_limit_reached, preventing divergence',
                    'lrChanged': self.lrChanged
                })
                
                self.lrChanged = 0  # Reset change counter
                if debug:print(f"Max limit reached. Learning rate decreased: {current_lr:.6f} -> {new_lr:.6f}")
                return new_lr
            else:
                # Standard increase, but more conservative now
                # Adaptive factor - increase more conservatively if we're already at a high learning rate
                adaptive_factor = factor * 0.5 * (1 - (current_lr / max_lr) * 0.8)  # More conservative increase
                new_lr = current_lr * (1 + adaptive_factor)
                new_lr = min(new_lr, max_lr)
                
                self.log_change('lr_up', {
                    'before_lr': current_lr,
                    'new_lr': new_lr,
                    'change': new_lr - current_lr,
                    'adaptive_factor': adaptive_factor,
                    'reason': f"strong improvement: {improvement:.6f} > threshold: {increase_threshold}",
                    'lrChanged': self.lrChanged
                })
                
                # Track consecutive increases
                if self.lrChanged >= 0:
                    self.lrChanged += 1
                else:
                    self.lrChanged = 1
                    
                if debug:print(f"Learning rate increased: {current_lr:.6f} -> {new_lr:.6f} (improvement: {improvement:.6f})")
                return new_lr
        
        # Case 3: Moderate improvement - maintain current learning rate with tiny decay
        if debug:
            print(f"Learning rate only slightly decayed: {current_lr:.6f} -> {decayed_lr:.6f} (improvement: {improvement:.6f})")
        # Mevcut mantÄ±ÄŸÄ± cosine decay ile birleÅŸtir
        return max(decayed_lr * 0.99, min_lr)  # Ekstra yavaÅŸ decay

    def backpropagation(self,input_data, target_data,partName):


        # 1. Ä°leri Besleme - GiriÅŸ verilerini aÄŸa ver
        # GiriÅŸ katmanÄ±ndaki nÃ¶ron deÄŸerlerini ayarla
        for i, value in enumerate(input_data):
            if i < len(self.parts[partName].layers[0]):
                self.parts[partName].layers[0][i].value = value

        # Ä°leri besleme iÅŸlemi - tÃ¼m aÄŸÄ± hesapla

        self.parts[partName].runAI()

        # 2. Hata Hesaplama
        # Ã‡Ä±kÄ±ÅŸ katmanÄ±ndaki her nÃ¶ron iÃ§in hata hesapla
        output_layer = self.parts[partName].layers[-1]
        output_errors = []

        for i, neuron in enumerate(output_layer):
            if i < len(target_data):
                error = target_data[i] - neuron.value
                output_errors.append(error)
            else:
                output_errors.append(0)  # Hedef veri yoksa hata 0

        # 3. Geri YayÄ±lÄ±m
        # Her katman iÃ§in delta deÄŸerlerini hesapla (Ã§Ä±kÄ±ÅŸtan giriÅŸe doÄŸru)
        deltas = [[] for _ in range(len(self.parts[partName].layers))]

        # Ã–nce Ã§Ä±kÄ±ÅŸ katmanÄ±ndaki delta deÄŸerlerini hesapla
        for i, neuron in enumerate(output_layer):
            if i < len(output_errors):
                # Delta = Hata * Aktivasyon fonksiyonunun tÃ¼revi
                delta = output_errors[i] * neuron.activation_derivative()
                deltas[-1].append(delta)
            else:
                deltas[-1].append(0)

        # Gizli katmanlar iÃ§in delta deÄŸerlerini hesapla (geriye doÄŸru)
        for layer_idx in range(len(self.parts[partName].layers)-2, 0, -1):  # Son gizli katmandan ilk gizli katmana
            for i, neuron in enumerate(self.parts[partName].layers[layer_idx]):
                error = 0
                # Bu nÃ¶rondan sonraki katmana olan tÃ¼m baÄŸlantÄ±larÄ± kontrol et
                for conn in self.parts[partName].connections[layer_idx].get(neuron.id, []):
                    # Sonraki katmandaki nÃ¶ronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(self.parts[partName].layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # Bu baÄŸlantÄ±nÄ±n aÄŸÄ±rlÄ±ÄŸÄ± * sonraki nÃ¶ronun deltasÄ±
                            error += conn.weight * deltas[next_layer_idx][j]
                            break
                        
                # Delta = Hata * Aktivasyon fonksiyonunun tÃ¼revi
                delta = error * neuron.activation_derivative()
                deltas[layer_idx].append(delta)

        # 4. AÄŸÄ±rlÄ±k GÃ¼ncelleme
        for layer_idx in range(len(self.parts[partName].layers)-1):
            for i, neuron in enumerate(self.parts[partName].layers[layer_idx]):
                for conn in self.parts[partName].connections[layer_idx].get(neuron.id, []):
                    # Bir sonraki katmandaki baÄŸlÄ± nÃ¶ronu bul
                    next_layer_idx = layer_idx + 1
                    for j, next_neuron in enumerate(self.parts[partName].layers[next_layer_idx]):
                        if conn.connectedTo[1] == next_neuron.id:
                            # AÄŸÄ±rlÄ±k deÄŸiÅŸimi = Ã¶ÄŸrenme oranÄ± * delta * nÃ¶ron Ã§Ä±ktÄ±sÄ±
                            
                            if debug:
                                weight_change = self.learningRate * deltas[next_layer_idx][j] * neuron.value
                                old_weight = conn.weight
                                #print("debug values :",weight_change,self.learningRate,deltas[next_layer_idx][j],neuron.value)
                            # Mevcut Connection sÄ±nÄ±fÄ±nÄ±zdaki update_weight metodunu kullan
                            conn.update_weight(self.learningRate, deltas[next_layer_idx][j] * neuron.value)
                            break
                        
        # Toplam hata deÄŸerini hesapla ve dÃ¶ndÃ¼r (MSE - Mean Squared Error)
        total_error = sum(error**2 for error in output_errors) / len(output_errors) if output_errors else 0
        


        if debug:
            # AÄŸÄ±rlÄ±k gÃ¼ncellemelerini logla
            weight_updates = []
            for layer_idx in range(len(self.parts[partName].layers)-1):
                for i, neuron in enumerate(self.parts[partName].layers[layer_idx]):
                    for conn in self.parts[partName].connections[layer_idx].get(neuron.id, []):
                        weight_updates.append({
                            'from_neuron': conn.connectedTo[0],
                            'to_neuron': conn.connectedTo[1],
                            'old_weight': old_weight,  # GÃ¼ncellemeden Ã¶nceki aÄŸÄ±rlÄ±k
                            'new_weight': conn.weight,
                            'change': conn.weight-old_weight
                        },partName=partName)

            if weight_updates:
                self.log_change('weight_updates', {
                    'count': len(weight_updates),
                    'average_change': sum(abs(w['change']) for w in weight_updates) / len(weight_updates),
                    'updates': weight_updates[:10]  # Ä°lk 10 gÃ¼ncellemeyi gÃ¶ster (performans iÃ§in)
                },partName=partName)

        return total_error
        
    def adapt_neurons(self):
        pass
                
    
    def calculate_neuron_health(self,neuron,partName):
        layers= self.parts[partName].layers
        connections = self.parts[partName].connections
        # 1. NÃ¶ronun aktivasyon deÄŸeri
        activation_score = neuron.output

        # 2. BaÄŸlantÄ±larÄ±n aÄŸÄ±rlÄ±k deÄŸerlerini kontrol et
        weight_sum = 0
        weight_count = 0

        # Gelen baÄŸlantÄ±larÄ± bul
        for layer_idx in range(len(layers) - 1):
            for prev_neuron in layers[layer_idx]:
                for conn in connections[layer_idx].get(prev_neuron.id, []):
                    if conn.connectedTo[1] == neuron.id:
                        weight_sum += abs(conn.weight)  # Mutlak deÄŸer kullan
                        weight_count += 1

        # Giden baÄŸlantÄ±larÄ± bul (eÄŸer bu nÃ¶ron Ã§Ä±kÄ±ÅŸ katmanÄ±nda deÄŸilse)
        current_layer_idx = None
        for layer_idx, layer in enumerate(layers):
            if any(n.id == neuron.id for n in layer):
                current_layer_idx = layer_idx
                break
            
        if current_layer_idx is not None and current_layer_idx < len(layers) - 1:
            for conn in connections.get(current_layer_idx, {}).get(neuron.id, []):
                weight_sum += abs(conn.weight)
                weight_count += 1

        # Ortalama aÄŸÄ±rlÄ±k (eÄŸer baÄŸlantÄ± varsa)
        avg_weight = weight_sum / weight_count if weight_count > 0 else 0

        # 3. Aktivasyon tÃ¼revi - nÃ¶ronun ne kadar Ã¶ÄŸrenmeye aÃ§Ä±k olduÄŸunu gÃ¶sterir
        learning_potential = neuron.activation_derivative()

        # TÃ¼m faktÃ¶rleri birleÅŸtirerek bir saÄŸlÄ±k puanÄ± hesapla
        # Bu formÃ¼lÃ¼ kendi ihtiyaÃ§larÄ±nÄ±za gÃ¶re ayarlayabilirsiniz
        health_score = (
            0.4 * activation_score +  # Aktivasyon deÄŸerine %40 aÄŸÄ±rlÄ±k ver
            0.4 * avg_weight +        # Ortalama aÄŸÄ±rlÄ±ÄŸa %40 aÄŸÄ±rlÄ±k ver
            0.2 * learning_potential  # Ã–ÄŸrenme potansiyeline %20 aÄŸÄ±rlÄ±k ver
        )

        return health_score
    

        


    def find_neuron_layer(self,neuron_id,partName):
        """Verilen ID'ye sahip nÃ¶ronun hangi katmanda olduÄŸunu bulur"""
        layers=self.parts[partName].layers
        for layer_idx, layer in enumerate(layers):
            for neuron in layer:
                if neuron.id == neuron_id:
                    return layer_idx,neuron
        return None

    def add_neuron_to_layer(self,partName, layer_index=None, neuron_id=None):
        """
        Belirli bir katmana yeni bir nÃ¶ron ekler ve sadece bu nÃ¶ronla ilgili baÄŸlantÄ±larÄ± oluÅŸturur

        Args:
            layer_index: NÃ¶ronun ekleneceÄŸi katmanÄ±n indeksi (None ise neuron_id'nin katmanÄ± kullanÄ±lÄ±r)
            neuron_id: Referans nÃ¶ron ID'si (bunun katmanÄ±na yeni nÃ¶ron eklenir, layer_index None ise)
        """


        # EÄŸer layer_index verilmediyse ve neuron_id verildiyse, nÃ¶ronun katmanÄ±nÄ± bul
        if layer_index is None and neuron_id is not None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                if debug:print(f"Hata: ID'si {neuron_id} olan nÃ¶ron bulunamadÄ±.")
                return None

        # Hala layer_index belirlenemedi ise, varsayÄ±lan olarak son gizli katmanÄ± kullan
        if layer_index is None:
            if len(self.parts[partName].layers) <= 2:  # Sadece giriÅŸ ve Ã§Ä±kÄ±ÅŸ katmanlarÄ± varsa
                layer_index = 0  # GiriÅŸ katmanÄ±na ekle (veya tercih ettiÄŸiniz baÅŸka bir strateji)
            else:
                layer_index = len(self.parts[partName].layers) - 2  # Son gizli katman (Ã§Ä±kÄ±ÅŸ katmanÄ±ndan bir Ã¶nceki)

            if debug:
                print(f"Katman indeksi belirlenmediÄŸi iÃ§in varsayÄ±lan olarak katman {layer_index} kullanÄ±lÄ±yor.")

        if layer_index < 0 or layer_index >= len(self.parts[partName].layers):
            print(f"Hata: {layer_index} indeksi geÃ§erli bir katman indeksi deÄŸil.")
            return None

        # EÄŸer eklenen nÃ¶ron Ã§Ä±kÄ±ÅŸ katmanÄ±na aitse linear, deÄŸilse default aktivasyon
        activation_type = defaultOutActivation if layer_index == len(self.parts[partName].layers) - 1 else defaultNeuronActivationType

        # Yeni nÃ¶ron oluÅŸtur
        new_neuron = Neuron(activation_type=activation_type)

        # Yeni nÃ¶ronu katmana ekle
        self.parts[partName].layers[layer_index].append(new_neuron)

        # Ã–nceki katmandan bu nÃ¶rona baÄŸlantÄ±lar oluÅŸtur
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron in self.parts[partName].layers[prev_layer_idx]:
                weight = random.uniform(-1 / np.sqrt(len(self.parts[partName].layers[layer_index])), 
                                        1 / np.sqrt(len(self.parts[partName].layers[layer_index])))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in self.parts[partName].connections[prev_layer_idx]:
                    self.parts[partName].connections[prev_layer_idx][prev_neuron.id] = []

                self.parts[partName].connections[prev_layer_idx][prev_neuron.id].append(conn)

        # Bu nÃ¶rondan sonraki katmana baÄŸlantÄ±lar oluÅŸtur
        if layer_index < len(self.parts[partName].layers) - 1:
            for next_neuron in self.parts[partName].layers[layer_index + 1]:
                weight = random.uniform(-1 / np.sqrt(len(self.parts[partName].layers[layer_index])), 
                                        1 / np.sqrt(len(self.parts[partName].layers[layer_index])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)

                if new_neuron.id not in self.parts[partName].connections[layer_index]:
                    self.parts[partName].connections[layer_index][new_neuron.id] = []

                self.parts[partName].connections[layer_index][new_neuron.id].append(conn)

        if debug:
            print(f"Katman {layer_index}'e yeni nÃ¶ron (ID: {new_neuron.id}) eklendi.")




        return new_neuron


    def remove_neuron_from_layer(self,layer_index=None, neuron_id=None,partName=None):
        """
        Belirli bir nÃ¶ronu ve sadece onunla ilgili baÄŸlantÄ±larÄ± siler

        Args:
            layer_index: NÃ¶ronun bulunduÄŸu katmanÄ±n indeksi (None ise otomatik bulunur)
            neuron_id: Silinecek nÃ¶ronun ID'si
        """

        if neuron_id is None:
            print("Hata: Silinecek nÃ¶ronun ID'si belirtilmedi.")
            return False

        # EÄŸer layer_index verilmediyse, nÃ¶ronun katmanÄ±nÄ± bul
        if layer_index is None:
            layer_index = self.find_neuron_layer(neuron_id)
            if layer_index is None:
                print(f"Hata: ID'si {neuron_id} olan nÃ¶ron bulunamadÄ±.")
                return False

        if layer_index < 0 or layer_index >= len(self.parts[partName].layers):
            print(f"Hata: {layer_index} indeksi geÃ§erli bir katman indeksi deÄŸil.")
            return False

        # NÃ¶ronu bul
        neuron_to_remove = None
        for i, neuron in enumerate(self.parts[partName].layers[layer_index]):
            if neuron.id == neuron_id:
                neuron_to_remove = neuron
                neuron_index = i
                break
            
        if neuron_to_remove is None:
            print(f"Hata: ID'si {neuron_id} olan nÃ¶ron, katman {layer_index}'de bulunamadÄ±.")
            return False

        # NÃ¶ronu katmandan Ã§Ä±kar
        self.parts[partName].layers[layer_index].pop(neuron_index)

        # Ã–nceki katmandan bu nÃ¶rona gelen baÄŸlantÄ±larÄ± sil
        if layer_index > 0:
            prev_layer_idx = layer_index - 1
            for prev_neuron_id in list(self.parts[partName].connections[prev_layer_idx].keys()):
                self.parts[partName].connections[prev_layer_idx][prev_neuron_id] = [
                    conn for conn in self.parts[partName].connections[prev_layer_idx][prev_neuron_id] 
                    if conn.connectedTo[1] != neuron_id
                ]

        # Bu nÃ¶rondan sonraki katmana giden baÄŸlantÄ±larÄ± sil
        if layer_index < len(self.parts[partName].layers) - 1:
            if neuron_id in self.parts[partName].connections[layer_index]:
                del self.parts[partName].connections[layer_index][neuron_id]

        if debug:
            print(f"Katman {layer_index}'den nÃ¶ron (ID: {neuron_id}) silindi.")



        return True


    
    def adapt_network_structure(self, avg_train_error,partName, avg_val_error=None):
        """
        Dinamik yapÄ± adaptasyonu:
        - Overfitting kontrolÃ¼ ve gerekirse nÃ¶ron kÄ±rpma
        - NÃ¶ron seviyesinde optimizasyon
        - Katman seviyesinde optimizasyon
        - Stratejik bÃ¼yÃ¼me
        """
        # 0. Overfitting kontrolÃ¼
        if avg_val_error is not None:
            self.val_error_history.append(avg_val_error)
            gap = avg_val_error - avg_train_error
            if gap > self.overfit_threshold:
                self.prune_for_overfitting(gap,partName=partName)
                return

        # 1. NÃ¶ron seviyesinde optimizasyon
        self.neuron_level_optimization(avg_train_error,partName=partName)
        # 2. Katman seviyesinde optimizasyon
        self.layer_level_optimization(avg_train_error,partName=partName)
        # 3. Stratejik bÃ¼yÃ¼me
        self.strategic_growth(avg_train_error,partName=partName)

    def prune_for_overfitting(self, gap,partName):
        """
        AÅŸÄ±rÄ± Ã¶ÄŸrenme tespit edilirse, nÃ¶ron sayÄ±sÄ±nÄ± azaltarak basitleÅŸtirir
        """
        # Oran: gap / overfit_threshold, en fazla %50 azaltma
        factor = min(gap / self.overfit_threshold, 1.0) * 0.5
        for idx in range(1, len(self.parts[partName].layers) - 1):
            layer = self.parts[partName].layers[idx]
            current_size = len(layer)
            desired_size = max(2, int(current_size * (1 - factor)))
            to_remove = current_size - desired_size
            if to_remove > 0:
                # SaÄŸlÄ±k skoru dÃ¼ÅŸÃ¼k nÃ¶ronlarÄ± Ã¶ncelikli kaldÄ±r
                scores = [(neuron, self.calculate_neuron_health(neuron,partName=partName)) for neuron in layer]
                scores.sort(key=lambda x: x[1])
                for neuron, _ in scores[:to_remove]:
                    self.remove_neuron_from_layer(idx, neuron.id,partName=partName)
                    self.log_change('pruned_for_overfit', {
                        'neuron_id': neuron.id,
                        'layer': idx,
                        'gap': gap,
                        'reason': 'Overfitting pruning'
                    })
    
    def calculate_optimal_layer_sizes(self, input_size, output_size):
        """
        GiriÅŸ ve Ã§Ä±kÄ±ÅŸ boyutuna gÃ¶re optimal hidden layer boyutlarÄ±nÄ± hesaplar
        """
        # Temel kural: Hidden layer boyutu giriÅŸ ve Ã§Ä±kÄ±ÅŸÄ±n ortalamasÄ±ndan bÃ¼yÃ¼k olmalÄ±
        # ama Ã§ok bÃ¼yÃ¼k olmamalÄ±
        avg_size = (input_size + output_size) / 2
        max_size = int(avg_size * 2)    # ortalamanın 2 katını geçmesin
        min_size = int(avg_size * 0.8)  # ortalamanın %80’inden az olmasın


        # Ã‡ok bÃ¼yÃ¼k giriÅŸler iÃ§in (500+ gibi) farklÄ± kurallar
        if input_size > 100:
            min_size = input_size * 1.2  # GiriÅŸin %20 fazlasÄ±
            max_size = input_size * 2    # GiriÅŸin 2 katÄ±

        return {
            'min_hidden': int(min_size),
            'max_hidden': int(max_size),
            'recommended': int(min(max_size, max(min_size, avg_size * 1.5)))
        }

    def detect_excessive_neurons(self, layer_idx,partName):
        """
        Belirli bir katmandaki fazla nÃ¶ronlarÄ± tespit eder
        """

        if layer_idx == 0 or layer_idx == len(self.parts[partName].layers)-1:
            return []  # GiriÅŸ/Ã§Ä±kÄ±ÅŸ katmanlarÄ±nda optimizasyon yapma

        current_layer = self.parts[partName].layers[layer_idx]
        input_size = len(self.parts[partName].layers[layer_idx-1])
        output_size = len(self.parts[partName].layers[layer_idx+1]) if layer_idx+1 < len(self.parts[partName].layers) else 0

        optimal_sizes = self.calculate_optimal_layer_sizes(input_size, output_size)

        # EÄŸer katman boyutu makul sÄ±nÄ±rlardaysa hiÃ§bir ÅŸey yapma
        if (optimal_sizes['min_hidden'] <= len(current_layer) <= optimal_sizes['max_hidden']):
            return []

        # Fazla nÃ¶ronlarÄ± belirle
        if len(current_layer) > optimal_sizes['max_hidden']:
            # En az etkin nÃ¶ronlarÄ± bul
            neuron_healths = []
            for neuron in current_layer:
                health = self.calculate_neuron_health(neuron,partName=partName)
                neuron_healths.append((health, neuron))

            # SaÄŸlÄ±ÄŸa gÃ¶re sÄ±rala (en dÃ¼ÅŸÃ¼k saÄŸlÄ±klÄ± olanlar Ã¶nce)
            neuron_healths.sort(key=lambda x: x[0])

            # Fazla olan nÃ¶ronlarÄ± seÃ§
            excess_count = len(current_layer) - optimal_sizes['max_hidden']
            excess_neurons = [neuron for (health, neuron) in neuron_healths[:excess_count]]

            return excess_neurons

        return []

    def neuron_level_optimization(self, avg_error, partName):
        """
        Gelişmiş nöron seviyesinde optimizasyon:
        - Fazla nöronları kaldırır
        - Gereksiz nöronları temizler
        - Eksik nöronları ekler (max_hidden'i aşmadan)
        """
        layers = self.parts[partName].layers

        # 1. Katman boyutlarını optimize et (fazla nöron silme)
        for idx in range(1, len(layers)-1):
            excess_neurons = self.detect_excessive_neurons(idx, partName=partName)
            for neuron in excess_neurons:
                self.remove_neuron_from_layer(idx, neuron.id, partName=partName)
                self.log_change('neuron_removed', {
                    'neuron_id': neuron.id,
                    'layer': idx,
                    'reason': 'Excessive neuron count'
                }, partName=partName)

        # 2. Normal sağlık kontrolü (küçük sağlık tarihçesine bakarak)
        health_threshold = max(0.2, min(0.5, 0.3 * (1 + avg_error)))
        for idx, layer in enumerate(layers):
            if idx == 0 or idx == len(layers)-1:
                continue  # giriş ve çıkış katmanını atla
            for neuron in layer[:]:
                health = self.calculate_neuron_health(neuron, partName=partName)
                if health < health_threshold and len(self.neuron_health_history.get(neuron.id, [])) >= 3:
                    last_3 = self.neuron_health_history[neuron.id][-3:]
                    if all(h < health_threshold for h in last_3):
                        self.remove_neuron_from_layer(idx, neuron.id, partName=partName)
                        self.log_change('neuron_removed', {
                            'neuron_id': neuron.id,
                            'layer': idx,
                            'health': health,
                            'reason': f'Low health (<{health_threshold:.2f})'
                        }, partName=partName)

        # 3. Eksik nöronları ekle (ancak max_hidden'i aşmadan)
        for idx in range(1, len(layers)):
            is_output = (idx == len(layers)-1)
            input_size  = len(layers[idx-1])
            output_size = len(layers[idx+1]) if not is_output else 0

            optimal = self.calculate_optimal_layer_sizes(input_size, output_size)
            min_hidden = optimal['min_hidden']
            max_hidden = optimal['max_hidden']
            # Eğer yanlışlıkla min_hidden > max_hidden ise normalize et
            if min_hidden > max_hidden:
                min_hidden = max_hidden

            current_size = len(layers[idx])

            # Eğer çıktı katmanıysa, en az 1 nöron kuralı vs. farklı olabilir
            required_min = min_hidden if not is_output else max(1, output_size)

            # Eksikse ekle, ama kesinlikle max_hidden'i aşma
            if current_size < required_min:
                can_add = min(required_min - current_size, max_hidden - current_size)
                for _ in range(can_add):
                    activation = defaultOutActivation if is_output else defaultNeuronActivationType
                    new_neuron = Neuron(activation_type=activation)
                    layers[idx].append(new_neuron)
                    self.log_change('neuron_added', {
                        'neuron_id': new_neuron.id,
                        'layer': idx,
                        'reason': f'Layer too small (added to reach {required_min}, capped by max_hidden={max_hidden})'
                    }, partName=partName)

        # 4. Bağlantıları güncelle
        self.parts[partName].layers=layers
        self.parts[partName].setConnections(preserve_weights=True)



    def layer_level_optimization(self, avg_error, partName):
        """
        Katmanları değerlendirir ve gereksiz olanları kaldırır.
        1) Sağlık tabanlı silme
        2) Giriş-çıkış boyutlarına göre fazla katman silme
        """
        layers = self.parts[partName].layers

        # En az giriş + çıkış katmanı kalmalı
        if len(layers) <= 2:
            return

        # Sağlık skorlarını hesapla
        layer_health_scores = []
        for idx in range(1, len(layers) - 1):
            h = self.calculate_layer_health(idx, partName=partName)
            layer_health_scores.append((idx, h))
        layer_health_scores.sort(key=lambda x: x[1])
        worst_idx, worst_health = layer_health_scores[0]

        # Sağlık eşiğini belirle
        health_thresh = max(0.3, min(0.6, 0.4 * (1 + avg_error)))

        # 1) Sağlık tabanlı silme
        if worst_health < health_thresh and len(layers) > 3:
            self.remove_layer(worst_idx, partName=partName)
            self.log_change(
                'layer_removed',
                {
                    'layer_idx': worst_idx,
                    'health': worst_health,
                    'reason': f'Low layer health (<{health_thresh:.2f})'
                },
                partName=partName
            )
            return

        # 2) Giriş-çıkış boyutlarına göre fazla katman kontrolü
        input_neurons  = len(layers[0])
        output_neurons = len(layers[-1])
        # İzin verilen maksimum gizli katman sayısı
        max_hidden = max(1, (input_neurons + output_neurons) // 2)
        curr_hidden = len(layers) - 2

        if curr_hidden > max_hidden:
            # Fazla sayıda gizli katman var, en düşük sağlığa sahip katmanı sil
            self.remove_layer(worst_idx, partName=partName)
            self.log_change(
                'layer_removed',
                {
                    'layer_idx': worst_idx,
                    'health': worst_health,
                    'reason': (
                        f'Excess hidden layers: {curr_hidden} > {max_hidden} '
                        f'for input={input_neurons}, output={output_neurons}'
                    )
                },
                partName=partName
            )
            return

    
    def strategic_growth(self, avg_error,partName):
        """
        AÄŸÄ±n bÃ¼yÃ¼mesini stratejik olarak yÃ¶netir:
        - ZayÄ±f katmanlarÄ± gÃ¼Ã§lendirir
        - Kritik bÃ¶lgelere nÃ¶ron ekler
        - GerektiÄŸinde yeni katman ekler
        """
        # 1. ZayÄ±f katmanlarÄ± gÃ¼Ã§lendir
        self.strengthen_weak_layers(avg_error,partName=partName)
        
        # 2. Kritik bÃ¶lgelere nÃ¶ron ekle
        self.add_neurons_to_critical_areas(partName=partName)
        
        # 3. Gerekirse yeni katman ekle
        self.add_layer_if_needed(avg_error,partName=partName)
    
    def strengthen_weak_layers(self, avg_error,partName):
        """
        ZayÄ±f katmanlara nÃ¶ron ekler
        """
        
        
        complexity_factor = max(0.4, min(0.8, 0.5 * (1 + avg_error)))
        
        for layer_idx in range(1, len(self.parts[partName].layers)-1):  # Gizli katmanlar
            layer_health = self.calculate_layer_health(layer_idx,partName=partName)
            if layer_health < complexity_factor:
                # Katmana 1-2 nÃ¶ron ekle
                num_neurons_to_add = 1 if len(self.parts[partName].layers[layer_idx]) < 10 else 2
                for _ in range(num_neurons_to_add):
                    new_neuron = self.add_neuron_to_layer(layer_index=layer_idx,partName=partName)
                    self.log_change('neuron_added', {
                        'neuron_id': new_neuron.id,
                        'layer': layer_idx,
                        'reason': f'Strengthening weak layer (health={layer_health:.2f})'
                    },partName=partName)
    
    def add_neurons_to_critical_areas(self,partName):
        """
        YÃ¼ksek hata Ã¼reten veya yÃ¼ksek Ã¶ÄŸrenme potansiyeli olan bÃ¶lgelere nÃ¶ron ekler
        """
        
        # En yÃ¼ksek aktivasyon tÃ¼revine sahip nÃ¶ronun katmanÄ±na ekleme yap
        max_derivative = -1
        target_layer = None
        
        for layer_idx, layer in enumerate(self.parts[partName].layers):
            for neuron in layer:
                derivative = neuron.activation_derivative()
                if derivative > max_derivative:
                    max_derivative = derivative
                    target_layer = layer_idx
                    
        if target_layer is not None and target_layer < len(self.parts[partName].layers)-1 and target_layer != 0:
            new_neuron = self.add_neuron_to_layer(layer_index=target_layer)
            self.log_change('neuron_added', {
                'neuron_id': new_neuron.id,
                'layer': target_layer,
                'reason': f'High learning potential (derivative={max_derivative:.2f})'
            })
    
    def add_layer_if_needed(self, avg_error,partName):
        """
        AÄŸÄ±n karmaÅŸÄ±klÄ±ÄŸÄ± yeterli deÄŸilse yeni katman ekler
        """
        
        if len(self.parts[partName].layers) >= 5:  # Maksimum 5 katman (giriÅŸ + 3 gizli + Ã§Ä±kÄ±ÅŸ)
            return
            
        # Ortalama katman saÄŸlÄ±ÄŸÄ±nÄ± hesapla
        total_health = 0
        for layer_idx in range(1, len(self.parts[partName].layers)-1):
            total_health += self.calculate_layer_health(layer_idx,partName=partName)
        avg_health = total_health / (len(self.parts[partName].layers)-2) if len(self.parts[partName].layers) > 2 else 0
        
        complexity_factor = max(0.4, min(0.7, 0.5 * (1 + avg_error)))
        
        if avg_health < complexity_factor * 0.7:  # Katmanlar Ã§ok yÃ¼klÃ¼yse
            # En yÃ¼klÃ¼ katmanÄ± bul
            max_load = -1
            busiest_layer = None
            for layer_idx in range(1, len(self.parts[partName].layers)-1):
                load = len(self.parts[partName].layers[layer_idx]) * self.calculate_connection_density(layer_idx)
                if load > max_load:
                    max_load = load
                    busiest_layer = layer_idx
                    
            if busiest_layer is not None:
                new_layer_idx = self.insert_hidden_layer(busiest_layer + 1)  # MeÅŸgul katmandan sonra ekle
                self.log_change('layer_added', {
                    'layer_idx': new_layer_idx,
                    'reason': f'High layer load (load={max_load:.2f}, avg_health={avg_health:.2f})'
                })
    
    def calculate_layer_health(self, layer_idx,partName):
        """
        Bir katmanÄ±n genel saÄŸlÄ±k skorunu hesaplar
        """
        
        layer = self.parts[partName].layers[layer_idx]
        if not layer:
            return 0
            
        # Katmandaki nÃ¶ronlarÄ±n ortalama saÄŸlÄ±ÄŸÄ±
        total_health = sum(self.calculate_neuron_health(neuron,partName=partName) for neuron in layer)
        avg_neuron_health = total_health / len(layer)
        
        # KatmanÄ±n baÄŸlantÄ± yoÄŸunluÄŸu
        connection_density = self.calculate_connection_density(layer_idx,partName=partName)
        
        # KatmanÄ±n Ã¶ÄŸrenme potansiyeli (aktivasyon tÃ¼revlerinin ortalamasÄ±)
        learning_potential = np.mean([neuron.activation_derivative() for neuron in layer])
        
        # Katman saÄŸlÄ±k skoru
        layer_health = 0.5 * avg_neuron_health + 0.3 * connection_density + 0.2 * learning_potential
        
        return layer_health
    
    def calculate_connection_density(self, layer_idx,partName):
        """
        Katmandaki baÄŸlantÄ± yoÄŸunluÄŸunu hesaplar
        """
        
        if layer_idx == 0:  # GiriÅŸ katmanÄ±
            prev_layer_size = len(self.parts[partName].layers[layer_idx])
            current_layer_size = len(self.parts[partName].layers[layer_idx+1])
            total_possible = prev_layer_size * current_layer_size
        elif layer_idx == len(self.parts[partName].layers)-1:  # Ã‡Ä±kÄ±ÅŸ katmanÄ±
            return 1.0  # Ã‡Ä±kÄ±ÅŸ katmanÄ± iÃ§in maksimum yoÄŸunluk
        else:
            prev_layer_size = len(self.parts[partName].layers[layer_idx-1])
            current_layer_size = len(self.parts[partName].layers[layer_idx])
            next_layer_size = len(self.parts[partName].layers[layer_idx+1])
            total_possible = (prev_layer_size * current_layer_size) + (current_layer_size * next_layer_size)
        
        # GerÃ§ek baÄŸlantÄ± sayÄ±sÄ±nÄ± hesapla
        actual_connections = 0
        if layer_idx > 0:
            for conn_list in self.parts[partName].connections[layer_idx-1].values():
                actual_connections += len(conn_list)
        
        if layer_idx < len(self.parts[partName].layers)-1:
            for conn_list in self.parts[partName].connections[layer_idx].values():
                actual_connections += len(conn_list)
                
        return actual_connections / total_possible if total_possible > 0 else 0
    
    def remove_layer(self, layer_idx,partName):
        """
        Bir katmanÄ± ve iliÅŸkili baÄŸlantÄ±larÄ± kaldÄ±rÄ±r.
        Silme sonrasÄ± baÄŸlantÄ± sÃ¶zlÃ¼ÄŸÃ¼nÃ¼ tutarlÄ± hale getirmek iÃ§in
        setConnections ile yeniden Ã¶rÃ¼lÃ¼yoruz.
        """

        if layer_idx <= 0 or layer_idx >= len(self.parts[partName].layers)-1:
            print("Hata: GiriÅŸ veya Ã§Ä±kÄ±ÅŸ katmanÄ± silinemez")
            return False

        # KatmanÄ± kaldÄ±r
        del self.parts[partName].layers[layer_idx]

        # Eski connections anahtarlarÄ±nÄ± da temizle
        # (KeyError vermemesi iÃ§in get ile guardlÄ±yoruz)
        self.parts[partName].connections.pop(layer_idx-1, None)
        self.parts[partName].connections.pop(layer_idx,   None)

        # Kalan aÄŸÄ±rlÄ±klarÄ± koruyarak tÃ¼m baÄŸlantÄ±larÄ± yeniden inÅŸa et
        # BÃ¶ylece indekste kayma ya da eksik anahtar kalma riski ortadan kalkar
        self.parts[partName].setConnections(preserve_weights=True)

        if debug:
            print(f"Katman {layer_idx} silindi ve baÄŸlantÄ±lar yeniden oluÅŸturuldu.")
            

        return True

    
    def insert_hidden_layer(self, position,partName):
        """
        Belirtilen pozisyona yeni bir gizli katman ekler
        """

        if position <= 0 or position >= len(self.parts[partName].layers):
            print("Hata: GeÃ§ersiz katman pozisyonu")
            return False

        # Yeni katman oluÅŸtur (mevcut katmanlarÄ±n ortalamasÄ± kadar nÃ¶ronla)
        size = (len(self.parts[partName].layers[position - 1]) + len(self.parts[partName].layers[position])) // 2
        new_layer = [Neuron(activation_type=defaultNeuronActivationType) for _ in range(max(2, size))]

        # KatmanÄ± ekle
        layers.insert(position, new_layer)

        # connections sÃ¶zlÃ¼ÄŸÃ¼ne yeni boÅŸ dict alanÄ± ekle
        self.parts[partName].connections.insert(position - 1, {})
        self.parts[partName].connections.insert(position, {})

        # Ã–nceki katmandan yeni katmana baÄŸlantÄ±lar oluÅŸtur
        for prev_neuron in self.parts[partName].layers[position - 1]:
            for new_neuron in new_layer:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(self.parts[partName].layers[position - 1]) + len(new_layer)))
                conn = Connection(connectedToArg=[prev_neuron.id, new_neuron.id], weight=weight)

                if prev_neuron.id not in self.parts[partName].connections[position - 1]:
                    self.parts[partName].connections[position - 1][prev_neuron.id] = []

                self.parts[partName].connections[position - 1][prev_neuron.id].append(conn)

        # Yeni katmandan sonraki katmana baÄŸlantÄ±lar oluÅŸtur
        for new_neuron in new_layer:
            for next_neuron in self.parts[partName].layers[position + 1]:
                weight = np.random.uniform(-1, 1) * np.sqrt(2.0 / (len(new_layer) + len(self.parts[partName].layers[position + 1])))
                conn = Connection(connectedToArg=[new_neuron.id, next_neuron.id], weight=weight)

                if new_neuron.id not in self.parts[partName].connections[position]:
                    self.parts[partName].connections[position][new_neuron.id] = []

                self.parts[partName].connections[position][new_neuron.id].append(conn)

        return position
















































    class someAI:
        
        def __init__(self,partName,outerCorticalClass,visualizeNetwork=False,enable_logging=False,showMatplot=False,randomMinWeight = -2.0 , randomMaxWeight = 2.0 ,
                      layers:list = [2,6,4,3],connections:dict = {}):

            self.Neuron = Neuron
            self.Connection = Connection

            self.partName=partName
            self.outerCorticalClass = outerCorticalClass

            self.visualizeNetwork =visualizeNetwork
              # Global debug deÄŸiÅŸkeni
            self.enable_logging = enable_logging  # Loglama varsayÄ±lan olarak kapalÄ±
            self.showMatplot = showMatplot
            #cmd = "train_custom(veri.csv;2,5,2;0.0004)" #program baÅŸlar baÅŸlamaz Ã§alÄ±ÅŸacak ilk komut

            self.randomMinWeight = randomMinWeight
            self.randomMaxWeight = randomMaxWeight



            self.activation_types = ['sigmoid', 'tanh','linear','doubleSigmoid']

            self.layers = layers.copy()
            self.connections = connections    

            self.setLayers(self.layers)
            

            error_history = "s"
            # Global deÄŸiÅŸkenler
            self.error_history = []
            self.epoch_history = []
            self.learning_rate_history=[]
            self.start_time = None

            self.stopEpoch = False #ctrl C yapÄ±nca eÄŸitimi durdurmasÄ± iÃ§in

            self.bias_is=False

            
            print(f"{self.partName} bölümü oluşturuldu.")

        def setConnections(self,preserve_weights=True):
            # Eski aÄŸÄ±rlÄ±klarÄ± sakla
            old_weights = {}
            if preserve_weights:
                for layer_idx in self.connections:
                    for neuron_id in self.connections[layer_idx]:
                        for conn in self.connections[layer_idx][neuron_id]:
                            key = (layer_idx, neuron_id, conn.connectedTo[1])
                            old_weights[key] = conn.weight
            # Yeni baÄŸlantÄ±larÄ± oluÅŸtur
            new_connections = {layer_idx: {} for layer_idx in range(len(self.layers) - 1)}
            for layer_idx in range(len(self.layers) - 1):
                for neuron in self.layers[layer_idx]:
                    for next_neuron in self.layers[layer_idx + 1]:
                        key = (layer_idx, neuron.id, next_neuron.id)
                        if preserve_weights and key in old_weights:
                            # Eski aÄŸÄ±rlÄ±ÄŸÄ± koru
                            weight = old_weights[key]
                        else:
                            # Yeni aÄŸÄ±rlÄ±k oluÅŸtur
                            weight = random.uniform(-1/np.sqrt(len(self.layers[layer_idx])), 
                                            1/np.sqrt(len(self.layers[layer_idx])))
                        conn = Connection(connectedToArg=[neuron.id, next_neuron.id], weight=weight)
                        if neuron.id not in new_connections[layer_idx]:
                            new_connections[layer_idx][neuron.id] = []
                        new_connections[layer_idx][neuron.id].append(conn)
            self.connections = new_connections
        def setLayers(self,neuronInLayers):
            """KatmanlarÄ± ve nÃ¶ron sayÄ±larÄ±nÄ± ayarlar"""
            self.layers=[]  # Ã–nceki katmanlarÄ± temizle
            
            for layerIndex,neuronCount in enumerate(neuronInLayers):

                # Her katman iÃ§in yeni nÃ¶ron listesi oluÅŸtur
                layer = [
                    self.Neuron(
                        default_value=1,
                        activation_type=defaultOutActivation if layerIndex == len(neuronInLayers) - 1 else defaultNeuronActivationType
                    )
                    for _ in range(neuronCount)
                ]
                self.layers.append(layer)
            self.setConnections(preserve_weights=False)

        def createFileName(self,symbol=""):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"trainingDatas/{symbol}_network_{timestamp}.pkl.gz"

        def denormalize_value(self,norm_val, min_val, max_val):
            return ((norm_val + 1) / 3) * (max_val - min_val) + min_val

        def denormalize_value0_1(self,norm_x, min_val, max_val):
            """
            0-1 aralÄ±ÄŸÄ±ndaki bir deÄŸeri min_val ile max_val aralÄ±ÄŸÄ±na geri dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.
            """
            return norm_x * (max_val - min_val) + min_val





        def scale_value(self,x, source_min, source_max, target_min, target_max):
            """
            Bir deÄŸeri kaynak aralÄ±ktan hedef aralÄ±ÄŸa Ã¶lÃ§eklendirir.

            :param x: DÃ¶nÃ¼ÅŸtÃ¼rÃ¼lecek deÄŸer
            :param source_min: Kaynak aralÄ±ÄŸÄ±n alt sÄ±nÄ±rÄ±
            :param source_max: Kaynak aralÄ±ÄŸÄ±n Ã¼st sÄ±nÄ±rÄ±
            :param target_min: Hedef aralÄ±ÄŸÄ±n alt sÄ±nÄ±rÄ±
            :param target_max: Hedef aralÄ±ÄŸÄ±n Ã¼st sÄ±nÄ±rÄ±
            :return: Ã–lÃ§eklendirilmiÅŸ deÄŸer
            """
            return target_min + ((x - source_min) / (source_max - source_min)) * (target_max - target_min)


        def runAI(self):
            for layer in self.layers[1:]:
                for neuron in layer:
                    #print(f"NÃ¶ron {neuron.id}: {neuron.value}")
                    neuron.calculate_weighted_sum(self.layers,self.connections)
            #print(f"Son deÄŸer: {scale_value(get_neuron_by_id(30).value,0,1,0,8)}")
            lastNeuronValues =[]
            for neuron in self.layers[-1]:
                lastNeuronValues.append(neuron.value)
            return lastNeuronValues


        def save_network_optimized(self,filename=None,symbol=""):
            if filename is None:
                filename = self.createFileName(symbol=symbol)
            """YÃ¼ksek performanslÄ± binary kayÄ±t fonksiyonu"""
            network_data = {
                'layers': [
                    [
                        {
                            'id': n.id,
                            'value': n.value,
                            'output': n.output,
                            'weightedSum': n.weightedSum,
                            'activation_type': n.activation_type
                        } 
                        for n in layer
                    ] 
                    for layer in self.layers
                ],
                'connections': [
                    (layer_idx, conn.connectedTo[0], conn.connectedTo[1], conn.weight, conn.bias)
                    for layer_idx in self.connections
                    for neuron_id in self.connections[layer_idx]
                    for conn in self.connections[layer_idx][neuron_id]
                ],
                'config': (
                    self.randomMinWeight,
                    self.randomMaxWeight,
                    defaultNeuronActivationType,
                    defaultOutActivation,
                    self.visualizeNetwork

                ),
                'next_id': Neuron.next_id
            }

            with gzip.open(filename, 'wb') as f:
                pickle.dump(network_data, f, protocol=pickle.HIGHEST_PROTOCOL)


        def load_network_optimized(self,filename):
            """YÃ¼ksek performanslÄ± binary yÃ¼kleme fonksiyonu"""


            with gzip.open(filename, 'rb') as f:
                network_data = pickle.load(f)

            # Global deÄŸiÅŸkenleri gÃ¼ncelle
            (self.randomMinWeight, self.randomMaxWeight, defaultNeuronActivationType,defaultOutActivation, 
             self.visualizeNetwork) = network_data['config']
            self.Neuron.next_id = network_data['next_id']

            # KatmanlarÄ± yeniden oluÅŸtur
            self.layers = []
            for layer_data in network_data['layers']:
                layer = []
                for n_data in layer_data:
                    n = self.Neuron(activation_type=n_data['activation_type'])
                    n.id = n_data['id']
                    n.value = n_data['value']
                    n.output = n_data['output']
                    n.weightedSum = n_data['weightedSum']
                    layer.append(n)
                self.layers.append(layer)

            # BaÄŸlantÄ±larÄ± yeniden oluÅŸtur (defaultdict ile hÄ±zlÄ± eriÅŸim)
            self.connections = defaultdict(dict)
            for conn_data in network_data['connections']:
                layer_idx, from_id, to_id, weight, bias = conn_data
                if from_id not in self.connections[layer_idx]:
                    self.connections[layer_idx][from_id] = []
                self.connections[layer_idx][from_id].append(
                    self.Connection(weight=weight, connectedToArg=[from_id, to_id], bias=bias)
                )

            return True

        def change_weight(self,connections, from_id, to_id, new_weight):
            """
            Belirli bir baÄŸlantÄ±nÄ±n aÄŸÄ±rlÄ±ÄŸÄ±nÄ± deÄŸiÅŸtirir.

            :param connections: Katmanlar arasÄ± baÄŸlantÄ±lar
            :param from_id: BaÄŸlantÄ±dan gelen nÃ¶ronun ID'si
            :param to_id: BaÄŸlantÄ±ya giden nÃ¶ronun ID'si
            :param new_weight: Yeni aÄŸÄ±rlÄ±k
            """
            # connections dict'si Ã¼zerinden gezerek doÄŸru baÄŸlantÄ±yÄ± bulalÄ±m
            for layer_connections in self.connections.values():
                for neuron_id, conn_list in layer_connections.items():
                    for conn in conn_list:
                        # BaÄŸlantÄ± [from_id, to_id] olup olmadÄ±ÄŸÄ±nÄ± kontrol et
                        if conn.connectedTo == [from_id, to_id]:
                            conn.weight = new_weight  # Yeni aÄŸÄ±rlÄ±ÄŸÄ± gÃ¼ncelle
                            print(f"BaÄŸlantÄ± gÃ¼ncellendi: {from_id} -> {to_id} yeni weight: {new_weight}")
                            return  # Ä°ÅŸlem tamamlandÄ±ÄŸÄ±nda fonksiyonu sonlandÄ±r

            print(f"Hata: {from_id} ile {to_id} arasÄ±nda baÄŸlantÄ± bulunamadÄ±.")  # BaÄŸlantÄ± bulunamazsa hata mesajÄ±


        def get_neuron_by_id(self,neuron_id):
            for layer in self.layers:
                for neuron in layer:
                    if neuron.id == neuron_id:
                        return neuron
            return None  # EÄŸer nÃ¶ron bulunamazsa None dÃ¶ndÃ¼r


        def get_connections(self,layer_idx=None, detailed=False):
            """
            AÄŸdaki baÄŸlantÄ± bilgilerini dÃ¶ndÃ¼rÃ¼r.

            Parametreler:
            - layer_idx: Belirli bir katmanÄ±n baÄŸlantÄ±larÄ±nÄ± getir (None ise tÃ¼m katmanlar)
            - detailed: DetaylÄ± bilgi (kimden kime) ekler

            DÃ¶nÃ¼ÅŸ DeÄŸeri:
            - EÄŸer detailed=False: {layer_idx: {from_id: [weight1, weight2, ...]}}
            - EÄŸer detailed=True: {layer_idx: [(from_id, to_id, weight), ...]}
            """
            result = {}

            # TÃ¼m katmanlar iÃ§in
            if layer_idx is None:
                target_layers = self.connections.keys()
            else:
                if layer_idx not in self.connections:
                    print(f"UyarÄ±: {layer_idx}. katman bulunamadÄ±!")
                    return {}
                target_layers = [layer_idx]

            for l_idx in target_layers:
                layer_conns = self.connections[l_idx]

                if not detailed:
                    # Basit format: {from_id: [weight1, weight2, ...]}
                    simple_format = {}
                    for from_id, conn_list in layer_conns.items():
                        simple_format[from_id] = [conn.weight for conn in conn_list]
                    result[l_idx] = simple_format
                else:
                    # DetaylÄ± format: [(from_id, to_id, weight), ...]
                    detailed_format = []
                    for from_id, conn_list in layer_conns.items():
                        for conn in conn_list:
                            detailed_format.append((
                                from_id,
                                conn.connectedTo[1],
                                conn.weight
                            ))
                    result[l_idx] = detailed_format

            return result


        def get_neuron_connections(self,neuron_id, incoming=True, outgoing=True):
            """
            Returns list of Connection objects instead of tuples
            """
            found = []

            # Ã–nce nÃ¶ronun hangi katmanda olduÄŸunu bulalÄ±m
            neuron_layer = None
            for layer_idx, layer in enumerate(self.layers):
                for neuron in layer:
                    if neuron.id == neuron_id:
                        neuron_layer = layer_idx
                        break
                if neuron_layer is not None:
                    break
                
            if neuron_layer is None:
                print(f"UyarÄ±: {neuron_id} ID'li nÃ¶ron bulunamadÄ±!")
                return found

            # Gelen baÄŸlantÄ±lar (Ã¶nceki katmandan)
            if incoming and neuron_layer > 0:
                prev_layer_idx = neuron_layer - 1
                if prev_layer_idx in self.connections:
                    for from_id, conn_list in self.connections[prev_layer_idx].items():
                        for conn in conn_list:
                            if conn.connectedTo[1] == neuron_id:
                                found.append(conn)  # Return the Connection object itself

            # Giden baÄŸlantÄ±lar (sonraki katmana)
            if outgoing and neuron_layer < len(self.layers) - 1:
                current_layer_idx = neuron_layer
                if current_layer_idx in self.connections and neuron_id in self.connections[current_layer_idx]:
                    for conn in self.connections[current_layer_idx][neuron_id]:
                        found.append(conn)  # Return the Connection object itself

            return found



        def signal_handler(self,sig, frame):
            """Ctrl+C ile Ã§Ä±kÄ±ÅŸ yakalandÄ±ÄŸÄ±nda Ã§aÄŸrÄ±lacak fonksiyon"""

            print("\nEÄŸitim durduruldu.")
            self.stopEpoch =True
            if self.enable_logging:
                print("Veriler kaydediliyor...")

                #visualize_saved_errors(save_and_plot_errors())




        def klasor_hazirla(self,yol):
            """Verilen yol iÃ§in klasÃ¶r yapÄ±sÄ±nÄ± hazÄ±rlar"""
            try:
                os.makedirs(yol, exist_ok=True)
                print(f"KlasÃ¶r yapÄ±sÄ± hazÄ±r: {yol}")
                return True
            except Exception as e:
                print(f"Hata oluÅŸtu: {e}")
                return False


        def save_and_plot_errors(self):
            """Hata geÃ§miÅŸini kaydet ve gÃ¶rselleÅŸtir"""


            if not self.error_history:
                print("Kaydedilecek veri yok.")
                return

            # Ã‡Ä±ktÄ± dosyasÄ±nÄ±n adÄ±nÄ± belirle - sadece bir dosya oluÅŸtur
            outputFolder="trainingDatas/"
            self.klasor_hazirla(outputFolder)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_base = f"training_errors_{timestamp}"
            output_file_json = f"{output_file_base}.json"
            output_file_png = f"{output_file_base}.png"

            # Toplam eÄŸitim sÃ¼resini hesapla
            total_time = time.time() - self.start_time if self.start_time else 0

            # Veriyi JSON formatÄ±nda kaydet
            data = {
                "errors": self.error_history,
                "epochs": self.epoch_history,
                "total_time_seconds": total_time,
                "final_error": self.error_history[-1] if self.error_history else None,
                "learning_rates": self.learning_rate_history  # Bu yeni eklenen kÄ±sÄ±m
            }

            with open(outputFolder+output_file_json, 'w') as f:
                json.dump(data, f)

            print(f"Hata verileri {output_file_json} dosyasÄ±na kaydedildi.")




            return outputFolder+output_file_json






        def testModel(self,testFile: str, inputNum: int, targetNum: int,DONTVisualize=False):
            X, y = modeltrainingprogram.read_csv_file(testFile)
            fail = 0
            success = 0
            if debug or self.enable_logging:
                targetsAndOutputs=[]

            for a, i in enumerate(X):
                self.cmd_set_input(i)
                self.cmd_refresh(DONTVisualize=DONTVisualize)
                output, maxIndex, _ = self.getOutput()
                if debug or self.enable_logging: targetsAndOutputs.append([output[0],y[a][0]])
                if targetNum == 1:
                    # Tek bir Ã§Ä±ktÄ± varsa, kÃ¼Ã§Ã¼k bir toleransla eÅŸitlik kontrolÃ¼ yapÄ±lÄ±r
                    if abs(output[0] - y[a][0]) < 0.1:  # tolerans isteÄŸe baÄŸlÄ± ayarlanabilir
                        success += 1
                    else:
                        fail += 1
                else:
                    # Ã‡oklu Ã§Ä±kÄ±ÅŸta, en yÃ¼ksek skora sahip indeks kontrolÃ¼
                    if y[a][maxIndex] == max(y[a]):
                        success += 1
                    else:
                        fail += 1

            total = success + fail
            accuracy = (success / total) * 100 if total > 0 else 0
            if debug or self.enable_logging:
                for b in targetsAndOutputs:
                    print(b)
            print(f"BaÅŸarÄ±lÄ±: {success}, BaÅŸarÄ±sÄ±z: {fail}, DoÄŸruluk: %{accuracy:.2f}")
            return accuracy



    # Hata payÄ± fonksiyonu
        def hata_payi(self,target, output):
            # Listeleri numpy dizilerine dÃ¶nÃ¼ÅŸtÃ¼r
            target = np.array(target)
            output = np.array(output)
            return np.mean((target - output) ** 2)


        def train_network(self,X_train, y_train,corticalColumn, batch_size=1, epochs=None, intelligenceValue=None, learning_rate=0.05,useDynamicModelChanges=True,symbol="",epochNumberForLimitError=None,returnModelFile=False):


            if self.enable_logging:
                print("Hata grafiÄŸi kaydÄ± etkin. EÄŸitim sonunda grafik oluÅŸturulacak.")

            self.error_history = []
            self.epoch_history = []
            self.learning_rate_history = []
            newLR = 0.0

            signal.signal(signal.SIGINT, self.signal_handler)
            print("Debug:",type(corticalColumn))
            cortical_column = self.outerCorticalClass

            avg_error = float('inf')
            epoch = 0
            total_samples = len(X_train)
            self.start_time = time.time()

            if len(self.layers[0]) != len(X_train[0]):
                print(f"UyarÄ±: GiriÅŸ boyutu uyumsuz! AÄŸ giriÅŸi: {len(self.layers[0])}, Veri giriÅŸi: {len(X_train[0])}")
                return

            try:
                while True:
                    cortical_column.current_epoch = epoch
                    total_error = 0
                    processed_samples = 0
                    epoch_gradients = []
                    korteksChanges = []
                    newLR,_ = cortical_column.monitor_network(avg_error,partName=self.partName)
                    if _ is not None and epochs < 1 :
                        epochs = _


                    for batch_start in range(0, total_samples, batch_size):
                        batch_end = min(batch_start + batch_size, total_samples)
                        X_batch = X_train[batch_start:batch_end]
                        y_batch = y_train[batch_start:batch_end]
                        batch_error = 0

                        for X, y in zip(X_batch, y_batch):
                            cortical_column.backpropagation(X, y,partName=self.partName)

                            output = [neuron.value for neuron in self.layers[-1][:len(y)]]
                            error = self.hata_payi(y, output)

                            batch_error += error

                        batch_error /= len(X_batch)
                        total_error += batch_error * len(X_batch)
                        processed_samples += len(X_batch)
                        avg_error = total_error / processed_samples

                        # SADECE BATCH SONUNDA LOGLAMA
                        elapsed_time = time.time() - self.start_time
                        samples_per_sec = processed_samples / elapsed_time if elapsed_time > 0 else 0



                        #sys.stdout.write(f"\nEpoch {epoch+1}/{epochs if epochs is not None else 'âˆ'} - Ä°lerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%) - Ortalama Hata: {avg_error:.6f}")
                        #sys.stdout.flush()
                        if debug:
                            print(f"\nEpoch {epoch+1}/{epochs if epochs is not None else 'âˆ'} - Ä°lerleme: {processed_samples}/{total_samples} ({100*processed_samples/total_samples:.1f}%)")
                            print(f"Ortalama Hata: {avg_error:.6f}")



                    if self.enable_logging:
                            self.error_history.append(avg_error)
                            self.epoch_history.append(epoch + processed_samples/total_samples)
                            self.learning_rate_history.append(newLR)   


                    if ((epochs > 1 and epoch >= epochs) or (epochs <1 and epochs>avg_error)) or self.stopEpoch == True:
                        if epoch % 50 == 0 and debug:
                            cortical_column.log_change('epoch_summary', {
                                'average_error': avg_error,
                                'batch_progress': processed_samples/total_samples,
                                'partName':self.partName
                            },partName=self.partName)
                        break
                    
                    epoch += 1

                total_time = time.time() - self.start_time
                print(f"\n=== EÄÄ°TÄ°M TAMAMLANDI ===")
                print(f"Toplam SÃ¼re: {total_time/60:.1f} dakika | Toplam saniye: {total_time:.3f}")
                print(f"Son Hata: {avg_error:.6f}")
                print(f"Toplam Epoch: {epoch}")
                print(f"Final AÄŸ YapÄ±sÄ±: {[len(layer) for layer in self.layers]}")

                try:
                    filename =self.createFileName(symbol=symbol)
                    self.save_network_optimized(filename,symbol=symbol)
                    print(f"AÄŸ yapÄ±sÄ± {filename} dosyasÄ±na kaydedildi")
                except Exception as a:
                    print("Modeli dosyaya kaydetme sÄ±rasÄ±nda hata meydana geldi.")
                    traceback.print_exc()



                if self.enable_logging:
                    self.visualize_saved_errors(self.save_and_plot_errors())

            except KeyboardInterrupt:
                pass
            except Exception as e:
                if debug:
                    cortical_column.log_change('training_error', {
                        'error_type': str(type(e)),
                        'message': str(e),
                        'last_epoch': epoch,
                        'partName':self.partName
                    },partName=self.partName)
                raise
            if returnModelFile:
                return filename
            else:
                return cortical_column, avg_error 


        def visualize_saved_errors(self,filename, last_20Arg=0.8):
            """KaydedilmiÅŸ hata verilerini geliÅŸmiÅŸ grafiklerle gÃ¶rselleÅŸtir"""
            with open(filename, 'r') as f:
                data = json.load(f)

            errors = np.array(data["errors"])
            epochs = np.array(data["epochs"])
            learning_rates = np.array(data.get("learning_rates", [0.05]*len(epochs)))

            # Ana grafik
            plt.figure(figsize=(15, 10))

            # 1. Hata eÄŸrisi (ana grafik)
            ax1 = plt.subplot(2, 2, (1, 3))  # 2 satÄ±r, 2 sÃ¼tun, 1 ve 3'Ã¼ birleÅŸtir
            main_plot = plt.plot(epochs, errors, 'b-', linewidth=1, label='Ortalama Hata')
            scatter1 = plt.plot(epochs, errors, 'ro', markersize=1)[0]

            # EÄŸilim Ã§izgisi ekleme
            z = np.polyfit(epochs, errors, 3)
            p = np.poly1d(z)
            trend_line = plt.plot(epochs, p(epochs), "r--", linewidth=2, label='EÄŸilim Ã‡izgisi')[0]

            # DÃ¶nÃ¼m noktalarÄ±nÄ± bulma
            diff = np.diff(errors)
            turning_points = np.where(np.diff(np.sign(diff)))[0] + 1

            if len(turning_points) > 0:
                for tp in turning_points:
                    plt.plot(epochs[tp], errors[tp], 'go', markersize=2, label='DÃ¶nÃ¼m NoktasÄ±' if tp == turning_points[0] else "")

            # Grafik Ã¶zelleÅŸtirme
            plt.title('EÄŸitim SÄ±rasÄ±nda Hata DeÄŸiÅŸimi ve EÄŸilimi')
            plt.xlabel('Epoch')
            plt.ylabel('Ortalama Hata')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Son 20% epoch iÃ§in yakÄ±nlaÅŸtÄ±rÄ±lmÄ±ÅŸ grafik
            ax2 = plt.subplot(2, 2, 2)
            last_20 = int(len(epochs) * last_20Arg)
            line2 = plt.plot(epochs[last_20:], errors[last_20:], 'b-', linewidth=1.5)[0]
            scatter2 = plt.plot(epochs[last_20:], errors[last_20:], 'ro', markersize=2)[0]

            # Son bÃ¶lÃ¼m iÃ§in lineer regresyon
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                epochs[last_20:], errors[last_20:])
            reg_line = plt.plot(epochs[last_20:], intercept + slope*epochs[last_20:], 
                     'g--', linewidth=2, 
                     label=f'EÄŸim: {slope:.2e}\nRÂ²: {r_value**2:.2f}')[0]

            plt.title(f'Son %{int(100-last_20Arg*100)} Epoch YakÄ±nlaÅŸtÄ±rma')
            plt.xlabel('Epoch')
            plt.ylabel('Hata')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Learning Rate DeÄŸiÅŸimi GrafiÄŸi
            ax3 = plt.subplot(2, 2, 4)
            lr_line = plt.plot(epochs, learning_rates, 'm-', linewidth=1.5, label='Learning Rate')[0]
            lr_scatter = plt.plot(epochs, learning_rates, 'co', markersize=2)[0]

            # Learning rate iÃ§in eÄŸilim Ã§izgisi
            z_lr = np.polyfit(epochs, learning_rates, 1)
            p_lr = np.poly1d(z_lr)
            plt.plot(epochs, p_lr(epochs), "k--", linewidth=1, label=f'EÄŸilim: {z_lr[0]:.2e}x + {z_lr[1]:.2f}')

            plt.title('Learning Rate DeÄŸiÅŸimi')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Genel bilgiler
            stats_text = (
                f"BaÅŸlangÄ±Ã§ Hata: {errors[0]:.6f}\n"
                f"Son Hata: {errors[-1]:.6f}\n"
                f"En DÃ¼ÅŸÃ¼k Hata: {np.min(errors):.6f}\n"
                f"Ortalama Hata: {np.mean(errors):.6f}\n"
                f"Standart Sapma: {np.std(errors):.6f}\n"
                f"DÃ¶nÃ¼m NoktalarÄ±: {len(turning_points)}\n"
                f"BaÅŸlangÄ±Ã§ LR: {learning_rates[0]:.6f}\n"
                f"Son LR: {learning_rates[-1]:.6f}\n"
                f"Toplam Epoch: {len(epochs)}\n"
                f"Toplam SÃ¼re: {data['total_time_seconds']:.2f} sn"
            )

            plt.figtext(0.75, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.5), 
                        fontsize=9)

            plt.tight_layout()

            # GrafiÄŸi kaydet
            output_file = os.path.splitext(filename)[0] + "_advanced_viz.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')

            print(f"GeliÅŸmiÅŸ hata grafiÄŸi {output_file} dosyasÄ±na kaydedildi.")

            # EÄŸilim analizi
            self.analyze_trend(errors, epochs, last_20Arg=last_20Arg)

            # Fare etkileÅŸimi ekleme
            def format_annotation(sel):
                x, y = sel.target
                epoch = int(x)
                if sel.artist in [scatter1, scatter2]:  # Hata grafiklerindeki noktalar
                    error = y
                    sel.annotation.set_text(f"Epoch: {epoch}\nHata: {error:.6f}")
                elif sel.artist == lr_scatter:  # Learning rate grafiÄŸindeki noktalar
                    lr = y
                    sel.annotation.set_text(f"Epoch: {epoch}\nLearning Rate: {lr:.6f}")
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

            # TÃ¼m grafikler iÃ§in cursor ekle
            crs1 = mplcursors.cursor([scatter1, scatter2, lr_scatter], hover=True)
            crs1.connect("add", format_annotation)
            if self.showMatplot:
                if debug:
                    plt.show(block=True)
                else:
                    plt.show()

        def analyze_trend(self,errors, epochs,last_20Arg):
            """Hata eÄŸilimini analiz eder ve yorumlar"""
            # Son %(default 20)'lik kÄ±sÄ±m iÃ§in eÄŸim analizi

            last_20 = int(len(epochs) * last_20Arg)
            slope, _, _, _, _ = stats.linregress(epochs[last_20:], errors[last_20:])

            print("\n=== HATA EÄÄ°LÄ°M ANALÄ°ZÄ° ===")
            print(f"Son hata deÄŸeri: {errors[-1]:.6f}")
            print(f"Son %{100-last_20Arg*100} epoch'taki ortalama hata eÄŸimi: {slope:.2e}")

            if slope > 1e-6:
                print("UYARI: Hatalarda artÄ±ÅŸ eÄŸilimi var! Model overfitting olabilir veya Ã¶ÄŸrenme oranÄ± yÃ¼ksek olabilir.")
            elif slope < -1e-6:
                print("Hatalarda dÃ¼ÅŸÃ¼ÅŸ eÄŸilimi devam ediyor. EÄŸitime devam edilebilir.")
            else:
                print("Hatalar sabitlenmiÅŸ gÃ¶rÃ¼nÃ¼yor. Daha fazla eÄŸitimin faydasÄ± olmayabilir.")

            # YakÄ±nsama kontrolÃ¼
            last_10_errors = errors[-10:]
            std_last_10 = np.std(last_10_errors)
            if std_last_10 < 0.001:
                print(f"Hatalar yakÄ±nsamÄ±ÅŸ (son 10 epoch std: {std_last_10:.6f})")
            else:
                print(f"Hatalar henÃ¼z tam yakÄ±nsamadÄ± (son 10 epoch std: {std_last_10:.6f})")

            # Ã–neriler
            print("\n=== Ã–NERÄ°LER ===")
            if slope > 0 and len(errors) > 50:
                print("- Ã–ÄŸrenme oranÄ±nÄ± azaltmayÄ± deneyin")
                print("- Regularization ekleyin")
                print("- Early stopping uygulayÄ±n")
            elif slope < -1e-4:
                print("- Model hala Ã¶ÄŸreniyor, eÄŸitime devam edebilirsiniz")
            else:
                print("- Model performansÄ±nÄ± artÄ±rmak iÃ§in mimariyi deÄŸiÅŸtirmeyi deneyin")

            return slope #hatalardaki artÄ±ÅŸ eÄŸimi


        def getOutput(self):
            output_values = []
            max_value = -1
            max_index = -1

            # TÃ¼m Ã§Ä±ktÄ± nÃ¶ronlarÄ±nÄ± iÅŸle
            for i, neuron in enumerate(self.layers[-1]):
                value = neuron.value
                weighted_sum = neuron.weightedSum

                # En yÃ¼ksek aktivasyonu takip et
                if value > max_value:
                    max_value = value
                    max_index = i

                output_values.append(value)

            # En yÃ¼ksek aktivasyon bilgisini ekle




            return output_values,max_index,max_value



        def disable_all_biases(self):
            self.bias_is=False
            for layer_idx in self.connections:
                for neuron_id in self.connections[layer_idx]:
                    for conn in self.connections[layer_idx][neuron_id]:
                        conn.bias = 0

        # KullanÄ±m:

        def enable_all_biases(self):
            self.bias_is=True
            for layer_idx in self.connections:
                for neuron_id in self.connections[layer_idx]:
                    for conn in self.connections[layer_idx][neuron_id]:
                        conn.bias = random.uniform(-0.1, 0.1)  # Rastgele kÃ¼Ã§Ã¼k deÄŸerlerle yeniden baÅŸlat









        # Terminal giriÅŸ dÃ¶ngÃ¼sÃ¼ - Dinamik versiyon

        def removeNeuron(self,layer_index, neuron_index):
            """
            Belirtilen katmandan bir nöron kaldırır ve bağlantıları günceller.
            """


            if layer_index < 0 or layer_index >= len(self.layers):
                print(f"Geçersiz katman indexi: {layer_index}")
                return

            if neuron_index < 0 or neuron_index >= len(self.layers[layer_index]):
                print(f"Geçersiz nöron indexi: {neuron_index}")
                return

            removed_neuron = self.layers[layer_index].pop(neuron_index)

            # Tüm bağlantıları güncelle (ağırlıkları korumadan)
            self.setConnections(preserve_weights=False)

            if debug:
                print(f"Nöron silindi -> Katman: {layer_index}, Nöron ID: {removed_neuron.id}")

        def addNeuron(self,layer_index):
            """
            Belirtilen katmana bir adet yeni nöron ekler ve bağlantıları günceller.
            """


            if layer_index < 0 or layer_index >= len(self.layers):
                print(f"Geçersiz katman indexi: {layer_index}")
                return

            is_output_layer = (layer_index == len(self.layers) - 1)
            activation = defaultOutActivation if is_output_layer else defaultNeuronActivationType

            new_neuron = Neuron(default_value=1, activation_type=activation)
            self.layers[layer_index].append(new_neuron)

            # Tüm bağlantıları güncelle (ağırlıkları korumadan)
            self.setConnections(preserve_weights=True)

            if debug:
                print(f"Yeni nöron eklendi -> Katman: {layer_index}, Nöron ID: {new_neuron.id}")


        # Command functions

        def cmd_refresh(self,refresh=True,DONTVisualize=False):
            #Refresh the network and visualize
            self.runAI()
            return self.getOutput()





        def cmd_print_network(self):
            """Print network structure and connections"""
            output = []
            output.append("=== AÄ YAPISI ===")
            for i, layer in enumerate(self.layers):
                output.append(f"\nKatman {i} ({len(layer)} nÃ¶ron):")
                for neuron in layer:
                    output.append(f"  NÃ¶ron ID: {neuron.id} | DeÄŸer: {neuron.value:.4f} | Aktivasyon: {neuron.activation_type}")
            output.append("\n=== BAÄLANTILAR ===")
            for layer_idx, conn_dict in self.connections.items():
                output.append(f"\nKatman {layer_idx} -> Katman {layer_idx+1}:")
                for src_id, conn_list in conn_dict.items():
                    for conn in conn_list:
                        output.append(f"  {src_id} â†’ {conn.connectedTo[1]} | AÄŸÄ±rlÄ±k: {conn.weight:.4f}")
            return "\n".join(output)


        def cmd_get_connection(self,from_id: int, to_id: int) -> str:
            """Get connection weight between two neurons"""
            for layer_idx, conn_dict in self.connections.items():
                if from_id in conn_dict:
                    for conn in conn_dict[from_id]:
                        if conn.connectedTo[1] == to_id:
                            return f"BaÄŸlantÄ± bilgisi: {from_id} â†’ {to_id} | AÄŸÄ±rlÄ±k: {conn.weight:.6f}"
            return f"BaÄŸlantÄ± bulunamadÄ±: {from_id} â†’ {to_id}"


        def cmd_toggle_visualize(self) -> str:
            """Toggle network visualization"""

            self.visualizeNetwork = not self.visualizeNetwork
            return f"GÃ¶rselleÅŸtirme {'aktif' if self.visualizeNetwork else 'pasif'}"


        def cmd_bias(self,param: str) -> str:
            """Enable or disable or show biases"""

            if param == "True":
                self.enable_all_biases()
                self.bias_is = True
            elif param == "False":
                self.disable_all_biases()
                self.bias_is = False
            return f"Bias is : {self.bias_is}"


        def cmd_load_model(self,filepath: str) -> str:
            """Load model from file"""
            try:
                self.load_network_optimized(filepath)
                return "Model baÅŸarÄ±yla yÃ¼klendi."
            except Exception:
                traceback.print_exc()
                return "Model yÃ¼klenirken hata oluÅŸtu."



        def cmd_train_custom(self,file_path: str,
                             network_structure=None,
                             epochs=None,
                             batch_size=None,
                             learning_rate=None,
                             intelligenceValue=None,useDynamicModelChanges=True,symbol="",epochNumberForLimitError=None,returnModelFile=False,corticalColumn=None) -> str:
            """Train network with custom data"""
            try:
                self.setLayers(network_structure or [2,4,1])
                X, y = modeltrainingprogram.read_csv_file(file_path)
                train_kwargs = {}
                if epochs is not None:
                    train_kwargs['epochs'] = epochs
                if batch_size is not None:
                    train_kwargs['batch_size'] = batch_size
                if learning_rate is not None:
                    train_kwargs['learning_rate'] = learning_rate
                if intelligenceValue is not None:
                    train_kwargs['intelligenceValue'] = intelligenceValue
                filename = self.train_network(X, y, **train_kwargs,useDynamicModelChanges=useDynamicModelChanges,symbol=symbol,epochNumberForLimitError=epochNumberForLimitError,returnModelFile=returnModelFile,corticalColumn=corticalColumn)
                if returnModelFile:
                    return filename
                else:
                    return "Eğitim Tamamlandı."
            except Exception as e:
                traceback.print_exc()
                return f"Hata: {e}"

        def cmd_change_weight(self,from_id: int, to_id: int, new_weight: float) -> str:
            """Change connection weight"""
            try:
                self.change_weight(self.connections, from_id, to_id, new_weight)
                return f"BaÄŸlantÄ± aÄŸÄ±rlÄ±ÄŸÄ± gÃ¼ncellendi: {from_id} â†’ {to_id} = {new_weight:.4f}"
            except Exception as e:
                traceback.print_exc()
                return f"Hata: {e}"


        def cmd_change_neuron(self,id: int, new_value: float) -> str:
            """Change neuron value"""
            try:
                self.get_neuron_by_id(id).value = new_value
                return f"NÃ¶ron {id} deÄŸeri gÃ¼ncellendi: {new_value:.4f}"
            except Exception as e:
                traceback.print_exc()
                return f"Hata: {e}"



        def cmd_set_input(self,values: list) -> dict:
            """Set input layer values and return stats"""
            layer0 = self.layers[0]
            stats = {}
            try:
                # assign values
                for i, val in enumerate(values[:len(layer0)]):
                    layer0[i].value = val
                stats['min'] = float(np.min(values[:len(layer0)]))
                stats['max'] = float(np.max(values[:len(layer0)]))
                stats['mean'] = float(np.mean(values[:len(layer0)]))
                stats['values'] = values[:len(layer0)]
                return stats
            except Exception as e:
                traceback.print_exc()
                return {'error': str(e)}


file_path="parity_problem.csv"
network_structure=[4,1,1,1,1,2]


#def train_network(self,X_train, y_train,corticalColumn, batch_size=1, epochs=None, intelligenceValue=None, learning_rate=0.05,
#useDynamicModelChanges=True,symbol="",epochNumberForLimitError=None,returnModelFile=False):
epochs = 0.05
AI=CorticalColumn(learning_rateArg=0.05, targetError=epochs if epochs <1 else None,
                                             maxEpochForTargetError=None,
                                             originalNetworkModel=network_structure,useDynamicModelChanges=True,targetEpoch=None if epochs <1 else epochs)
AI.createPartOfAI("one")

filename=AI.parts["one"].cmd_train_custom(file_path=file_path,network_structure=network_structure,epochs=epochs,learning_rate=1,returnModelFile=True,corticalColumn=AI)
print(filename)

AI.parts["one"].cmd_load_model(filename)
AI.parts["one"].cmd_set_input([1,0,0,0])
print(AI.parts["one"].cmd_refresh())

#
#trainingModelFile = cmd_train_custom(file_path=file_path,network_structure=network_structure,epochs=0.1,learning_rate=1,returnModelFile=True)
##hata=testModel(file_path.replace("trainingDatas/", "trainingDatas/test"),inputNum=network_structure[0],targetNum=network_structure[-1],DONTVisualize=True)
#cmd_load_model(trainingModelFile)
#cmd_refresh(refresh=False)



