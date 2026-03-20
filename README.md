 🎯 CS-Mage: 
 ##  Video Multimodal Dialect Dataset： Sentiment Analysis + Speech Recognition 
<p align="center">
  <img src="https://github.com/user-attachments/assets/3ffb7f87-aa79-453e-926c-1cfaab9d5e9f" alt="CS-MSASR Dataset Preview" width="400"/>
</p>

---

### 📌 Background
### Base on Changsha city --- Representative dialects of southern China
Intelligent human-computer interaction systems should extend beyond mainstream languages to embrace **regional dialects**, which carry rich emotional nuances and cultural heritage. As the capital of Hunan Province, **Changsha** is home to the **Changsha dialect**—a prominent variety in southern China known for its **dynamic intonation** and **expressive vocabulary**. However, despite its linguistic value, the Changsha dialect remains **largely underrepresented** in existing Artificial Intelligence (AI) corpora.

This project aims to fill that gap by providing resources and tools tailored for the Changsha dialect, contributing to more inclusive and culturally aware AI systems.

---

### 📂 Dataset Overview

We introduce **CS-Mage**, the **first multimodal video dataset** for the Changsha dialect, aimed at **sentiment analysis** and **speech recognition** research.

- 🎥 **1085 video clips** covering diverse real-life scenarios
- 🗣️ Speakers ranging from **8 to 93 years old**, ensuring diversity
- 🧾 Each video is **manually transcribed** with authentic Changsha dialect text
- ❤️ **5 categories of multimodal sentiment labels**:
  - `Negative`
  - `Weakly Negative`
  - `Neutral`
  - `Weakly Positive`
  - `Positive`
- 🧠 **Unimodal sentiment annotations** for:
  - Text
  - Audio
  - Visual
- ✂️ **Fine-grained temporal segmentation**

---
### 🔗 Dataset

- **Google Drive**  
  https://drive.google.com/drive/folders/1g5zbyc6ZMVdqC95yfTl4lZZSIkK9V_E5?usp=drive_link

- **百度网盘**  
     https://pan.baidu.com/s/1lYznkyVZ0GsaDKosHb9fKQ
  提取码: 2cbi 

- The dataset after feature extraction is in the file CS-Mage_fulldata.pkl.
- The file contains all the data and does not differentiate between the training set, testing set and validation set. This is convenient for users to divide the dataset by themselves. If you want to synchronize with the article, please use 8:1:1 division, random_state=42, or contact the author.

---

### 📊 Benchmark

We evaluated:

- **11 mainstream multimodal sentiment analysis models**
- **8 speech recognition models** using:
  - Direct inference
  - Fine-tuning on CS-Mage

## Result speech recognition models

| Model | CER | WER |
| :--- | :---: | :---: |
| HMM-GMM | 133.44% | 133.44% |
| MFCC+SVM | 111.54% | 111.51% |
| Facebook/wav2vec2-large-960h | 99.88% | 1126.73% |
| Jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn | 82.06% | 116.22% |
| Whisper-base | 77.62% | 200.66% |
| Whisper-small | 138.32% | 138.32% |
| Whisper-small* | 48.36% | 48.36% |
| Paraformer-zh | 54.84% | 102.59% |
| Paraformer | 60.44% | 98.15% |
| Paraformer* | **34.24%** | **94.44%** |

> **Note:** The model with * is fine-tuned.
The experiments evaluate a range of speech recognition models, including traditional methods, pre-trained models and end-to-end architectures. We use Character Error Rate (CER) and Word Error Rate (WER) as the main evaluation metrics. Among all models, the Paraformer \cite{gao2022paraformer} series achieved the best performance. Among them, the fine-tuned Paraformer model (labeled as “*”) achieves a CER of 34.24\% and a WER of 94.44\% on CS- Mage, which is significantly better than the other models, indicating that it has strong modeling ability in adapting to dialectal speech, and verifying the advantages of non-autoregressive structure in dialect recognition. It also verifies the advantage of non-autoregressive structure in dialect recognition.
The pre-trained Transformer models show significant differences, depending on the language coverage of the model and whether it is adapted to the Chinese corpus. For example, the English corpus version, wav2vec2-large-960h \cite{baevski2020wav2vec} (CER: 99.88\%), can indicate that the model is completely wrong for the recognition task. Meanwhile, Whisper-small \cite{radford2023robust} predicted texts are presented as tokens, so WER is effectively equivalent to CER, and both are compared in terms of “words”. Whisper-small* \cite{radford2023robust} (fine-tuned version) outperforms its zero-sample version Whisper-base \cite{radford2023robust} (CER: 48.36\%), further indicating that the fine-tuning strategy for the target language is the key to improve the model's performance in the dialect recognition task.



---

### 📎 Citation

For detailed citation information, please refer to our [citations.json](https://github.com/Yaoooyu/CS-MSASR/blob/main/citations.json) file.

