<div align="center">

# 😊 The Official Implementation of **TTA-FedDG: Leveraging Test-Time Adaptation to Address Federated Domain Generalization**

<p align="center">
    <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34053"><img src="https://img.shields.io/badge/AAAI-2025-003973?style=for-the-badge" alt="AAAI 2025"></a>
</p>

🎉🎉 **We have been accepted at AAAI-2025!**

If you like our project, please give us a star ⭐ on GitHub for the latest update.My coding skills are not very good, please bear with me and give me more advice.If you have any other questions, please contact me by email or WeChat.
</div>

---
This is the official code for the 《TTA-FedDG: Leveraging Test-Time Adaptation to Address Federated Domain Generalization》. 

![Overall](Fig/mFedDG.png)
<div align="center">
    
*Fig1: This illustrates the framework of FedSPL Leverages Test-Time Adaptation to Address Federated Domain Generalization.*

</div>

---

## 🚀 Get Started! (Take PACS as an example.)

### 🔧 Step 1: Create the Environment
``` bash
git clone https://github.com/liangno/TTA-FedDG.git
cd TTA_FedDG
conda create --name TTA_FedDG python=3.9
conda activate TTA_FedDG
pip install -r requirements.txt
```

### 📂 Step 2: Download the Dataset
- You can download the dataset at the link below：
   - [PACS](https://github.com/MachineLearning2020/Homework3-PACS)📦
   - [Office-Home](https://huggingface.co/datasets/flwrlabs/office-home)📦
   - [Digit-5](https://wjdcloud.blob.core.windows.net/dataset/dg5.tar.gz)📦

-You need to compile the dataset into a `.txt` file and put it in the corresponding datatset file.

### 💾 Step 3: Specify parameters and modify paths
-Our parameter and path modifications are mainly in `Fedspl/configs/default.py` and `/Fedspl/main_fedspl/configs`. I did not delete the original path for your convenience.

### ▶️ Step 4: Run!
- After completing the above basic settings, you only need one step to reproduce our method.
```bash
bash run_fedspl.sh
```

## 🙏 Acknowledgment
Our code is structurally referenced to [MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter) ,[FedDG-GA](https://github.com/MediaBrain-SJTU/FedDG-GA) and [C-SFDA](https://github.com/nazmul-karim170/C-SFDA). Thanks to their open-source spirit for saving us a lot of time. 💖


## 📕Reference

If you find this work helpful to your own work, please consider citing us:
```bash
@inproceedings{liang2025tta,
  title={TTA-FedDG: Leveraging Test-Time Adaptation to Address Federated Domain Generalization},
  author={Liang, Haoyuan and Zhang, Xinyu and Cao, Shilei and Li, Guowen and Zheng, Juepeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={18},
  pages={18658--18666},
  year={2025}
}
```
