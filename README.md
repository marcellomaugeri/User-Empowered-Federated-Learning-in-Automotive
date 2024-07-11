# User-Empowered-Federated-Learning-in-Automotive
This repo contains the code and experiments of the paper "User-Empowered Federated Learning in Android" to appear in the first TRUSTCHAIN workshop co-located at the IEEE Blockchain 2024 conference \(Copenhagen, August 19-22, 2024\).

## Disclaimer
This repository *under restoration*, hence it is subject to change. The code is provided as is and is not guaranteed to work. Please, feel free to reach out to the authors for any questions or clarifications. Pull requests are welcome after the presentation of the paper.

## Abstract
The proliferation of data generated through everyday device usage has prompted privacy concerns among users.
In the automotive sector, this issue is particularly acute, given the substantial volumes of data collected in accordance with manufacturers' privacy policies. Privacy-Enhancing Technologies (PETs), such as Federated Learning (FL), offer a solution by safeguarding the confidentiality of car data while enabling decentralised machine learning model training, thus preventing the need for centralised data aggregation.

These FL-based models stand to benefit significantly from the diverse data distributions inherent in training across various features extracted from different cars.
However, it remains imperative to ensure user awareness regarding their data processing, despite FL's privacy-preserving mechanisms.

To address this, we propose a User-Empowered FL approach, built upon the Flower Framework, empowering users to decide their participation in model training or merely inference without impacting the global model.
We demonstrate this approach through an automotive case study utilising the EngineFaultDB dataset.

Finally, we outline future directions, particularly focusing on handling unlabelled data through self-supervised learning methodologies.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Download the EngineFaultDB dataset 
```bash
git clone https://github.com/Leo-Thomas/EngineFaultDB
cp EngineFaultDB/EngineFaultDB_Final.csv ./EngineFaultDB.csv
rm -rf EngineFaultDB
```

## Build the tflite model
```bash
cd gen_tflite
python gen_tflite.py
```
Now you should have a enginefaultdb.tflite file in the gen_tflite folder. This must be copied to the client app under the path client/app/src/main/assets/model/enginefaultdb.tflite

## Run the server
```bash
python server.py
```

## Run the client
The client is an Android Automotive app, please refer to the [Android Automotive documentation](https://developer.android.com/training/cars/testing/emulator) to run the app on an emulator.


## Troubleshooting
- The generator for the tflite model requires tensorflow-cpu==2.9.2, if you have a different version of tensorflow installed, please create a virtual environment and install the required version. If your architecture does not support tensorflow-cpu (e.g. Apple Silicon), you can run the code on Colab.
- In MacOS there is a broken dependency in the client code. We will polish the solution soon.