# User-Empowered-Federated-Learning-in-Automotive
This repo contains the code and experiments of the paper "User-Empowered Federated Learning in Android" to appear in the first TRUSTCHAIN workshop co-located at the IEEE Blockchain 2024 conference \(Copenhagen, August 19-22, 2024\).

## Disclaimer
The provided code is a proof of concept and is not intended for production use. The code is provided as is. Please, feel free to reach out to the authors for any questions or clarifications. Pull requests are welcome after the presentation of the paper.

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
cp EngineFaultDB/EngineFaultDB_Final.csv .
rm -rf EngineFaultDB
```

## Prepare the dataset
```bash
cd utilities
cp ../EngineFaultDB_Final.csv .
python3 dataset_scaler.py
python3 dataset_splitter.py
```
The tflite model expects the features to be between 0 and 1. To achieve this, we use the MinMaxScaler from sklearn. After that, we  split the dataset into two training sets and one test set (40/40/20). The train_1.csv, train_2.csv, and test_1.csv files are generated in the utilities folder. These must be copied to the client app under the path client/app/src/main/assets/data. You can use the shorthand command `cp train_1.csv train_2.csv test_1.csv ../client/app/src/main/assets/data` to copy the files.

## Build the tflite model
```bash
cd utilities
python gen_tflite.py
```
Now you should have a enginefaultdb.tflite file in the gen_tflite folder. This must be copied to the client app under the path client/app/src/main/assets/model/enginefaultdb.tflite. You can use the shorthand command `cp enginefaultdb.tflite ../client/app/src/main/assets/model/enginefaultdb.tflite` to copy the file.

## Run the server
```bash
python server.py
```

## Run the client
The client is an Android Automotive app, please refer to the [Android Automotive documentation](https://developer.android.com/training/cars/testing/emulator) to run the app on an emulator.

## Reproduce the experiments
- Run the server
- Run three different clients, by setting the `DEVICE_ID` variable in the client app to 1, 2, and 3. This constant is found in the `TabScreen.kt` file.
- The clients will train the model for 10 local epochs, and the server will aggregate the models for 10 rounds.
- The main screen will show if an engine fault is detected in the car.

## Expected results
We set the number of local epochs to 10, the number of round to 10 and batch size of 16. We managed to obtain an accuracy of ~75% on the test set.

## Troubleshooting
- The generator for the tflite model requires tensorflow-cpu==2.9.2, if you have a different version of tensorflow installed, please create a virtual environment and install the required version. If your architecture does not support tensorflow-cpu (e.g. Apple Silicon), you can run the code on Colab.
- In MacOS there is a broken dependency in the client code (protoc). We will polish the solution soon.