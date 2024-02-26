SPAM FILTER 

binary classification model that takes messages as input and returns the label (ham or spam). The messages are divided into low risk and high risk messages. Low risk messages are displayed directly while high risk messages are moved to a folder for further investigatioon. 

Folder Structure Explanation: 
1. data (directory that stores all of the projects data)
2. raw_data (original data set, not preprocessed)
3. preprocessed_data (preprocessed data, ready for model training)
4. models (this directory stores the moodel output of high and low risk data)
5. high_risk (here all messages that were classified as spam or high risk are stored for further investitation)
6. low_risk (here all the other messages that are directly displayed are stored)
7. output (here all of the data is stored with a third column called "label". This third label was predicted by the model)
8. script (here the model script and the script for the testing part are stored)
