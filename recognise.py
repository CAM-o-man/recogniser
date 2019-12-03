from imageai.Prediction import ImagePrediction
import os
import sys
execution_path = os.getcwd()
print(execution_path)
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( execution_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()


predictions, percentage_probabilities = prediction.predictImage(execution_path + "/" + sys.argv[1], result_count=5)
for index in range(len(predictions)):
    print(predictions[index] , " : " , percentage_probabilities[index])