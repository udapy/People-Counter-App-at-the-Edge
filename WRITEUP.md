# Project Write-Up
---
## Explaining Custom Layers
---
- Before building Intermediate Representation, Model Optimizer search each layer in inpul model inside given list of know layers. If supported layer is not found then that layer refered as Custom Layer.
- To support this unknown layer, they are registered as extension to model optimizer. Further, on registration they generate valid and optimized Intermidiate Representation.
- Every frameworks have different procedures to handle custom layers 
Refrences:
[Supported Framework Layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
[Custom Layers for different frameworks in the Model Optimizer](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)

### The process behind converting custom layers
- When handling custom layers for the model, add extensions to Model Optimizer and Inference Engine.
- For registerting custom layers, refer to these links 
[Custom Layers Guide](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html) | [Walk-through creating and executing a custom layer](https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md) 
- Whenever unsupported layers found during loading IR into Inference Engine, CPU extension is used.
CPU extension for linux
`/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_xxxx.so`

### Some of the potential reasons for handling custom layers
- Custom layers needs to handle becuase even if they are not supported they will be required at inference.
- Model Optimizer can't convert specific models to IR.
- For some edges cases, if developer tweaks some layers to solve that edge case then they will able to deploy model

## Comparing Model Performance
---
My method(s) to compare models before and after conversion to Intermediate Representations.

SSD based model are fast and small but accuracy is not good. Faster RCNN have good accuracy whereas they lack in speed. Intel pretrained & optimized models give better prediction. Hence, I choose Intel's OpenVINO Pre Optimised IR file i.e person-detection-retail-0013 to use in app.

|             Model             | Frozen Size | IR Size |
| :---------------------------: | :---------: | :-----: |
|     ssd_mobilenet_v2_coco     |   66.5MB    | 64.5MB  |
| faster_rcnn_inception_v2_coco |   54.5MB    | 50.8MB  |

Inferencing speed before optimization

|             Model             | Inference Speed |
| :---------------------------: | :-------------: |
|     ssd_mobilenet_v2_coco     |      87ms       |
| faster_rcnn_inception_v2_coco |      134ms      |

Inferencing speed and Frame per second after optimization 

|             Model             | Inference Speed | Frames per Second |
| :---------------------------: | :-------------: | :---------------: |
|     ssd_mobilenet_v2_coco     |      68ms       |         7         |
| faster_rcnn_inception_v2_coco |      862ms      |         1         |


The difference between model accuracy pre- and post-conversion

| Accuracy of pre-conversion model | Accuracy of post-conversion model |
| :------------------------------: | :-------------------------------: |
|             Moderate             |               Good                |

### The size of the model pre- and post-conversion
| Size of the frozen inference graph | Size of post conversion model |
| :--------------------------------: | :---------------------------: |
|               69.7MB               |            67.5MB             |

### The inference time of the model pre- and post-conversion

Inference time of the pre-conversion model

| Average  |   Min   |    Max    |
| :------: | :-----: | :-------: |
| 147.86ms | 87.42ms | 5971.23ms |

Inference time of the post-conversion model

| Average |  Min   |  Max   |
| :-----: | :----: | :----: |
|  3.4ms  | 0.28ms | 69.3ms |


## Assess Model Use Cases
---
Some of the potential use cases of the people counter app
-  During COVID pandemic for monitoring in quartine red zones, if people are gathering in more number than advice count.
    - Alerting authorities to if social distancing is followed or not.
- To monitor some suspicious activities, if some intruder is spending more time towards any banking kiosk
- Smart toddler monitoring system, so whenver toddler is toddle is not present in frame. parents will be alerted.

## Assess Effects on End User Needs
---
Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows

- Lighting -
Lighting condition is essential factor towards accuracy. Accuracy during low light conditions can be affected however this problem can be tackled with a high precision IR.
High precision models are slower. If speed and accuracy is higher priority lighting problem can reduced.
Low light cameras can be other solution but it will increase budget cost.

- Model Accuracy -
Edge models are working in real-time, model accuracy becomes high priority. 

- Camera Focal Length -
Angle of view of camera is based on camera focal length. Low focal length camera gives you wider angle, can useful in indoor applications & based on customer requirements. High focal length cameras are good outdoor application but model will extract less information about objects.

## Model Research
---
Model used for actual inference was from Intelss Model Zoo [person-detection-retail-0013]() pre-trained model.

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD MobileNet V2] (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  - I converted the model to an Intermediate Representation with the following argument
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o ./ir`
  - The model was insufficient for the app because it failed to detect boxes in specific time frames
  - I tried to improve the model for the app by lowering probablity threshold it didn't worked.
  
- Model 2: [Faster RCNN Inception V2]()
  - I converted the model to an Intermediate Representation with the following arguments
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json -o ./ir`
  - Recieved Segmentation faults

- Model 3: [SSD Inception V2]()
  - I converted the model to an Intermediate Representation with the following arguments
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json -o ./ir`
  - The model was insufficient for the app because detected people and trying to detect other things which leads to less accuracy
  - I tried to improve the model for the app by curating different probality threshold which leads to unsatisfactory results.
