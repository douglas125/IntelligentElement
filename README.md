# IntelligentElement
Learning from hierarchical nested structures with Keras

## Introduction

Deep learning has seen success in many applications, from image recognition to translation. However, all models receive inputs from a single domain (image, text, audio etc).

Data from the real world is often a combination of multiple inputs in a nested structure containing data from multiple domains. For example: a webpage may describe textually "our favorite animal" while containing pictures of horses; a bank may have data from one client and also all of his past transaction history as well as the history of similar clients.

Our proposed Intelligent Element aims at automating the task of handling multiple nested structures from different domains. This means that the data scientist can focus on using the correct tool learn features from each domain and let the IntelligentElements do the feature aggregation.

## TODO

- [x] Create repository 
- [x] Initial exploration of model and data automation 
- [ ] Create Python abstract class
- [ ] Automate model creation
- [ ] Automate generation of data with Keras generator
- [ ] Create examples using toy datasets
- [ ] Apply to existing real datasets
