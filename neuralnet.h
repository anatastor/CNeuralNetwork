
#ifndef _NEURALNET_H_
#define _NEURALNET_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define LEARNING_RATE 0.5


typedef struct NeuralNetNeuron
{
    int nInputs; // number of inputs
    double *inputs; // pointer to the begin of inputs
    double *weights; // pointer to the begin of weights
    double *oldWeights; 
    double *output; // pointer to the output
    double *error;  // pointer to the error
} NeuralNetNeuron;


typedef struct NeuralNetLayer
{
    int nNeurons;
    NeuralNetNeuron *neurons;
} NeuralNetLayer;


typedef struct NeuralNet
{
    int nInputs; // number of inputs to the neural network
    double *netInputs; // pointer to the input array
    double *outputs; // pointer to the outputs

    double *weights; // array of weights
    double *oldWeights;
    int nWeights; // number of weights

    int nNeurons; // number of neurons
    NeuralNetNeuron *neurons; // array of neurons
    double *neuronOutputs; // array of neuron outputs
    double *neuronErrors;  // array of the neuron errors

    int nLayer; // number of layers
    NeuralNetLayer *layers;
} NeuralNet;


double NeuralNet_rand (void);

double NeuralNetNeuron_sigmoid_derivate (NeuralNetNeuron *neuron);


NeuralNet *NeuralNet_create (const int nInputs, double *netInputs, const int nLayer, int *nNeurons);
void NeuralNet_free (NeuralNet **net);

void NeuralNet_calculate (NeuralNet *net);

void NeuralNet_train (NeuralNet *net, double *trainingIn, double *trainingOut, const int iterations);


#endif /* _NEURALNET_H_ */
