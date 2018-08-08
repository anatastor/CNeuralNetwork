
#ifndef _NEURALNET_H_
#define _NEURALNET_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define LEARNING_RATE 1


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

/* prints the weights of the give network
 *
 * \param pointer to the neural network
 */
void NeuralNet_print (NeuralNet *net);

/* creates a random number between -1.0 and 1.0
 * to work properly srand () have to been called before
 */
double NeuralNet_rand (void);

/* returns the sigmoid of the output of the neuron */
double NeuralNetNeuron_sigmoid (NeuralNetNeuron *neuron);

/* creates a neural network
 *
 * \param number of inputs to the neural network
 * \param pointer to the inputs of the neural network
 * \param number of layers (including input and output layer)
 * \param pointer to an array of ints defining the number of neurons per layer
 *
 * returns a pointer to the created network or NULL on failure
 */
NeuralNet *NeuralNet_create (const int nInputs, double *netInputs, const int nLayer, int *nNeurons);

/* frees a neural network with all members */
void NeuralNet_free (NeuralNet **net);

/* calculates the output of the given network
 * the inputs have to be set by the main program via the previous given netInputs (see create)
 */
void NeuralNet_calculate (NeuralNet *net);

/* trains a neural network with the given training input and output with 
 * the number of iterations
 */
void NeuralNet_train (NeuralNet *net, double *trainingIn, double *trainingOut, const int iterations);

/* saves the weights, with some other important data, to the given filename */
int NeuralNet_save (NeuralNet *net, const char *filename);

/* loads the weights, and some other important data, from the given filename */
NeuralNet *NeuralNet_load (const char *filename, double **netInputs);


#endif /* _NEURALNET_H_ */
