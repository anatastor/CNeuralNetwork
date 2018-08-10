
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neuralnet.h"


int
main (int argc, char **argv)
{
    srand (time(NULL));
    int nNeurons[] = {5, 3, 2}; // number of neurons per layer
    double netInputs[2];
    NeuralNet *net = NeuralNet_create (2, netInputs, 3, nNeurons);

    double trainingIn[2];
    double trainingOut[2];
    NeuralNet_train (net, trainingIn, trainingOut, 100);

    NeuralNet_free (&net);

    return EXIT_SUCCESS;
}
