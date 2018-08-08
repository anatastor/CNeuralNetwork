
#include "neuralnet.h"


void NeuralNet_print (NeuralNet *net)
{
    for (int i = 0; i < net->nLayer; i++)
    {
        for (int j = 0; j < net->layers[i].nNeurons; j++)
        {
            for (int k = 0; k < net->layers[i].neurons[j].nInputs; k++)
            {
                printf ("\ni: %i | j: %i | k: %i", i, j, k);
                printf (" | input: %lf | weight: %lf | output: %lf",
                        net->layers[i].neurons[j].inputs[k],
                        net->layers[i].neurons[j].weights[k],
                        *net->layers[i].neurons[j].output);
            }
        }
    }
    printf ("\n");
}


double NeuralNet_rand (void)
{
    return (double)rand()/RAND_MAX * 2.0 - 1.0;
}


double
NeuralNetNeuron_sigmoid (NeuralNetNeuron *neuron)
{
    return (1.0 / (1.0 + exp(-*neuron->output)));
}


NeuralNet *
NeuralNet_create (const int nInputs, double *netInputs, const int nLayer, int *nNeurons)
{   
    NeuralNet *net = malloc (sizeof(NeuralNet));
    if (! net)
        return NULL;

    net->nInputs = nInputs;
    net->nLayer = nLayer;
    net->netInputs = netInputs;
    
    // use members as counters and initialize them
    net->nNeurons = 0;
    net->nWeights = nInputs;
    for (int i = 0; i < net->nLayer; i++)
    {
        net->nNeurons += nNeurons[i];
        if (i > 0)
            net->nWeights += nNeurons[i] * nNeurons[i - 1];
    }
    
    // initialize weights
    net->weights = malloc (sizeof(double) * net->nWeights);
    net->oldWeights = malloc (sizeof(double) * net->nWeights);
    for (int i = 0; i < net->nWeights; i++)
    {
        net->weights[i] = NeuralNet_rand ();
        net->oldWeights[i] = 0.0; // todo: check if random is better
    }

    // create neurons and neuronOutputs
    net->neurons = malloc (sizeof(NeuralNetNeuron) * net->nNeurons);
    net->neuronOutputs = malloc (sizeof(double) * net->nNeurons);
    net->neuronErrors = malloc (sizeof(double) * net->nNeurons);

    
    // set the pointer to the outputs of the network, this way it is easier to handle later on
    //net->outputs = &net->neuronOutputs[net->nNeurons - net->layers[net->nLayer - 1].nNeurons];
    net->outputs = &net->neuronOutputs[net->nNeurons - nNeurons[net->nLayer - 1]];

    // initialize outputs and errors with 0 and set the neuron pointers
    for (int i = 0; i < net->nNeurons; i++)
    {
        net->neuronOutputs[i] = 0.0;
        net->neuronErrors[i] = 0.0;
    
        net->neurons[i].output = &net->neuronOutputs[i];
        net->neurons[i].error = &net->neuronErrors[i];
    }


    net->layers = malloc (sizeof(NeuralNetLayer) * net->nLayer); // create layers

    int neuronCounter = 0;
    int weightCounter = 0;
    int inputCounter = 0;

    for (int i = 0; i < net->nLayer; i++) // cycling the layers
    {
        net->layers[i].nNeurons = nNeurons[i];

        if (i == 0)
            net->layers[i].neurons = &net->neurons[0];
        else
            net->layers[i].neurons = net->layers[i - 1].neurons + net->layers[i].nNeurons;
        

        for (int j = 0; j < net->layers[i].nNeurons; j++) // cycling the neurons in one layer
        {
            if (i == 0)
            {
                net->layers[i].neurons[j].inputs = netInputs;
                net->layers[i].neurons[j].nInputs = nInputs;
            }
            else
            {
                net->neurons[neuronCounter].inputs = &net->neuronOutputs[inputCounter];
                // define the outputs of previous neuron as inputs to the next ones
                
                net->layers[i].neurons[j].nInputs = net->layers[i - 1].nNeurons;
            }

            net->neurons[neuronCounter].weights = &net->weights[weightCounter];
            net->neurons[neuronCounter].oldWeights = &net->oldWeights[weightCounter];

            neuronCounter++;
            weightCounter += net->layers[i].neurons->nInputs;
        }
        
        if (i > 0)
            inputCounter += net->layers[i - 1].neurons->nInputs;
    }

    return net;
}


void
NeuralNet_free (NeuralNet **net)
{   
    free ((*net)->netInputs);

    free ((*net)->neurons);
    free ((*net)->weights);
    free ((*net)->oldWeights);
    free ((*net)->neuronOutputs);
    free ((*net)->neuronErrors);

    for (int i = 0; i < (*net)->nLayer; i++)
        free ((*net)->layers[i].neurons);
    free ((*net)->layers);
    
    free (net);
}


void
NeuralNet_calculate (NeuralNet *net)
{
    for (int i = 0; i < net->nLayer; i++)
    {
        for (int j = 0; j < net->layers[i].nNeurons; j++)
        {
            *net->layers[i].neurons[j].output = 0;
            int nInputs = net->layers[i].neurons->nInputs;
            for (int k = 0; k < nInputs; k++)
            {
                *net->layers[i].neurons[j].output += net->layers[i].neurons[j].inputs[k] *
                    net->layers[i].neurons[j].weights[k];
                *net->layers[i].neurons[j].output = NeuralNetNeuron_sigmoid (&net->layers[i].neurons[j]);
            }
        }
    }
}


void
NeuralNet_train (NeuralNet *net, double *trainingIn, double *trainingOut, const int iterations)
{   
    // copy the data from the trainingInputSet to the input set
    for (int i = 0; i < net->nInputs; i++)
        net->netInputs[i] = trainingIn[i];
    
    for (int it = 0; it < iterations; it++)
    {
        NeuralNet_calculate (net);

        // calculate errors
        for (int i = net->nLayer - 1; i >= 0; i--)
        {
            if (i == net->nLayer - 1) // output Layer
            {
                for (int j = 0; j < net->layers[i].nNeurons; j++)
                {
                    *net->layers[i].neurons[j].error = pow (trainingOut[j] - *net->layers[i].neurons[j].output, 2.0);
                }
            }
            else
            {
                for (int j = 0; j < net->layers[i].nNeurons; j++)
                {
                    double temp = 0.0;
                    for (int k = 0; k < net->layers[i + 1].nNeurons; k++)
                    {
                        temp += *net->layers[i + 1].neurons[k].error * 
                            net->layers[i + 1].neurons[k].weights[j];
                    }
                    *net->layers[i].neurons[j].error = *net->layers[i].neurons[j].output * temp;
                }
            }
        }
    
        
        // update weights
        for (int i = net->nLayer - 1; i >= 0; i--)
        {
            for (int j = 0; j < net->layers[i].nNeurons; j++)
            {
                for (int k = 0; k < net->layers[i].neurons->nInputs; k++)
                {
                    double tempWeight = net->layers[i].neurons[j].weights[k];
                    /*
                    net->layers[i].neurons[j].weights[k] += (LEARNING_RATE * 
                            *net->layers[i].neurons[j].error *
                            net->layers[i].neurons[j].inputs[k]) + 
                        net->layers[i].neurons[j].weights[k] -
                        net->layers[i].neurons[j].oldWeights[k];
                    */
                    net->layers[i].neurons[j].weights[k] += LEARNING_RATE * 
                        *net->layers[i].neurons[j].error *
                        net->layers[i].neurons[j].inputs[k];
                    net->layers[i].neurons[j].oldWeights[k] = tempWeight;
                }
    
            }
        }

        if (it == 0 || it == iterations - 1)
            NeuralNet_print (net);
    }
}


int
NeuralNet_save (NeuralNet *net, const char *filename)
{
    FILE *file = fopen (filename, "w");
    if (file)
    {
        fprintf (file, "%i\n", net->nInputs);
        fprintf (file, "%i\n", net->nLayer);

        for (int i = 0; i < net->nLayer; i++)
            fprintf (file, "%i\n", net->layers[i].nNeurons);

        for (int i = 0; i < net->nNeurons; i++)
            fprintf (file, "%lf\n", net->weights[i]);
        
        return 1;
    }
    
    return 0;
}


NeuralNet *
NeuralNet_load (const char *filename, double **netInputs)
{
    FILE *file = fopen (filename, "r");
    if (file)
    {
        int nInputs;
        fscanf (file, "%i\n", &nInputs);

        int nLayer;
        fscanf (file, "%i\n", &nLayer);

        int *neurons = malloc (sizeof(int) * nLayer);

        for (int i = 0; i < nLayer; i++)
            fscanf (file, "%i\n", &neurons[i]);

        NeuralNet *net = NeuralNet_create (nInputs, *netInputs, nLayer, neurons);
        for (int i = 0; i < net->nNeurons; i++)
            fscanf (file, "%lf\n", &net->weights[i]);

    
        return net;
    }

    return NULL;
}
