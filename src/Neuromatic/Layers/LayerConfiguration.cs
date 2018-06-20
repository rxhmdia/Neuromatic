using System;
using System.Collections.Generic;
using TensorFlow;

namespace Neuromatic.Layers
{
    /// <summary>
    /// Defines the configuration for a layer.
    /// This configuration is used to collect things like optimizable parameters and 
    /// initializers that should be ran at the beginning of the training process.
    /// </summary>
    public class LayerConfiguration
    {
        /// <summary>
        /// Initializes a new instance of <see cref="LayerConfiguration"/>
        /// </summary>
        /// <param name="parameters">Trainable parameters in the layer</param>
        /// <param name="output">Output of the layer</param>
        /// <param name="initializers">Initializes used in the layer</param>
        public LayerConfiguration(TFOutput[] parameters, TFOutput output, TFOperation[] initializers)
        {
            Parameters = parameters;
            Output = output;
            Initializers = initializers;
        }

        /// <summary>
        /// Gets the trainable parameters for the layer
        /// </summary>
        public TFOutput[] Parameters { get; }

        /// <summary>
        /// Gets the output of the layer
        /// </summary>
        public TFOutput Output { get; }

        /// <summary>
        /// Gets the initializers for the layer
        /// </summary>
        public TFOperation[] Initializers { get; }
    }
}
