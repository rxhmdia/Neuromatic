using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace Neuromatic.Layers
{
    public class LayerConfiguration
    {
        /// <summary>
        /// Initializes a new instance of <see cref="LayerConfiguration"/>
        /// </summary>
        /// <param name="parameters">Trainable parameters for the layer</param>
        /// <param name="initializers">Initializers for the layer</param>
        /// <param name="output">Output node for the layer</param>
        public LayerConfiguration(IEnumerable<TFOutput> parameters, IEnumerable<TFOperation> initializers, TFOutput output)
        {
            Parameters = parameters;
            Initializers = initializers;
            Output = output;
        }

        /// <summary>
        /// Gets the parameters for the layer
        /// </summary>
        public IEnumerable<TFOutput> Parameters { get; }

        /// <summary>
        /// Gets the initializers for the layer
        /// </summary>
        public IEnumerable<TFOperation> Initializers { get; }

        /// <summary>
        /// Gets the output node for the layer
        /// </summary>
        public TFOutput Output { get; }
    }
}
