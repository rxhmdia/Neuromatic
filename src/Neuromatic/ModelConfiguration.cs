using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using TensorFlow;

namespace Neuromatic
{
    /// <summary>
    /// Stores all required configuration used for training a model and making predictions with a model
    /// </summary>
    public class ModelConfiguration
    {
        /// <summary>
        /// Initializes a new instance of <see cref="ModelConfiguration"/>
        /// </summary>
        /// <param name="parameters">Trainable parameters for the model</param>
        /// <param name="initializers">Initializers for the parameters of the model</param>
        /// <param name="modelLoss">Overall loss function for the model</param>
        /// <param name="inputMapping">The mapping between input layers and corresponding graph nodes</param>
        /// <param name="outputMapping">The mapping between output layers and corresponding graph nodes</param>
        public ModelConfiguration(IEnumerable<TFOutput> parameters, IEnumerable<TFOperation> initializers, Dictionary<Input, TFOutput> inputMapping, Dictionary<Layer, TFOutput> outputMapping)
        {
            Parameters = parameters;
            Initializers = initializers;
            InputMapping = inputMapping;
            OutputMapping = outputMapping;
        }

        /// <summary>
        /// Gets the computational graph for the model
        /// </summary>
        public TFGraph Graph { get; }

        /// <summary>
        /// Gets the initializers used for the parameters in the model
        /// </summary>
        public IEnumerable<TFOperation> Initializers { get; }

        /// <summary>
        /// Gets the trainable parameters of the model
        /// </summary>
        public IEnumerable<TFOutput> Parameters { get; }

        /// <summary>
        /// Gets the mapping between input layers and the corresponding computation graph nodes
        /// </summary>
        public IDictionary<Input, TFOutput> InputMapping { get; }

        /// <summary>
        /// Gets the mapping between output layers and the corresponding computation graph nodes
        /// </summary>
        public IDictionary<Layer, TFOutput> OutputMapping { get; }
    }
}
