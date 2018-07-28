using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using TensorFlow;

namespace Neuromatic
{
    /// <summary>
    /// Keeps track of information used during compilation of the model.
    /// This context is used to gather up trainable parameters and other
    /// information that is important for training and using models.
    /// </summary>
    public class ModelCompilationContext
    {
        private List<TFOutput> _parameters;
        private List<TFOperation> _initializers;

        /// <summary>
        /// Initializes a new instance of <see cref="ModelCompilationContext"/>
        /// </summary>
        /// <param name="graph">Graph to use for compiling the model</param>
        public ModelCompilationContext(TFGraph graph)
        {
            Graph = graph;
            _parameters = new List<TFOutput>();
            _initializers = new List<TFOperation>();
        }

        /// <summary>
        /// Gets the graph for building the final neural network structure.
        /// </summary>
        public TFGraph Graph { get; }

        /// <summary>
        /// Gets the trainable parameters gathered during compilation
        /// </summary>
        public IEnumerable<TFOutput> Parameters => _parameters;

        /// <summary>
        /// Gets the initializers for the parameters in the model
        /// </summary>
        public IEnumerable<TFOperation> Initializers => _initializers;

        /// <summary>
        /// Adds trainable parameters that need to be optimized
        /// </summary>
        /// <param name="parameters">Set of parameters to keep for optimization</param>
        public void AddParameters(params TFOutput[] parameters)
        {
            _parameters.AddRange(parameters);
        }

        /// <summary>
        /// Adds initializers to be run upon training
        /// </summary>
        /// <param name="initializers">Set of initializers to keep for running during the training phase</param>
        public void AddInitializers(params TFOperation[] initializers)
        {
            _initializers.AddRange(initializers);
        }

        /// <summary>
        /// Adds initializers to be run upon training
        /// </summary>
        /// <param name="initializers">Set of initializers to keep for running during the training phase</param>
        public void AddInitializers(IEnumerable<TFOperation> initializers)
        {
            _initializers.AddRange(initializers);
        }
    }
}
