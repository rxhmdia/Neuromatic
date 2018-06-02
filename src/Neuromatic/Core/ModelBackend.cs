using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Visitor used to compile the model to an executable graph
    /// </summary>
    public abstract class ModelBackend : IDisposable
    {
        /// <summary>
        /// Visits an input layer
        /// </summary>
        /// <param name="layer">Input layer to visit</param>
        public abstract ExecutableModelNode Placeholder(string name, long[] shape);

        /// <summary>
        /// Gets the output for the visitor
        /// </summary>
        public abstract ExecutableModel Output { get; }

        /// <summary>
        /// Locates a node in the graph that is managed by the backend
        /// </summary>
        /// <param name="name">Name of the model node</param>
        /// <returns>Returns the found model node. Returns null when the node could not be found.</returns>
        public abstract ExecutableModelNode Node(string name);

        /// <summary>
        /// Create a node that performs matrix multiplication
        /// </summary>
        /// <param name="left">Left node</param>
        /// <param name="right">Right node</param>
        /// <param name="name">Name of the node</param>
        /// <returns>Returns the matrix multiplication node</returns>
        public abstract ExecutableModelNode Dot(ExecutableModelNode left, ExecutableModelNode right, string name);

        /// <summary>
        /// Initializes a tensor with a normal distribution
        /// </summary>
        /// <param name="shape">Shape of the tensor</param>
        /// <param name="mean">Mean value for the distribution</param>
        /// <param name="standardDeviation">Standard deviation</param>
        /// <param name="seed">Random seed to use</param>
        /// <returns></returns>
        public abstract ExecutableModelNode RandomNormal(long[] shape, float mean = 0.0f, float standardDeviation = 0.05f, int? seed = null);

        /// <summary>
        /// Creates a new set of weights for the model
        /// </summary>
        /// <param name="initializer">Initializer to use</param>
        /// <param name="name">Name of the node</param>
        /// <returns>Returns the model node for the weights</returns>
        public abstract ExecutableModelNode Weights(long[] shape, InitializationFunction initializer, string name);

        /// <summary>
        /// Adds a bias term to the given model node
        /// </summary>
        /// <param name="node">Node to add the bias term to</param>
        /// <param name="bias">Bias term to add</param>
        /// <returns>Returns the new model node with the bias term added</returns>
        public abstract ExecutableModelNode BiasAdd(ExecutableModelNode node, ExecutableModelNode bias);

        /// <summary>
        /// Creates a new constant value
        /// </summary>
        /// <param name="value">Value for the constant</param>
        /// <param name="name">Name of the constant</param>
        /// <returns>Returns the node for the constant value</returns>
        public abstract ExecutableModelNode Constant(object value, long[] shape, string name);

        /// <summary>
        /// Creates a functions that adds two inputs
        /// </summary>
        /// <param name="a">First input</param>
        /// <param name="b">Second input</param>
        /// <param name="name">Name of the operation</param>
        /// <returns>Returns the node for the add operation</returns>
        public abstract ExecutableModelNode Add(ExecutableModelNode a, ExecutableModelNode b, string name);

        /// <summary>
        /// Creates a function bound to the backend
        /// </summary>
        /// <param name="defaultInputs">The default values for all placeholders in the function</param>
        /// <param name="outputs">The list of outputs to fetch as part of the function</param>
        /// <returns>Returns an executable function</returns>
        public abstract BackendFunction Function(IDictionary<ExecutableModelNode, object> defaultInputs, IEnumerable<ExecutableModelNode> outputs);

        /// <summary>
        /// Gets the initializers supported by the model backend
        /// </summary>
        public abstract Initializers Initializers { get; }

        /// <summary>
        /// Gets the activation functions supported by the model backend
        /// </summary>
        public abstract Activations Activations { get; }

        #region IDisposable Support

        /// <summary>
        /// Disposes any resources used by the backend
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {

        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            Dispose(true);
        }

        #endregion
    }
}
